import os
import requests
import base64
import json
import threading
import yt_dlp


from flask import Blueprint, request, jsonify
from db import reference_artists_collection
from google.cloud import storage
from googleapiclient.discovery import build

from gradio_client import Client, file

extraction_blueprint = Blueprint('extraction', __name__)
SPOTIFY_CLIENT_ID = os.environ.get('SPOTIFY_CLIENT_ID')
SPOTIFY_CLIENT_SECRET = os.environ.get('SPOTIFY_CLIENT_SECRET')
SPOTIFY_API_BASE_URL = 'https://api.spotify.com/v1'

key_json = os.environ.get('GOOGLE_CLOUD_KEY_JSON')
key_info = json.loads(key_json)
client = storage.Client.from_service_account_info(key_info)
bucket_name = 'opus-storage-bucket'
bucket = client.bucket(bucket_name)
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')


@extraction_blueprint.route('/top-song', methods=['GET'])
def get_top_song():
    artist_name = request.args.get('artist_name')
    if not artist_name:
        return 'Please provide an artist name', 400

    SPOTIFY_API_TOKEN = get_access_token()
    headers = {
        'Authorization': f'Bearer {SPOTIFY_API_TOKEN}'
    }
    search_query = f'artist:{artist_name}'
    search_params = {
        'q': search_query,
        'type': 'artist',
        'limit': 1
    }

    print("searching for artist")
    artist_id = search_for_artist(headers, search_params, artist_name)
    if not artist_id:
        return f'No artist found with the name "{artist_name}"', 404

    threading.Thread(target=process_top_song, args=(artist_name, artist_id)).start()
    return jsonify({'artist_id': artist_id}), 200


def process_top_song(artist_name, artist_id):
    SPOTIFY_API_TOKEN = get_access_token()
    headers = {
        'Authorization': f'Bearer {SPOTIFY_API_TOKEN}'
    }

    print(f"getting top track for {artist_name}")
    top_track_preview_url, top_track = get_top_track(headers, artist_id, artist_name)
    preview_response = requests.get(top_track_preview_url)

    print(f"processing split and upload for {top_track['name']}")
    process_split_and_upload(artist_name, artist_id, top_track, preview_response)


def search_for_artist(headers, search_params, artist_name):
    search_response = requests.get(
        f'{SPOTIFY_API_BASE_URL}/search',
        headers=headers,
        params=search_params
    )
    if search_response.status_code != 200:
        return f'Error fetching artist data: {search_response.text}', 500
    search_data = search_response.json()
    if not search_data['artists']['items']:
        return f'No artist found with the name "{artist_name}"', 404
    artist_id = search_data['artists']['items'][0]['id']

    return artist_id


def get_top_track(headers, artist_id, artist_name):
    top_tracks_response = requests.get(
        f'{SPOTIFY_API_BASE_URL}/artists/{artist_id}/top-tracks',
        headers=headers,
    )
    if top_tracks_response.status_code != 200:
        return f'Error fetching top tracks data: {top_tracks_response.text}', 500
    top_tracks_data = top_tracks_response.json()
    if not top_tracks_data['tracks']:
        return f'No top tracks found for the artist "{artist_name}"', 404

    # Iterate until we find the first solo track with a preview URL
    top_track = None
    for track in top_tracks_data['tracks']:
        if track['preview_url'] is not None:
            top_track = track
            break

    if top_track is None:
        query = f"{artist_name} official audio"
        video_id = search_youtube(query)

        if video_id:
            public_url = download_from_youtube(video_id, artist_name)
            return public_url, {"name": "Youtube video"}

        else:
            return 'No preview URL found for any of the top tracks', 404

    return top_track['preview_url'], top_track


def process_split_and_upload(artist_name, artist_id, top_track, preview_response):
    hb_client = Client("r3gm/Audio_separator")

    rawblob = bucket.blob(f"reference-artist-audios/{artist_name}/raw/{top_track['name']}.mp3")
    rawblob.upload_from_string(preview_response.content)

    split_result = hb_client.predict(
        media_file=file(rawblob.public_url),
        stem="vocal",
        main=False,
        dereverb=False,
        api_name="/sound_separate"
    )
    voice_stem_path = split_result[0]
    with open(voice_stem_path, 'rb') as source1_file:
        source1_blob = bucket.blob(f"reference-artist-audios/{artist_name}/split/{top_track['name']}-vocal.wav")
        source1_blob.upload_from_file(source1_file)

    if reference_artists_collection.find_one({'spotifyArtistId': artist_id}) is None:
        reference_artists_collection.insert_one({
            'name': artist_name,
            'spotifyArtistId': artist_id,
            'audioStemUrl': source1_blob.public_url
        })

    return "Splitting and uploading successful"


def get_access_token():
    auth_string = f"{SPOTIFY_CLIENT_ID}:{SPOTIFY_CLIENT_SECRET}"
    auth_bytes = auth_string.encode("utf-8")
    auth_base64 = str(base64.b64encode(auth_bytes), "utf-8")

    headers = {
        "Authorization": f"Basic {auth_base64}"
    }

    data = {
        "grant_type": "client_credentials"
    }

    token_response = requests.post("https://accounts.spotify.com/api/token", headers=headers, data=data)

    if token_response.status_code != 200:
        raise Exception(f"Failed to get access token: {token_response.text}")

    token_data = token_response.json()
    access_token = token_data["access_token"]
    return access_token


def search_youtube(query):
    youtube = build('youtube', 'v3', developerKey=GOOGLE_API_KEY)
    search_response = youtube.search().list(
        q=query,
        type='video',
        part='id,snippet',
        maxResults=1
    ).execute()

    if search_response.get('items'):
        video_id = search_response['items'][0]['id']['videoId']
        return video_id
    else:
        return None


def download_from_youtube(video_id, artist_name):
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': 'track-name.%(ext)s',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192'
        }]
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(f'https://www.youtube.com/watch?v={video_id}', download=True)
        print(f"Downloaded YouTube video as: {info}")

    blob_name = f"reference-artist-audios/{artist_name}/raw/youtube-track.mp3"
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(downloaded_filename)

    blob.make_public()
    return blob.public_url