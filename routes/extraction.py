import os
import requests
import base64
import json
import asyncio

from flask import Blueprint, request
from db import reference_artists_collection
from google.cloud import storage
from gradio_client import Client

extraction_blueprint = Blueprint('extraction', __name__)
SPOTIFY_CLIENT_ID = os.environ.get('SPOTIFY_CLIENT_ID')
SPOTIFY_CLIENT_SECRET = os.environ.get('SPOTIFY_CLIENT_SECRET')
SPOTIFY_API_BASE_URL = 'https://api.spotify.com/v1'

key_json = os.environ.get('GOOGLE_CLOUD_KEY_JSON')
key_info = json.loads(key_json)
client = storage.Client.from_service_account_info(key_info)
bucket_name = 'opus-storage-bucket'
bucket = client.bucket(bucket_name)


@extraction_blueprint.route('/top-song', methods=['GET'])
def get_top_song():
    asyncio.run(process_top_song())
    return "Splitting successfully underway. Please wait for completion."


async def process_top_song():
    SPOTIFY_API_TOKEN = get_access_token()
    artist_name = request.args.get('artist_name')
    if not artist_name:
        return 'Please provide an artist name', 400
    headers = {
        'Authorization': f'Bearer {SPOTIFY_API_TOKEN}'
    }
    search_query = f'artist:{artist_name}'
    search_params = {
        'q': search_query,
        'type': 'artist',
        'limit': 1
    }

    artist_id = await search_for_artist(headers, search_params, artist_name)

    top_track_preview_url, top_track = await get_top_track(headers, artist_id, artist_name)
    preview_response = requests.get(top_track_preview_url)

    await process_split_and_upload(artist_name, artist_id, top_track, preview_response)


async def search_for_artist(headers, search_params, artist_name):
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


async def get_top_track(headers, artist_id, artist_name):
    top_tracks_response = requests.get(
        f'{SPOTIFY_API_BASE_URL}/artists/{artist_id}/top-tracks',
        headers=headers,
    )
    if top_tracks_response.status_code != 200:
        return f'Error fetching top tracks data: {top_tracks_response.text}', 500
    top_tracks_data = top_tracks_response.json()
    if not top_tracks_data['tracks']:
        return f'No top tracks found for the artist "{artist_name}"', 404

    # iterate until get top track with preview URL
    top_track = next((track for track in top_tracks_data['tracks'] if track['preview_url']), None)
    if top_track is None:
        return f'No top track found with preview URL for the artist "{artist_name}"', 404

    return top_track['preview_url'], top_track['name']


async def process_split_and_upload(artist_name, artist_id, top_track, preview_response):
    hb_client = Client("https://younver-speechbrain-speech-separation.hf.space/--replicas/lp1ql/")

    # save raw track
    rawblob = bucket.blob(f"reference-artist-audio/{artist_name}/raw/{top_track['name']}.mp3")
    rawblob.upload_from_string(preview_response.content)

    # splitting
    splitResult = hb_client.predict(rawblob.public_url, api_name="/predict")
    print(splitResult)

    if reference_artists_collection.find_one({'spotifyArtistId': artist_id}) is None:
        reference_artists_collection.insert_one({
            'spotifyArtistId': artist_id,
            'artistName': artist_name
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

