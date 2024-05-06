import os
import requests
import base64

from flask import Blueprint, request
from spotdl import Spotdl

from db import reference_artists_collection

extraction_blueprint = Blueprint('extraction', __name__)
SPOTIFY_CLIENT_ID = os.environ.get('SPOTIFY_CLIENT_ID')
SPOTIFY_CLIENT_SECRET = os.environ.get('SPOTIFY_CLIENT_SECRET')
SPOTIFY_API_BASE_URL = 'https://api.spotify.com/v1'


@extraction_blueprint.route('/top-song', methods=['GET'])
def get_top_song():
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

    if reference_artists_collection.find_one({'spotifyArtistId': artist_id}) is None:
        reference_artists_collection.insert_one({
            'spotifyArtistId': artist_id,
            'artistName': artist_name
        })

    top_tracks_response = requests.get(
        f'{SPOTIFY_API_BASE_URL}/artists/{artist_id}/top-tracks',
        headers=headers,
    )

    if top_tracks_response.status_code != 200:
        return f'Error fetching top tracks data: {top_tracks_response.text}', 500

    top_tracks_data = top_tracks_response.json()
    if not top_tracks_data['tracks']:
        return f'No top tracks found for the artist "{artist_name}"', 404

    top_track = top_tracks_data['tracks'][0]

    top_track_mp3 = download_mp3(top_track['external_urls']['spotify'])
    return {
        'artist_name': artist_name,
        'top_track_name': top_track['name'],
        'top_track_url': top_track['external_urls']['spotify'],
        'top_track_mp3': top_track_mp3,
    }


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


def download_mp3(track_url):
    try:
        obj = Spotdl(client_id=SPOTIFY_CLIENT_ID, client_secret=SPOTIFY_CLIENT_SECRET, no_cache=True)
        song_objs = obj.search([track_url])
        return obj.download_songs(song_objs)

    except Exception as e:
        return f'Error downloading MP3 file: {str(e)}', 500
