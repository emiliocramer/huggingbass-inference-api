import threading
import zipfile
import requests
import tempfile
import os
import json
from io import BytesIO

from flask import Blueprint, request
from db import models_collection, reference_artists_collection
from google.cloud import storage
from bson.objectid import ObjectId
from gradio_client import Client, file


inference_blueprint = Blueprint('inference', __name__)

key_json = os.environ.get('GOOGLE_CLOUD_KEY_JSON')
key_info = json.loads(key_json)
client = storage.Client.from_service_account_info(key_info)
bucket_name = 'opus-storage-bucket'
bucket = client.bucket(bucket_name)


@inference_blueprint.route('/get-inferred-audio', methods=['POST'])
def get_inferred_audio():
    data = request.get_json()
    if 'modelId' not in data or 'spotifyArtistId' not in data:
        return 'Missing modelId or spotifyArtistId in request body', 400

    model_id = data['modelId']
    artist_id = data['spotifyArtistId']
    threading.Thread(target=process_inferred_audio, args=(model_id, artist_id)).start()
    return "Inferring successfully underway. Please wait for completion."


def process_inferred_audio(model_id, artist_id):
    model = models_collection.find_one({'_id': ObjectId(model_id)})
    if not model:
        return 'Model not found', 404

    # Unzip the model file at index 1
    zipped_file_url = model['fileUrls'][1]
    zipped_file_bytes = requests.get(zipped_file_url).content
    zipped_file = BytesIO(zipped_file_bytes)

    pth_file_url, index_file_url = unzip_model_files(zipped_file)

    reference_artist = reference_artists_collection.find_one({'spotifyArtistId': artist_id})
    if not reference_artist:
        return 'Reference artist not found', 404

    reference_url = reference_artist['audioStemUrl']
    if not reference_url:
        return 'Reference artist audio not found', 404

    inferred_audios = []
    for i in range(-15, 16):
        inferred_audio = infer_audio(pth_file_url, index_file_url, reference_url, i, model['name'])
        inferred_audios.append(inferred_audio)

    model['inferredAudioUrls'] = inferred_audios
    models_collection.update_one({'_id': ObjectId(model_id)}, {'$set': model})


def unzip_model_files(zipped_file):
    with zipfile.ZipFile(zipped_file, 'r') as zip_ref:
        tmp_dir = tempfile.mkdtemp()
        zip_ref.extractall(tmp_dir)

        for file_name in os.listdir(tmp_dir):
            if file_name.endswith(".pth"):
                pth_file_url = os.path.join(tmp_dir, file_name)
            elif file_name.endswith(".index"):
                index_file_url = os.path.join(tmp_dir, file_name)

    if not pth_file_url or not index_file_url:
        return 'Model file not found', 404

    return pth_file_url, index_file_url


def infer_audio(pth_file_url, index_file_url, reference_url, pitch, model_name):

    print("inferring pitch: ", pitch)

    hb_client = Client("r3gm/rvc_zero")
    result = hb_client.predict(
        audio_files=[file(reference_url)],
        file_m=file(pth_file_url),
        pitch_alg="rmvpe",
        pitch_lvl=pitch,
        file_index=file(index_file_url),
        index_inf=1,
        r_m_f=4,
        e_r=1,
        c_b_p=0.35,
        api_name="/run"
    )
    print("finished inferring pitch: ", pitch)

    audio_url = result[0]
    with open(audio_url, 'rb') as audio_file:
        audio_file_blob = bucket.blob(f"model-inferred-audios/{model_name}/pitch{pitch}-vocal.wav")
        audio_file_blob.upload_from_file(audio_file)

    return audio_file_blob.public_url
