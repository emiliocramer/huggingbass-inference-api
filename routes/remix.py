import zipfile
import requests
import librosa
import os
import json
import queue
import numpy as np
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor
from pydub import AudioSegment

from flask import Blueprint, jsonify, request
from db import models_collection
from .inference import unzip_model_files
from google.cloud import storage
from bson.objectid import ObjectId
from gradio_client import Client, file


remix_blueprint = Blueprint('remix', __name__)

key_json = os.environ.get('GOOGLE_CLOUD_KEY_JSON')
key_info = json.loads(key_json)
client = storage.Client.from_service_account_info(key_info)
bucket_name = 'huggingbass-bucket'
bucket = client.bucket(bucket_name)

task_queue = queue.Queue()
MAX_WORKER_THREADS = 50
executor = ThreadPoolExecutor(max_workers=MAX_WORKER_THREADS)


# @remix_blueprint.route('/combine-for-output', methods=['POST'])
# def process_combine_song_components(vocal_track_url, background_track_url):
#     data = request.get_json()
#     track_url = data.get('trackUrl')
#     track_id = data.get('trackId')
#
#     if not track_url or not track_id:
#         return jsonify({'error': 'Missing trackUrl or trackId'}), 400
#
#     sound1 = AudioSegment.from_file("/path/to/my_sound.wav")
#     sound2 = AudioSegment.from_file("/path/to/another_sound.wav")
#
#     combined = sound1.overlay(sound2)
#
#     combined.export("/path/to/combined.wav", format='wav')

@remix_blueprint.route('/split-for-remix', methods=['POST'])
def process_split_and_upload_from_mp3():
    data = request.get_json()
    track_url = data.get('trackUrl')
    track_id = data.get('trackId')

    if not track_url or not track_id:
        return jsonify({'error': 'Missing trackUrl or trackId'}), 400

    hb_client = Client("r3gm/Audio_separator")
    vocal_split_result = hb_client.predict(
        media_file=file(track_url),
        stem="vocal",
        main=False,
        dereverb=False,
        api_name="/sound_separate"
    )
    voice_stem_path = vocal_split_result[0]

    background_split_result = hb_client.predict(
        media_file=file(track_url),
        stem="vocal",
        main=False,
        dereverb=False,
        api_name="/sound_separate"
    )
    background_stem_path = background_split_result[0]

    with open(voice_stem_path, 'rb') as source1_file:
        voice_file_blob = bucket.blob(f"remix-seperated-files/{track_id}/vocal.wav")
        voice_file_blob.upload_from_file(source1_file)

    with open(background_stem_path, 'rb') as source2_file:
        background_file_blob = bucket.blob(f"remix-seperated-files/{track_id}/background.wav")
        background_file_blob.upload_from_file(source2_file)

    return jsonify({'vocal': voice_file_blob.public_url, 'background': background_file_blob.public_url}), 200


@remix_blueprint.route('/remix', methods=['POST'])
def remix_audio():
    data = request.get_json()
    model_id = data.get('modelId')
    reference_url = data.get('referenceUrl')

    if not model_id or not reference_url:
        return jsonify({'error': 'Missing modelId or referenceUrl'}), 400

    model = models_collection.find_one({'_id': ObjectId(model_id)})
    if not model:
        return 'Model not found', 404

    print("found model: ", model['name'])

    zipped_file_url = model['fileUrls'][0]
    zipped_file_response = requests.get(zipped_file_url)
    with zipped_file_response:
        zipped_file_bytes = zipped_file_response.content
        zipped_file = BytesIO(zipped_file_bytes)
        with zipfile.ZipFile(zipped_file, 'r') as zip_ref:
            pth_file_url, index_file_url = unzip_model_files(zip_ref, model_id)

    print("unzipped model files")
    print(f'pth_file_url: {pth_file_url}')
    print(f'index_file_url: {index_file_url}')

    inferred_audio_url = infer_audio(pth_file_url, index_file_url, reference_url, model['name'])

    # Clear variables
    pth_file_url = None
    index_file_url = None
    return jsonify({'inferredAudioUrl': inferred_audio_url}), 200


def infer_audio(pth_file_url, index_file_url, reference_url, model_name):
    hb_client = Client("mealss/rvc_zero")

    pitch = detect_pitch(reference_url)
    print(f'inferring for remix: for {model_name}')
    result = hb_client.predict(
        audio_files=[file(reference_url)],
        file_m=pth_file_url,
        pitch_alg="rmvpe+",
        pitch_lvl=pitch,
        file_index=index_file_url,
        index_inf=0.75,
        r_m_f=3,
        e_r=0.25,
        c_b_p=0.5,
        api_name="/run"
    )
    if "error" in result:
        raise ValueError(result["error"])
    print(f'finished inferring for {model_name}')
    audio_url = result[0]

    with open(audio_url, 'rb') as audio_file:
        audio_file_blob = bucket.blob(f"remix-inferred-audios/{model_name}/isolated-vocal.wav")
        audio_file_blob.upload_from_file(audio_file)

    public_url = audio_file_blob.public_url
    return public_url


def detect_pitch(audio_path):
    y, sr = librosa.load(audio_path)
    pitches, magnitudes = librosa.core.piptrack(y=y, sr=sr)
    pitches = pitches[magnitudes > np.median(magnitudes)]
    pitch = np.median(pitches)
    return pitch
