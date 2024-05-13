import threading
import zipfile
import requests
import tempfile
import os
import json
import queue
import librosa
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor

from flask import Blueprint, request, jsonify
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

task_queue = queue.Queue()
MAX_WORKER_THREADS = 50
executor = ThreadPoolExecutor(max_workers=MAX_WORKER_THREADS)


def worker():
    while True:
        model_id, reference_url = task_queue.get()
        try:
            executor.submit(process_inferred_audio, model_id, reference_url)
        except Exception as e:
            print(f"Error submitting task: {e}")
        finally:
            task_queue.task_done()


@inference_blueprint.route('/get-inferred-audio-from-wav', methods=['POST'])
def get_inferred_audio_wav():
    data = request.get_json()
    if 'modelId' not in data or 'isolatedVocalUrl' not in data:
        return 'Missing modelId or isolatedVocal in request body', 400

    model_id = data['modelId']
    isolated_vocal_url = data['isolatedVocalUrl']

    task_queue.put((model_id, isolated_vocal_url))
    return jsonify({'message': "Task added to the queue. It will be processed soon."}), 200


@inference_blueprint.route('/get-inferred-audio-from-artist', methods=['POST'])
def get_inferred_audio_artist():
    data = request.get_json()
    if 'modelId' not in data or 'spotifyArtistId' not in data:
        return 'Missing modelId or spotifyArtistId in request body', 400

    model_id = data['modelId']
    artist_id = data['spotifyArtistId']

    reference_artist = None

    # try to get reference artist from collection until its found
    while not reference_artist:
        reference_artist = reference_artists_collection.find_one({'spotifyArtistId': artist_id})

    reference_url = reference_artist['audioStemUrl']

    task_queue.put((model_id, reference_url))
    return jsonify({'message': "Task added to the queue. It will be processed soon.", 'referenceAudio': reference_url}), 200


def process_inferred_audio(model_id, reference_url):
    model = models_collection.find_one({'_id': ObjectId(model_id)})
    if not model:
        return 'Model not found', 404

    print("found model: ", model['name'])

    pth_file_url = None
    index_file_url = None

    # Unzip the model file at index 1
    if len(model['fileUrls']) == 1:
        zipped_file_url = model['fileUrls'][0]

        zipped_file_response = requests.get(zipped_file_url)
        with zipped_file_response:  # Use context manager for the response
            zipped_file_bytes = zipped_file_response.content
            zipped_file = BytesIO(zipped_file_bytes)
            with zipfile.ZipFile(zipped_file, 'r') as zip_ref:  # Use context manager for ZipFile
                pth_file_url, index_file_url = unzip_model_files(zip_ref)

    else:
        for file_name in model['fileUrls']:
            if file_name.endswith(".pth"):
                pth_file_url = file_name
            elif file_name.endswith(".index"):
                index_file_url = file_name

    inferred_audios = []
    for i in range(-12, 13):
        inferred_audio = infer_audio(pth_file_url, index_file_url, reference_url, i, model['name'])
        inferred_audios.append(inferred_audio)

    model['inferredAudioUrls'] = inferred_audios
    models_collection.update_one({'_id': ObjectId(model_id)}, {'$set': model})

    # Clear variables
    pth_file_url = None
    index_file_url = None
    inferred_audios = None


def infer_audio(pth_file_url, index_file_url, reference_url, pitch, model_name):
    print(f'inferring pitch: {pitch} for {model_name}')
    hb_client = Client("mealss/rvc_zero")

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
    print(f'finished inferring pitch: {pitch} for {model_name}')
    audio_url = result[0]

    with open(audio_url, 'rb') as audio_file:
        audio_file_blob = bucket.blob(f"model-inferred-audios/{model_name}/pitch{pitch}-vocal.wav")
        audio_file_blob.upload_from_file(audio_file)

    return audio_file_blob.public_url


def unzip_model_files(zip_ref):
    pth_file_url = None
    index_file_url = None
    tmp_dir = tempfile.mkdtemp()
    try:
        zip_ref.extractall(tmp_dir)
        extracted_files = os.listdir(tmp_dir)

        if len(extracted_files) == 1:
            extracted_folder = os.path.join(tmp_dir, extracted_files[0])

            for file_name in os.listdir(extracted_folder):
                file_path = os.path.join(extracted_folder, file_name)
                if file_name.endswith(".pth"):
                    pth_file_url = file_path
                elif file_name.endswith(".index"):
                    index_file_url = file_path
        else:
            for file_name in extracted_files:
                file_path = os.path.join(tmp_dir, file_name)
                if file_name.endswith(".pth"):
                    pth_file_url = file_path
                elif file_name.endswith(".index"):
                    index_file_url = file_path
    finally:
        # Clear the temporary directory
        for root, dirs, files in os.walk(tmp_dir, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
        os.rmdir(tmp_dir)

    if not pth_file_url or not index_file_url:
        return 'Model file not found', 404

    return pth_file_url, index_file_url


worker_thread = threading.Thread(target=worker, daemon=True)
worker_thread.start()