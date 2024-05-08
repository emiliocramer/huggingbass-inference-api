import os
import requests
import io
import json
import threading
import librosa
from bson.objectid import ObjectId
from flask import Blueprint, request, jsonify
from google.cloud import storage
import numpy as np
from db import models_collection


comparison_blueprint = Blueprint('comparison', __name__)

key_json = os.environ.get('GOOGLE_CLOUD_KEY_JSON')
key_info = json.loads(key_json)
client = storage.Client.from_service_account_info(key_info)
bucket_name = 'opus-storage-bucket'
bucket = client.bucket(bucket_name)


@comparison_blueprint.route('/get-score', methods=['POST'])
def get_comparison_score():
    data = request.get_json()
    if 'inferredAudioUrls' not in data or 'referenceAudioUrl' not in data or 'modelId' not in data:
        return 'Missing inferredAudioUrls or referenceAudioUrl or modelId in request body', 400

    inferred_audio_urls = data['inferredAudioUrls']
    reference_audio_url = data['referenceAudioUrl']
    model_id = data['modelId']

    threading.Thread(target=process_comparison, args=(inferred_audio_urls, reference_audio_url, model_id)).start()
    return jsonify({'message': "Comparison successfully underway. Please wait for completion."}), 200


def process_comparison(inferred_audio_urls, reference_audio_url, model_id):
    # Load the reference audio
    print("loading reference audio")
    response = requests.get(reference_audio_url)
    reference_audio, _ = librosa.load(io.BytesIO(response.content), sr=22050)
    # Compute the Mel spectrogram of the reference audio
    print("getting it's mel spectrogram")
    reference_mel_spec = librosa.feature.melspectrogram(y=reference_audio, sr=22050)

    similarity_scores = []
    for inferred_audio_url in inferred_audio_urls:
        # Load the inferred audio
        print("loading inferred audio")
        response = requests.get(inferred_audio_url)
        inferred_audio, _ = librosa.load(io.BytesIO(response.content), sr=22050)
        print("getting it's mel spectrogram")
        # Compute the Mel spectrogram of the inferred audio
        inferred_mel_spec = librosa.feature.melspectrogram(y=inferred_audio, sr=22050)
        print("computing similarity score")
        # Compute the cosine similarity between the Mel spectrograms
        similarity_score = np.dot(reference_mel_spec.flatten(), inferred_mel_spec.flatten()) / (
            np.linalg.norm(reference_mel_spec.flatten()) * np.linalg.norm(inferred_mel_spec.flatten())
        )
        similarity_scores.append(similarity_score)

    print(similarity_scores)
    print(max(similarity_scores))
    model = models_collection.find_one({'_id': ObjectId(model_id)})
    model['qualityScore'] = max(similarity_scores) * 100  # Multiply by 100 to get a percentage
    models_collection.update_one({'_id': ObjectId(model_id)}, {'$set': model})
    return max(similarity_scores) * 100  # Return the maximum similarity score as a percentage
