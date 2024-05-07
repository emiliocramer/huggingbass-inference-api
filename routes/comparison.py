import os
import requests
import base64
import json
import threading
import numpy as np
import librosa
import jsonify


from flask import Blueprint, request
from google.cloud import storage
from scipy.spatial.distance import euclidean


comparison_blueprint = Blueprint('comparison', __name__)

key_json = os.environ.get('GOOGLE_CLOUD_KEY_JSON')
key_info = json.loads(key_json)
client = storage.Client.from_service_account_info(key_info)
bucket_name = 'opus-storage-bucket'
bucket = client.bucket(bucket_name)


@comparison_blueprint.route('/get-score', methods=['POST'])
def get_comparison_score():
    data = request.get_json()
    if 'inferredAudioUrls' not in data or 'referenceAudioUrl' not in data:
        return 'Missing inferredAudioUrls or referenceAudioUrl in request body', 400

    inferred_audio_urls = data['inferredAudioUrls']
    reference_audio_url = data['referenceAudioUrl']

    threading.Thread(target=process_comparison, args=(inferred_audio_urls, reference_audio_url)).start()
    return "Comparison successfully underway. Please wait for completion."


def process_comparison(inferred_audio_urls, reference_audio_url):
    # Load the reference audio
    reference_audio, _ = librosa.load(requests.get(reference_audio_url).content, sr=22050)

    # Compute the Mel spectrogram of the reference audio
    reference_mel_spec = librosa.feature.melspectrogram(y=reference_audio, sr=22050)

    similarity_scores = []

    for inferred_audio_url in inferred_audio_urls:
        # Load the inferred audio
        inferred_audio, _ = librosa.load(requests.get(inferred_audio_url).content, sr=22050)

        # Compute the Mel spectrogram of the inferred audio
        inferred_mel_spec = librosa.feature.melspectrogram(y=inferred_audio, sr=22050)

        # Compute the Euclidean distance between the Mel spectrograms
        distance = euclidean(reference_mel_spec.flatten(), inferred_mel_spec.flatten())

        # Normalize the distance to a similarity score between 0 and 100
        similarity_score = 100 * (1 / (1 + distance))

        similarity_scores.append(similarity_score)

    return jsonify(similarity_scores)
