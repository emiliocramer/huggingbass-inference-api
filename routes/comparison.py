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
bucket_name = 'huggingbass-bucket'
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


def calculate_snr(reference_audio, inferred_audio):
    signal_power = np.mean(reference_audio ** 2)
    noise_power = np.mean((reference_audio - inferred_audio) ** 2)
    snr = 10 * np.log10(signal_power / noise_power)
    return snr


def process_comparison(inferred_audio_urls, reference_audio_url, model_id):
    # Load the reference audio
    print("loading reference audio")
    response = requests.get(reference_audio_url)
    reference_audio, _ = librosa.load(io.BytesIO(response.content), sr=22050)

    similarity_scores = []
    snr_scores = []
    for inferred_audio_url in inferred_audio_urls:
        # Load the inferred audio
        print("loading inferred audio")
        response = requests.get(inferred_audio_url)
        inferred_audio, _ = librosa.load(io.BytesIO(response.content), sr=22050)

        # Ensure the audio signals have the same length
        if len(reference_audio) < len(inferred_audio):
            # Pad the reference audio with zeros
            reference_audio = np.pad(reference_audio, (0, len(inferred_audio) - len(reference_audio)), 'constant')
        elif len(reference_audio) > len(inferred_audio):
            # Truncate the reference audio
            reference_audio = reference_audio[:len(inferred_audio)]

        # Compute the Mel spectrogram of the reference audio
        print("getting reference mel spectrogram")
        reference_mel_spec = librosa.feature.melspectrogram(y=reference_audio, sr=22050)

        # Compute the Mel spectrogram of the inferred audio
        print("getting inferred mel spectrogram")
        inferred_mel_spec = librosa.feature.melspectrogram(y=inferred_audio, sr=22050)

        print("computing similarity score")
        # Compute the cosine similarity between the Mel spectrograms
        similarity_score = np.dot(reference_mel_spec.flatten(), inferred_mel_spec.flatten()) / (
            np.linalg.norm(reference_mel_spec.flatten()) * np.linalg.norm(inferred_mel_spec.flatten())
        )
        similarity_scores.append(similarity_score)

        # Compute SNR
        snr = calculate_snr(reference_audio, inferred_audio)
        snr_scores.append(snr)

    print("Similarity Scores:", similarity_scores)
    print("SNR Scores:", snr_scores)
    max_similarity_score = max(similarity_scores)
    max_snr_score = max(snr_scores)
    
    print("Max Similarity Score:", max_similarity_score)
    print("Max SNR Score:", max_snr_score)

    weighted_average_score = 0.5 * max_similarity_score * 100 + 0.5 * max_snr_score

    model = models_collection.find_one({'_id': ObjectId(model_id)})
    model['qualityScore'] = weighted_average_score
    model['similarityScore'] = max_similarity_score * 100
    model['snrScore'] = max_snr_score
    models_collection.update_one({'_id': ObjectId(model_id)}, {'$set': model})
    
    return weighted_average_score, max_similarity_score * 100, max_snr_score
