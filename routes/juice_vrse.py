import requests
import os
import tempfile
import json
import logging
from google.cloud import storage
from flask import Blueprint, jsonify, request
from gradio_client import Client, file
from pydub import AudioSegment
from pydub.silence import split_on_silence
from bson.objectid import ObjectId
import librosa
import numpy as np
from scipy.signal import find_peaks

from routes.remix import detect_pitch

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

min_silence_len = 1000
silence_thresh = -22

key_json = os.environ.get('GOOGLE_CLOUD_KEY_JSON')
key_info = json.loads(key_json)
client = storage.Client.from_service_account_info(key_info)
bucket_name = 'huggingbass-bucket'
bucket = client.bucket(bucket_name)

juice_blueprint = Blueprint('juice', __name__)


@juice_blueprint.route('/juice-vrse', methods=['POST'])
def juice_vrse():
    logger.info("Received POST request to /juice-vrse")
    data = request.get_json()
    suno_persistant_link = data.get('sunoLink')
    logger.info(f"Processing Suno persistent link: {suno_persistant_link}")

    logger.info("Splitting audio into vocal and background")
    vocal_url, background_url = split_for_juice(suno_persistant_link)
    logger.info(f"Split complete. Vocal URL: {vocal_url}, Background URL: {background_url}")

    logger.info("Starting voice conversion process")
    inferred_audio_url = infer_audio_juice_vrse(
        'https://storage.googleapis.com/huggingbass-bucket/model.pth',
        'https://storage.googleapis.com/huggingbass-bucket/model.index',
        vocal_url,
    )
    logger.info(f"Voice conversion complete. Inferred audio URL: {inferred_audio_url}")

    logger.info("Combining converted voice with background")
    combined_song_url = combine_song_components(inferred_audio_url, background_url)
    logger.info(f"Combination complete. Final song URL: {combined_song_url}")

    logger.info("Returning response")
    return jsonify({'inferredAudioUrl': combined_song_url}), 200


def split_for_juice(track_url):
    logger.info(f"Starting audio splitting process for track: {track_url}")
    random_id = ObjectId()
    logger.info(f"Generated random ID for file storage: {random_id}")

    hb_client = Client("mealss/Audio_separator")
    logger.info("Initialized Audio separator client")

    logger.info("Isolating vocals")
    vocal_split_result = hb_client.predict(
        media_file=file(track_url),
        stem="vocal",
        main=False,
        dereverb=False,
        api_name="/sound_separate"
    )
    all_voice_stem_path = vocal_split_result[0]
    logger.info(f"Vocals isolated. Stem path: {all_voice_stem_path}")

    logger.info("Uploading isolated vocals to GCS")
    with open(all_voice_stem_path, 'rb') as source1_file:
        all_voice_file_blob = bucket.blob(f"juice-seperated-files/{random_id}/all-vocals.wav")
        all_voice_file_blob.upload_from_file(source1_file)
    logger.info(f"Vocals uploaded. Blob path: {all_voice_file_blob.name}")

    logger.info("Isolating background")
    background_split_result = hb_client.predict(
        media_file=file(track_url),
        stem="background",
        main=False,
        dereverb=False,
        api_name="/sound_separate"
    )
    background_stem_path = background_split_result[0]
    logger.info(f"Background isolated. Stem path: {background_stem_path}")

    logger.info("Uploading isolated background to GCS")
    with open(background_stem_path, 'rb') as source2_file:
        background_file_blob = bucket.blob(f"juice-seperated-files/{random_id}/background.wav")
        background_file_blob.upload_from_file(source2_file)
    logger.info(f"Background uploaded. Blob path: {background_file_blob.name}")

    logger.info("Applying deverb to vocals")
    deverb_vocal_split_result = hb_client.predict(
        media_file=file(all_voice_file_blob.public_url),
        stem="vocal",
        main=False,
        dereverb=True,
        api_name="/sound_separate"
    )
    deverb_voice_stem_path = deverb_vocal_split_result[0]
    logger.info(f"Deverb applied. Stem path: {deverb_voice_stem_path}")

    logger.info("Uploading deverbed vocals to GCS")
    with open(deverb_voice_stem_path, 'rb') as source1_file:
        deverb_voice_file_blob = bucket.blob(f"juice-seperated-files/{random_id}/deverbed-vocal.wav")
        deverb_voice_file_blob.upload_from_file(source1_file)
    logger.info(f"Deverbed vocals uploaded. Blob path: {deverb_voice_file_blob.name}")

    logger.info("Audio splitting process complete")
    return deverb_voice_file_blob.public_url, background_file_blob.public_url


def analyze_voice(audio_path):
    logger.info(f"Starting voice analysis for audio: {audio_path}")
    y, sr = librosa.load(audio_path)
    logger.info(f"Audio loaded. Sample rate: {sr}")

    logger.info("Performing pitch analysis")
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitches = pitches[pitches > 0]
    pitch_range = {
        'min': np.min(pitches),
        'max': np.max(pitches),
        'mean': np.mean(pitches)
    }
    logger.info(f"Pitch analysis complete. Range: {pitch_range}")

    logger.info("Performing formant analysis")
    S = librosa.stft(y)
    D = librosa.amplitude_to_db(np.abs(S), ref=np.max)
    frequencies = librosa.fft_frequencies(sr=sr)
    formants = []
    for col in D.T:
        peaks, _ = find_peaks(col, height=0, distance=100)
        formant_freqs = frequencies[peaks][:3]
        formants.append(formant_freqs)
    avg_formants = np.mean(formants, axis=0)
    logger.info(f"Formant analysis complete. Average formants: {avg_formants}")

    logger.info("Voice analysis complete")
    return {
        'pitch_range': pitch_range,
        'formants': avg_formants.tolist()
    }


def preprocess_audio(input_audio, reference_characteristics):
    logger.info(f"Starting audio preprocessing for input: {input_audio}")
    y, sr = librosa.load(input_audio)
    logger.info(f"Audio loaded. Sample rate: {sr}")

    logger.info("Analyzing input audio characteristics")
    input_characteristics = analyze_voice(input_audio)

    logger.info("Performing pitch shifting")
    pitch_shift = reference_characteristics['pitch_range']['mean'] - input_characteristics['pitch_range']['mean']
    y_shifted = librosa.effects.pitch_shift(y, sr=sr, n_steps=pitch_shift)
    logger.info(f"Pitch shifted by {pitch_shift} steps")

    logger.info("Performing formant shifting")
    n_fft = 2048
    f = librosa.stft(y_shifted, n_fft=n_fft)
    input_formants = input_characteristics['formants']
    ref_formants = reference_characteristics['formants']

    for i in range(min(len(input_formants), len(ref_formants))):
        shift_ratio = ref_formants[i] / input_formants[i]
        freq_bins = np.fft.fftfreq(n_fft, 1 / sr)
        shift_mask = np.exp(1j * 2 * np.pi * freq_bins * (shift_ratio - 1) / sr)
        f *= shift_mask[:, np.newaxis]
    logger.info("Formant shifting complete")

    y_processed = librosa.istft(f)
    logger.info("Audio preprocessing complete")

    return y_processed, sr


def adaptive_voice_conversion(input_audio, reference_characteristics, conversion_model):
    logger.info(f"Starting adaptive voice conversion for input: {input_audio}")
    y_preprocessed, sr = preprocess_audio(input_audio, reference_characteristics)
    logger.info("Audio preprocessing complete")

    logger.info("Saving preprocessed audio to temporary file")
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
        librosa.output.write_wav(temp_file.name, y_preprocessed, sr)
        preprocessed_path = temp_file.name
    logger.info(f"Preprocessed audio saved to: {preprocessed_path}")

    logger.info("Applying voice conversion")
    result = conversion_model.predict(
        audio_files=[file(preprocessed_path)],
        pitch_alg="rmvpe+",
        pitch_lvl=float(detect_pitch(preprocessed_path)),
        file_m=pth_file_url,
        file_index=index_file_url,
        index_inf=0.75,
        r_m_f=3,
        e_r=0.25,
        c_b_p=0.5,
        api_name="/run"
    )
    logger.info("Voice conversion complete")
    return result[0]


def infer_audio_juice_vrse(pth_file_url, index_file_url, reference_url):
    logger.info(f"Starting audio inference process. Reference URL: {reference_url}")
    hb_client = Client("mealss/rvc_zero")
    logger.info("Initialized RVC Zero client")

    logger.info("Analyzing reference voice")
    reference_characteristics = analyze_voice(reference_url)
    logger.info(f"Reference voice characteristics: {reference_characteristics}")

    logger.info("Downloading reference audio")
    response = requests.get(reference_url)
    response.raise_for_status()
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(response.content)
        temp_reference_file_path = temp_file.name
    logger.info(f"Reference audio downloaded to: {temp_reference_file_path}")

    logger.info("Loading audio file")
    audio = AudioSegment.from_wav(temp_reference_file_path)

    logger.info("Splitting audio based on silence")
    chunks = split_on_silence(
        audio,
        min_silence_len=min_silence_len,
        silence_thresh=silence_thresh,
        keep_silence=True
    )
    logger.info(f"Audio split into {len(chunks)} chunks")

    inferred_chunks = []
    for i, chunk in enumerate(chunks):
        logger.info(f"Processing chunk {i + 1}/{len(chunks)}")
        input_folder = f"/tmp"
        chunk_filename = f"chunk{i}.wav"
        chunk_path = os.path.join(input_folder, chunk_filename)
        os.makedirs(input_folder, exist_ok=True)
        chunk.export(chunk_path, format="wav")
        logger.info(f"Chunk saved to: {chunk_path}")

        logger.info("Applying adaptive voice conversion")
        converted_chunk = adaptive_voice_conversion(chunk_path, reference_characteristics, hb_client)
        inferred_chunks.append(converted_chunk)
        logger.info(f"Chunk {i + 1} conversion complete")

    logger.info("Combining inferred chunks")
    combined = AudioSegment.empty()
    for chunk in inferred_chunks:
        combined += AudioSegment.from_file(chunk, format="wav")
    logger.info("All chunks combined")

    logger.info("Exporting combined audio")
    combined_file_path = f"/tmp/combined.wav"
    combined.export(combined_file_path, format="wav")
    logger.info(f"Combined audio exported to: {combined_file_path}")

    logger.info("Uploading combined file to GCS")
    random_id = ObjectId()
    with open(combined_file_path, 'rb') as combined_file:
        audio_file_blob = bucket.blob(f"juice-inferred-audios/{random_id}/isolated-vocal.wav")
        audio_file_blob.upload_from_file(combined_file)
    public_url = audio_file_blob.public_url
    logger.info(f"Combined audio uploaded. Public URL: {public_url}")

    return public_url


def combine_song_components(vocal_url, background_url):
    logger.info(f"Starting song component combination. Vocal URL: {vocal_url}, Background URL: {background_url}")

    logger.info("Downloading vocal file")
    vocal_track_url_response = requests.get(vocal_url)
    vocal_track_url_response.raise_for_status()
    with tempfile.NamedTemporaryFile(delete=False) as temp_file_vocal:
        temp_file_vocal.write(vocal_track_url_response.content)
        temp_file_vocal_path = temp_file_vocal.name
    logger.info(f"Vocal file downloaded to: {temp_file_vocal_path}")

    logger.info("Downloading background file")
    background_track_url_response = requests.get(background_url)
    background_track_url_response.raise_for_status()
    with tempfile.NamedTemporaryFile(delete=False) as temp_file_background:
        temp_file_background.write(background_track_url_response.content)
        temp_file_background_path = temp_file_background.name
    logger.info(f"Background file downloaded to: {temp_file_background_path}")

    logger.info("Loading audio files")
    sound1 = AudioSegment.from_wav(temp_file_vocal_path)
    sound2 = AudioSegment.from_wav(temp_file_background_path)

    logger.info("Overlaying vocal and background")
    combined = sound1.overlay(sound2)

    logger.info("Exporting combined audio")
    input_folder = f"/tmp/combined/{vocal_url}/{background_url}"
    combined_filename = f"combined.wav"
    combined_path = os.path.join(input_folder, combined_filename)
    os.makedirs(input_folder, exist_ok=True)
    combined.export(combined_path, format='wav')
    logger.info(f"Combined audio exported to: {combined_path}")

    logger.info("Uploading combined audio to GCS")
    random_id = ObjectId()
    with open(combined_path, 'rb') as combined_file:
        combined_audio_blob = bucket.blob(f"juice-newly-combined/{random_id}/recombined.wav")
        combined_audio_blob.upload_from_file(combined_file)
    logger.info(f"Combined audio uploaded. Blob path: {combined_audio_blob.name}")

    return combined_audio_blob.public_url