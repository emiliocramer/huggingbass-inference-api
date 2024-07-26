import requests
import os
import tempfile
import json
from google.cloud import storage
from flask import Blueprint, jsonify, request
from gradio_client import Client, file
from pydub import AudioSegment
from pydub.silence import split_on_silence
from bson.objectid import ObjectId

from routes.remix import detect_pitch

min_silence_len = 500
silence_thresh = -22

key_json = os.environ.get('GOOGLE_CLOUD_KEY_JSON')
key_info = json.loads(key_json)
client = storage.Client.from_service_account_info(key_info)
bucket_name = 'huggingbass-bucket'
bucket = client.bucket(bucket_name)

juice_blueprint = Blueprint('juice', __name__)



@juice_blueprint.route('/juice-vrse', methods=['POST'])
def juice_vrse():
    data = request.get_json()
    suno_persistant_link = data.get('sunoLink')
    # get split_for_juice output
    vocal_url, background_url = split_for_juice(suno_persistant_link)

    inferred_audio_url = infer_audio_juice_vrse(
        'https://storage.googleapis.com/huggingbass-bucket/model.pth',
        'https://storage.googleapis.com/huggingbass-bucket/model.index',
        vocal_url,
    )

    # Clear variables
    pth_file_url = None
    index_file_url = None
    return jsonify({'inferredAudioUrl': inferred_audio_url}), 200


def infer_audio_juice_vrse(pth_file_url, index_file_url, reference_url):
    hb_client = Client("mealss/rvc_zero")

    # Download the reference audio
    response = requests.get(reference_url)
    response.raise_for_status()
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(response.content)
        temp_reference_file_path = temp_file.name

    # Load the audio file
    audio = AudioSegment.from_wav(temp_reference_file_path)
    # Split the audio based on silence
    chunks = split_on_silence(
        audio,
        min_silence_len=min_silence_len,
        silence_thresh=silence_thresh,
        keep_silence=True
    )
    print(f'found {len(chunks)} chunks')

    inferred_chunks = []
    for i, chunk in enumerate(chunks):
        input_folder = f"/tmp"
        chunk_filename = f"chunk{i}.wav"
        chunk_path = os.path.join(input_folder, chunk_filename)
        os.makedirs(input_folder, exist_ok=True)
        chunk.export(chunk_path, format="wav")
        chunkPitch = detect_pitch(chunk_path)
        chunkPitch = float(chunkPitch)
        print(f'chunk {i} pitch: {chunkPitch}')

        print(f'inferring chunk {i}')
        result = hb_client.predict(
            audio_files=[file(chunk_path)],
            file_m=pth_file_url,
            pitch_alg="rmvpe+",
            pitch_lvl=chunkPitch,
            file_index=index_file_url,
            index_inf=0.75,
            r_m_f=3,
            e_r=0.25,
            c_b_p=0.5,
            api_name="/run"
        )
        if "error" in result:
            raise ValueError(result["error"])
        print(f'finished inferring chunk {i}')
        inferred_chunks.append(result[0])

    # Combine the inferred chunks
    combined = AudioSegment.empty()
    for chunk in inferred_chunks:
        combined += AudioSegment.from_file(chunk, format="wav")

    # Define the path for the combined audio file
    combined_file_path = f"/tmp/combined.wav"
    combined.export(combined_file_path, format="wav")

    # create random ID
    random_id = ObjectId()

    # Upload the combined file
    with open(combined_file_path, 'rb') as combined_file:
        audio_file_blob = bucket.blob(f"juice-inferred-audios/{random_id}/isolated-vocal.wav")
        audio_file_blob.upload_from_file(combined_file)
    public_url = audio_file_blob.public_url
    return public_url


def split_for_juice(track_url):
    random_id = ObjectId()

    # isolate vocals
    hb_client = Client("mealss/Audio_separator")
    vocal_split_result = hb_client.predict(
        media_file=file(track_url),
        stem="vocal",
        main=False,
        dereverb=False,
        api_name="/sound_separate"
    )
    all_voice_stem_path = vocal_split_result[0]
    with open(all_voice_stem_path, 'rb') as source1_file:
        all_voice_file_blob = bucket.blob(f"juice-seperated-files/{random_id}/all-vocals.wav")
        all_voice_file_blob.upload_from_file(source1_file)

    # isolate background
    background_split_result = hb_client.predict(
        media_file=file(track_url),
        stem="background",
        main=False,
        dereverb=False,
        api_name="/sound_separate"
    )
    background_stem_path = background_split_result[0]
    with open(background_stem_path, 'rb') as source2_file:
        background_file_blob = bucket.blob(f"juice-seperated-files/{random_id}/background.wav")
        background_file_blob.upload_from_file(source2_file)

    # isolate primary
    primary_vocal_split_result = hb_client.predict(
        media_file=file(all_voice_file_blob.public_url),
        stem="vocal",
        main=False,
        dereverb=False,
        api_name="/sound_separate"
    )
    primary_voice_stem_path = primary_vocal_split_result[0]
    with open(primary_voice_stem_path, 'rb') as source1_file:
        primary_voice_file_blob = bucket.blob(f"juice-seperated-files/{random_id}/primary-vocal.wav")
        primary_voice_file_blob.upload_from_file(source1_file)

    # deverb primary
    deverb_vocal_split_result = hb_client.predict(
        media_file=file(primary_voice_file_blob.public_url),
        stem="vocal",
        main=False,
        dereverb=True,
        api_name="/sound_separate"
    )
    deverb_voice_stem_path = deverb_vocal_split_result[0]
    with open(deverb_voice_stem_path, 'rb') as source1_file:
        deverb_voice_file_blob = bucket.blob(f"juice-seperated-files/{random_id}/deverbed-vocal.wav")
        deverb_voice_file_blob.upload_from_file(source1_file)

    return deverb_voice_file_blob.public_url, background_file_blob.public_url

