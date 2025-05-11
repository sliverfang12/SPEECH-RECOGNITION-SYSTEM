from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch
import librosa

def mp3_to_wav2vec(mp3_file):
    # Load MP3 directly with librosa (it supports MP3 format)
    audio_input, _ = librosa.load(mp3_file, sr=16000)
    return audio_input

def transcribe_audio_wav2vec(mp3_file):
    # Load pre-trained Wav2Vec2 model and processor
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-xlsr-53")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-xlsr-53")

    # Process and predict
    audio_input = mp3_to_wav2vec(mp3_file)
    input_values = processor(audio_input, return_tensors="pt").input_values
    logits = model(input_values).logits

    # Decode the predicted ids
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.decode(predicted_ids[0])

    print(f"Transcription: {transcription}")

# Test the function with an MP3 file
transcribe_audio_wav2vec("your_audio_file.mp3")
