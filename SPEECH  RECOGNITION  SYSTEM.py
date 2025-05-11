import speech_recognition as sr
from pydub import AudioSegment

def mp3_to_wav(mp3_file):
    # Convert MP3 to WAV using pydub
    audio = AudioSegment.from_mp3(mp3_file)
    wav_file = "converted_audio.wav"
    audio.export(wav_file, format="wav")
    return wav_file

def transcribe_audio(audio_file):
    recognizer = sr.Recognizer()

    # Open the audio file
    with sr.AudioFile(audio_file) as source:
        audio_data = recognizer.record(source)

    # Use Google Web Speech API for transcription
    try:
        print("Transcribing...")
        text = recognizer.recognize_google(audio_data)
        print("Transcription: " + text)
    except sr.UnknownValueError:
        print("Sorry, I could not understand the audio.")
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")

# Convert MP3 to WAV and transcribe
wav_file = mp3_to_wav("path_to_audio.mp3")
transcribe_audio(wav_file)
