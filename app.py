import streamlit as st
from pydub import AudioSegment
import whisper
from deep_translator import GoogleTranslator
from gtts import gTTS
import os

def trim_audio(audio_path, duration=30000):  # Duration in milliseconds
    audio = AudioSegment.from_file(audio_path)
    trimmed_audio = audio[:duration]  # Trim to the first 30 seconds
    trimmed_path = "trimmed_audio.wav"
    trimmed_audio.export(trimmed_path, format="wav")  # Save the trimmed audio
    return trimmed_path

def language_detection(audio_path, model_size="medium"):
    model = whisper.load_model(model_size)
    result = model.transcribe(audio_path, verbose=False)
    return result['language']

def transcribe_audio(audio_path, model_size="medium"):
    model = whisper.load_model(model_size)
    result = model.transcribe(audio_path, verbose=False,language='english')
    return result['text']

def translate_text(text, src_lang, target_lang):
    translated = GoogleTranslator(source=src_lang, target=target_lang).translate(text)
    return translated

def text_to_speech(text, lang):
    tts = gTTS(text=text, lang=lang, slow=False)
    audio_path = f"audio_{lang}.mp3"
    tts.save(audio_path)
    return audio_path

# Streamlit interface
st.title('Audio Transcription and Translation App')

uploaded_file = st.file_uploader("Choose an audio file", type=['mp3', 'wav'])
if uploaded_file is not None:
    duration =  30 * 1000
    trimmed_path = trim_audio(uploaded_file, duration)


    language, transcription = language_detection(trimmed_path)
    st.write("Detected Language:", language)
    st.write("Transcription:", transcription)
    
    transcription = transcribe_audio(uploaded_file)
    st.write("Transcription:", transcription)

    target_language = st.text_input("Translate to language (ISO code)", 'es')
    if st.button('Translate'):
        translated_text = translate_text(transcription, language, target_language)
        st.write("Translated Text:", translated_text)

        audio_path = text_to_speech(translated_text, target_language)
        st.audio(audio_path)
        with open(audio_path, 'rb') as f:
            st.download_button('Download Translated Audio', f, file_name=audio_path)
