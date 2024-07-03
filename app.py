import streamlit as st
import os
import shutil
import numpy as np
from loading_model import load_whisper_model



def save_uploaded_file(uploaded_file, save_path):
    os.makedirs(save_path, exist_ok=True)  # Ensure the directory exists
    file_path = os.path.join(save_path, uploaded_file.name)
    
    with open(file_path, "wb") as out_file:
        shutil.copyfileobj(uploaded_file, out_file)

def transcribe_audio(filepath, model):
    # Transcribe using whisper
    result = model.transcribe(filepath)
    transcription = result['text']
    return transcription

def main():
    st.title("Audio Transcriber")

    # Sidebar for model selection
    model_name = st.sidebar.selectbox(
        "Select Whisper Model",
        ("tiny", "base", "small", "medium", "large")
    )
    model = load_whisper_model(model_name)

    # File uploader widget, allowing multiple audio file types
    uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "m4a", "flac", "ogg", "aac"])

    if uploaded_file is not None:
        save_path = "audio_files"  # Folder to save the uploaded file
        os.makedirs(save_path, exist_ok=True)

        # Display the uploaded file
        st.audio(uploaded_file, format=uploaded_file.type)

        # Save the uploaded file
        save_uploaded_file(uploaded_file, save_path)
        st.success("Audio file uploaded successfully.")
        
        # Transcribe the audio
        transcribed_text = transcribe_audio(
            f"audio_files/{uploaded_file.name}", model)
        st.subheader("Transcribed Text:")
        st.text_area("Transcribed Text", value=transcribed_text, height=200)

if __name__ == "__main__":
    main()
