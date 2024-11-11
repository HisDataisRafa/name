import streamlit as st
import librosa
import soundfile as sf
import numpy as np
from pydub import AudioSegment
import os
import tempfile

st.set_page_config(page_title="Separador de Voces", layout="wide")

st.title("🎤 Separador de Voces")
st.write("Esta aplicación separa voces masculinas y femeninas de un archivo de audio.")

def process_audio(audio_file):
    # Crear directorio temporal
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Guardar el archivo subido temporalmente
        temp_input = os.path.join(temp_dir, "input.mp3")
        with open(temp_input, "wb") as f:
            f.write(audio_file.getvalue())
        
        # Barra de progreso
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Cargar audio
        status_text.text("Cargando audio...")
        progress_bar.progress(10)
        y, sr = librosa.load(temp_input, sr=None)
        
        # Detectar segmentos
        status_text.text("Detectando segmentos de voz...")
        progress_bar.progress(30)
        intervals = librosa.effects.split(y, top_db=20)
        
        # Preparar arrays
        male_voice = np.zeros_like(y)
        female_voice = np.zeros_like(y)
        
        # Procesar segmentos
        status_text.text("Separando voces...")
        total_intervals = len(intervals)
        
        for i, (start, end) in enumerate(intervals):
            segment = y[start:end]
            pitches, magnitudes = librosa.piptrack(y=segment, sr=sr)
            pit = pitches[magnitudes > 0.1]
            if len(pit) > 0:
                avg_pitch = np.mean(pit)
                if avg_pitch < 185:
                    male_voice[start:end] = segment
                else:
                    female_voice[start:end] = segment
            
            # Actualizar progreso
            progress = 30 + (i / total_intervals * 40)
            progress_bar.progress(int(progress))
            status_text.text(f"Procesando segmento {i+1} de {total_intervals}")
        
        # Guardar archivos
        status_text.text("Guardando archivos...")
        progress_bar.progress(80)
        
        # Guardar WAVs temporales
        male_wav = os.path.join(temp_dir, 'male.wav')
        female_wav = os.path.join(temp_dir, 'female.wav')
        
        sf.write(male_wav, male_voice, sr)
        sf.write(female_wav, female_voice, sr)
        
        # Convertir a MP3
        male_mp3 = os.path.join(temp_dir, 'male.mp3')
        female_mp3 = os.path.join(temp_dir, 'female.mp3')
        
        AudioSegment.from_wav(male_wav).export(male_mp3, format='mp3')
        AudioSegment.from_wav(female_wav).export(female_mp3, format='mp3')
        
        progress_bar.progress(100)
        status_text.text("¡Proceso completado!")
        
        # Leer archivos MP3 para descargar
        with open(male_mp3, 'rb') as f:
            male_data = f.read()
        with open(female_mp3, 'rb') as f:
            female_data = f.read()
        
        return male_data, female_data
        
    finally:
        # Limpiar archivos temporales
        for file in os.listdir(temp_dir):
            try:
                os.remove(os.path.join(temp_dir, file))
            except:
                pass
        try:
            os.rmdir(temp_dir)
        except:
            pass

# Interfaz de usuario
st.write("### Instrucciones:")
st.write("1. Sube tu archivo MP3")
st.write("2. Espera a que se procese")
st.write("3. Descarga los archivos separados")

uploaded_file = st.file_uploader("Escoge un archivo MP3", type=['mp3'])

if uploaded_file is not None:
    if st.button("Separar Voces"):
        try:
            with st.spinner('Procesando...'):
                male_data, female_data = process_audio(uploaded_file)
            
            # Botones de descarga
            col1, col2 = st.columns(2)
            
            with col1:
                st.download_button(
                    label="⬇️ Descargar Voz Masculina",
                    data=male_data,
                    file_name="voz_masculina.mp3",
                    mime="audio/mp3"
                )
            
            with col2:
                st.download_button(
                    label="⬇️ Descargar Voz Femenina",
                    data=female_data,
                    file_name="voz_femenina.mp3",
                    mime="audio/mp3"
                )
                
        except Exception as e:
            st.error(f"Error durante el procesamiento: {str(e)}")
