import streamlit as st
import librosa
import soundfile as sf
import numpy as np
import io
import os
import tempfile
import scipy.io.wavfile

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
        
        # Convertir a WAV en memoria
        male_buffer = io.BytesIO()
        female_buffer = io.BytesIO()
        
        # Normalizar y convertir a int16
        male_voice_norm = np.int16(male_voice * 32767)
        female_voice_norm = np.int16(female_voice * 32767)
        
        # Guardar como WAV
        scipy.io.wavfile.write(male_buffer, sr, male_voice_norm)
        scipy.io.wavfile.write(female_buffer, sr, female_voice_norm)
        
        progress_bar.progress(100)
        status_text.text("¡Proceso completado!")
        
        return male_buffer.getvalue(), female_buffer.getvalue()
        
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
                    file_name="voz_masculina.wav",
                    mime="audio/wav"
                )
            
            with col2:
                st.download_button(
                    label="⬇️ Descargar Voz Femenina",
                    data=female_data,
                    file_name="voz_femenina.wav",
                    mime="audio/wav"
                )
                
            st.success("✅ Audio procesado correctamente! Los archivos están listos para descargar.")
                
        except Exception as e:
            st.error(f"Error durante el procesamiento: {str(e)}")

# Información adicional
st.markdown("---")
st.write("### Notas:")
st.write("""
- La separación se basa en el tono de voz (pitch)
- Los silencios se mantienen para que los archivos permanezcan sincronizados
- La calidad de la separación depende de la calidad del audio original
- Los archivos se guardan en formato WAV para mantener la calidad
- Si las voces se solapan, la separación puede no ser perfecta
""")
