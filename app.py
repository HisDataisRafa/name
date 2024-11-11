import streamlit as st
import librosa
import soundfile as sf
import numpy as np
import io
import os
import tempfile
import scipy.io.wavfile

st.set_page_config(page_title="Separador de Voces", layout="wide")

st.title("🎤 Separador de Voces Mejorado")
st.write("Esta aplicación separa voces masculinas y femeninas de un archivo de audio.")

def get_pitch_features(y, sr, frame_length=2048, hop_length=512):
    # Obtener características más detalladas del pitch
    pitches, magnitudes = librosa.piptrack(
        y=y, 
        sr=sr,
        n_fft=frame_length,
        hop_length=hop_length,
        fmin=librosa.note_to_hz('C2'),
        fmax=librosa.note_to_hz('C7')
    )
    
    # Obtener el pitch predominante en cada frame
    pitch_mean = []
    for i in range(magnitudes.shape[1]):
        index = magnitudes[:, i].argmax()
        pitch_mean.append(pitches[index, i])
    
    return np.array(pitch_mean)

def smooth_array(arr, window_size=5):
    # Suavizar las transiciones para evitar cortes bruscos
    kernel = np.ones(window_size) / window_size
    return np.convolve(arr, kernel, mode='same')

def process_audio(audio_file, pitch_threshold, smoothing_window):
    try:
        # Crear directorio temporal
        temp_dir = tempfile.mkdtemp()
        temp_input = os.path.join(temp_dir, "input.mp3")
        
        with open(temp_input, "wb") as f:
            f.write(audio_file.getvalue())
        
        # Mostrar progreso
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Cargar audio
        status_text.text("Cargando audio...")
        progress_bar.progress(10)
        
        # Cargar con una duración máxima más larga
        y, sr = librosa.load(temp_input, sr=None)
        
        # Normalizar audio
        y = librosa.util.normalize(y)
        
        # Detectar segmentos de voz
        status_text.text("Detectando segmentos de voz...")
        progress_bar.progress(30)
        
        # Parámetros para el análisis
        frame_length = 2048
        hop_length = 512
        
        # Obtener características del pitch
        status_text.text("Analizando características de voz...")
        pitch_mean = get_pitch_features(y, sr, frame_length, hop_length)
        
        # Crear máscaras para separación
        time_steps = np.arange(len(pitch_mean)) * hop_length
        mask_male = pitch_mean < pitch_threshold
        mask_female = ~mask_male
        
        # Suavizar máscaras
        mask_male = smooth_array(mask_male.astype(float), smoothing_window)
        mask_female = smooth_array(mask_female.astype(float), smoothing_window)
        
        # Crear señales separadas
        status_text.text("Separando voces...")
        progress_bar.progress(60)
        
        # Convertir máscaras a longitud de señal
        mask_male_full = np.repeat(mask_male, hop_length)[:len(y)]
        mask_female_full = np.repeat(mask_female, hop_length)[:len(y)]
        
        # Aplicar máscaras con superposición suave
        male_voice = y * mask_male_full
        female_voice = y * mask_female_full
        
        # Normalizar resultados
        male_voice = librosa.util.normalize(male_voice)
        female_voice = librosa.util.normalize(female_voice)
        
        status_text.text("Guardando archivos...")
        progress_bar.progress(90)
        
        # Convertir a int16 con normalización mejorada
        male_voice_int = np.int16(male_voice * 32767)
        female_voice_int = np.int16(female_voice * 32767)
        
        # Guardar en buffer
        male_buffer = io.BytesIO()
        female_buffer = io.BytesIO()
        
        scipy.io.wavfile.write(male_buffer, sr, male_voice_int)
        scipy.io.wavfile.write(female_buffer, sr, female_voice_int)
        
        progress_bar.progress(100)
        status_text.text("¡Proceso completado!")
        
        return male_buffer.getvalue(), female_buffer.getvalue()
        
    except Exception as e:
        st.error(f"Error en el procesamiento: {str(e)}")
        return None, None
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

# Interfaz de usuario con parámetros ajustables
st.write("### Configuración:")
col1, col2 = st.columns(2)

with col1:
    pitch_threshold = st.slider(
        "Umbral de pitch (Hz)",
        min_value=100,
        max_value=300,
        value=165,
        help="Ajusta este valor para determinar el límite entre voz masculina y femenina"
    )

with col2:
    smoothing_window = st.slider(
        "Ventana de suavizado",
        min_value=1,
        max_value=21,
        value=11,
        step=2,
        help="Un valor más alto hace las transiciones más suaves pero puede mezclar más las voces"
    )

uploaded_file = st.file_uploader("Escoge un archivo MP3", type=['mp3'])

if uploaded_file is not None:
    if st.button("Separar Voces"):
        male_data, female_data = process_audio(uploaded_file, pitch_threshold, smoothing_window)
        
        if male_data and female_data:
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
            
            st.success("✅ Audio procesado correctamente! Prueba ajustar los parámetros si la separación no es óptima.")

st.markdown("---")
st.write("### Consejos para mejor resultado:")
st.write("""
- Ajusta el umbral de pitch según las voces específicas de tu audio
- Si las voces se mezclan demasiado, reduce la ventana de suavizado
- Si hay cortes bruscos, aumenta la ventana de suavizado
- Prueba diferentes combinaciones de parámetros para encontrar el mejor resultado
- La calidad de la separación depende mucho de la calidad del audio original
""")
