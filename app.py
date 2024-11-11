import streamlit as st
import librosa
import soundfile as sf
import numpy as np
import io
import os
import tempfile
import scipy.io.wavfile
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

st.set_page_config(page_title="Separador de Voces Inteligente", layout="wide")

st.title("🎤 Separador de Voces con Detección Automática")
st.write("Esta aplicación analiza y separa voces basándose en la detección automática de tonos dominantes.")

def analyze_pitch_distribution(y, sr, frame_length=2048, hop_length=512):
    """Analiza la distribución de pitch en el audio y encuentra los picos dominantes"""
    
    # Obtener características detalladas del pitch
    pitches, magnitudes = librosa.piptrack(
        y=y, 
        sr=sr,
        n_fft=frame_length,
        hop_length=hop_length,
        fmin=librosa.note_to_hz('C2'),
        fmax=librosa.note_to_hz('C7')
    )
    
    # Obtener el pitch predominante en cada frame
    pitch_values = []
    for i in range(magnitudes.shape[1]):
        index = magnitudes[:, i].argmax()
        if magnitudes[index, i] > 0:  # Solo considerar frames con suficiente energía
            pitch_values.append(pitches[index, i])
    
    pitch_values = np.array(pitch_values)
    
    # Crear histograma de pitch
    hist, bins = np.histogram(pitch_values, bins=100, range=(50, 400))
    
    # Suavizar el histograma para encontrar picos más claros
    hist_smooth = np.convolve(hist, np.hamming(10), mode='same')
    
    # Encontrar picos en el histograma suavizado
    peaks, _ = find_peaks(hist_smooth, distance=20, prominence=max(hist_smooth)*0.1)
    peak_frequencies = bins[peaks]
    peak_magnitudes = hist_smooth[peaks]
    
    # Ordenar picos por magnitud
    sorted_indices = np.argsort(peak_magnitudes)[::-1]
    dominant_peaks = peak_frequencies[sorted_indices][:2]  # Tomar los 2 picos más fuertes
    
    # Visualizar la distribución
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(bins[:-1], hist_smooth, label='Distribución de pitch')
    ax.plot(peak_frequencies, hist_smooth[peaks], "x", label='Picos detectados')
    for peak in dominant_peaks:
        ax.axvline(x=peak, color='r', linestyle='--', alpha=0.5)
    ax.set_xlabel('Frecuencia (Hz)')
    ax.set_ylabel('Cantidad')
    ax.set_title('Distribución de Pitch en el Audio')
    ax.legend()
    
    return dominant_peaks, fig

def separate_voice_by_pitch(y, sr, target_pitch, tolerance, smoothing_window):
    """Separa una voz específica basada en el pitch objetivo"""
    frame_length = 2048
    hop_length = 512
    
    # Obtener características del pitch
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
    
    pitch_mean = np.array(pitch_mean)
    
    # Crear máscara basada en el pitch objetivo y tolerancia
    mask = np.abs(pitch_mean - target_pitch) <= tolerance
    
    # Suavizar máscara
    mask_smooth = smooth_array(mask.astype(float), smoothing_window)
    
    # Aplicar máscara al audio
    mask_full = np.repeat(mask_smooth, hop_length)[:len(y)]
    voice_separated = y * mask_full
    
    return librosa.util.normalize(voice_separated)

def smooth_array(arr, window_size=5):
    kernel = np.ones(window_size) / window_size
    return np.convolve(arr, mode='same', a=arr, v=kernel)

def process_audio(audio_file, target_pitch, tolerance, smoothing_window):
    try:
        # Crear directorio temporal
        temp_dir = tempfile.mkdtemp()
        temp_input = os.path.join(temp_dir, "input.mp3")
        
        with open(temp_input, "wb") as f:
            f.write(audio_file.getvalue())
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Cargar audio
        status_text.text("Cargando audio...")
        progress_bar.progress(10)
        y, sr = librosa.load(temp_input, sr=None)
        y = librosa.util.normalize(y)
        
        # Separar la voz específica
        status_text.text("Separando voz...")
        progress_bar.progress(50)
        
        separated_voice = separate_voice_by_pitch(y, sr, target_pitch, tolerance, smoothing_window)
        
        # Convertir a int16
        voice_int = np.int16(separated_voice * 32767)
        
        # Guardar en buffer
        voice_buffer = io.BytesIO()
        scipy.io.wavfile.write(voice_buffer, sr, voice_int)
        
        progress_bar.progress(100)
        status_text.text("¡Proceso completado!")
        
        return voice_buffer.getvalue()
        
    except Exception as e:
        st.error(f"Error en el procesamiento: {str(e)}")
        return None
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
uploaded_file = st.file_uploader("Escoge un archivo MP3", type=['mp3'])

if uploaded_file is not None:
    # Botón para análisis inicial
    if st.button("Analizar Voces"):
        with st.spinner("Analizando distribución de pitch..."):
            # Cargar audio para análisis
            temp_dir = tempfile.mkdtemp()
            temp_input = os.path.join(temp_dir, "input.mp3")
            
            with open(temp_input, "wb") as f:
                f.write(uploaded_file.getvalue())
            
            y, sr = librosa.load(temp_input, sr=None)
            
            # Analizar distribución de pitch
            dominant_pitches, fig = analyze_pitch_distribution(y, sr)
            
            # Mostrar resultados
            st.write("### Pitches Dominantes Detectados:")
            for i, pitch in enumerate(dominant_pitches, 1):
                st.write(f"Pitch {i}: {pitch:.1f} Hz")
            
            # Mostrar gráfica
            st.pyplot(fig)
            
            # Guardar pitches en session state
            st.session_state['dominant_pitches'] = dominant_pitches
    
    # Parámetros para la separación
    if 'dominant_pitches' in st.session_state:
        st.write("### Configuración de Separación:")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            target_pitch = st.selectbox(
                "Pitch Objetivo (Hz)",
                options=st.session_state['dominant_pitches'],
                format_func=lambda x: f"{x:.1f} Hz"
            )
        
        with col2:
            tolerance = st.slider(
                "Tolerancia de Pitch (Hz)",
                min_value=10,
                max_value=50,
                value=30,
                help="Rango de frecuencias alrededor del pitch objetivo"
            )
        
        with col3:
            smoothing_window = st.slider(
                "Ventana de Suavizado",
                min_value=1,
                max_value=21,
                value=11,
                step=2
            )
        
        if st.button("Separar Voz"):
            separated_voice = process_audio(uploaded_file, target_pitch, tolerance, smoothing_window)
            
            if separated_voice:
                st.download_button(
                    label=f"⬇️ Descargar Voz ({target_pitch:.1f} Hz)",
                    data=separated_voice,
                    file_name=f"voz_{target_pitch:.1f}hz.wav",
                    mime="audio/wav"
                )
                
                st.success("✅ Voz separada correctamente!")

st.markdown("---")
st.write("### Cómo funciona:")
st.write("""
1. **Análisis Inicial**: El botón "Analizar Voces" examina el audio y detecta los pitches dominantes.
2. **Visualización**: Muestra un gráfico de la distribución de pitch y marca los picos detectados.
3. **Separación**: Puedes seleccionar uno de los pitches detectados y ajustar la tolerancia y suavizado.
4. **Ajuste Fino**: 
   - Aumenta la tolerancia si la voz suena entrecortada
   - Reduce la tolerancia si se filtran otras voces
   - Ajusta el suavizado para mejorar las transiciones
""")
