import streamlit as st
from PIL import Image
import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis

# --- Configuración de la Página ---
st.set_page_config(
    page_title="Face Swap App",
    page_icon="🎭",
    layout="wide"
)

# --- Inicialización del Modelo (se ejecuta una sola vez) ---
@st.cache_resource
def load_model():
    """Carga el modelo de insightface y el swapper."""
    app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))
    swapper = insightface.model_zoo.get_model('inswapper_128.onnx', download=True, download_zip=True, providers=['CPUExecutionProvider'])
    return app, swapper

app, swapper = load_model()

# --- Función para el Intercambio de Caras ---
def swap_face(source_img, target_img, face_analyzer, face_swapper):
    """
    Realiza el intercambio de caras usando insightface.
    :param source_img: Imagen con la cara de origen (en formato OpenCV BGR).
    :param target_img: Imagen de destino (en formato OpenCV BGR).
    :param face_analyzer: El analizador de caras de insightface.
    :param face_swapper: El modelo de intercambio de caras de insightface.
    :return: La imagen resultante con la cara intercambiada.
    """
    try:
        # Analizar las caras en ambas imágenes
        source_faces = face_analyzer.get(source_img)
        target_faces = face_analyzer.get(target_img)

        if not source_faces:
            st.error("No se encontró ninguna cara en la imagen de origen.")
            return None
        if not target_faces:
            st.error("No se encontró ninguna cara en la imagen de destino.")
            return None

        # Usar la primera cara encontrada como origen
        source_face = source_faces[0]
        
        # Realizar el swap en la imagen de destino
        result_img = face_swapper.get(target_img, target_faces[0], source_face, paste_back=True)
        return result_img

    except Exception as e:
        st.error(f"Ocurrió un error durante el proceso de face swap: {e}")
        return None

# --- Interfaz de Streamlit ---
st.title("🎭 Aplicación de Intercambio de Caras (Face Swap)")
st.write("Sube una imagen de origen (la cara que quieres usar) y una imagen de destino.")

col1, col2 = st.columns(2)

with col1:
    source_image_file = st.file_uploader("Elige la imagen de ORIGEN", type=['jpg', 'jpeg', 'png'])
    if source_image_file:
        st.image(source_image_file, caption="Imagen de Origen", use_column_width=True)

with col2:
    target_image_file = st.file_uploader("Elige la imagen de DESTINO", type=['jpg', 'jpeg', 'png'])
    if target_image_file:
        st.image(target_image_file, caption="Imagen de Destino", use_column_width=True)

if st.button("✨ Realizar Intercambio de Caras", use_container_width=True) and source_image_file and target_image_file:
    with st.spinner('Procesando...'):
        # Convertir archivos subidos a formato OpenCV
        source_pil = Image.open(source_image_file).convert('RGB')
        target_pil = Image.open(target_image_file).convert('RGB')
        source_cv = np.array(source_pil)[:, :, ::-1] # RGB -> BGR
        target_cv = np.array(target_pil)[:, :, ::-1] # RGB -> BGR

        result_image = swap_face(source_cv, target_cv, app, swapper)

        if result_image is not None:
            # Convertir resultado de vuelta a RGB para mostrar en Streamlit
            result_image_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
            st.success("¡Intercambio completado!")
            st.image(result_image_rgb, caption="Resultado del Face Swap", use_column_width=True)
