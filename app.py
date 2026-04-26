import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
from streamlit_option_menu import option_menu  # type: ignore
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, WebRtcMode

# Load your pre-trained model
model = tf.keras.models.load_model('best_model.keras')

# Emotion mapping dictionary
emotion_mapping = {
    0: "😠 Angry",
    1: "🤢 Disgust",
    2: "😨 Fear",
    3: "😊 Happy",
    4: "😐 Neutral",
    5: "😢 Sad",
    6: "😲 Surprise"
}

# Function to preprocess audio
def preprocess_audio(audio_data):
    """Preprocess the audio data for your model."""
    audio_data = librosa.resample(audio_data, orig_sr=44100, target_sr=16000)  # Resample to 16kHz if needed
    audio_data = np.expand_dims(audio_data, axis=0)
    return audio_data

# Sidebar Navigation
st.set_page_config(page_title="Speech Emotion Recognition (SER) Model", layout="wide", page_icon="🧑‍⚕️")

with st.sidebar:
    page = option_menu('SER Model',
                       ['Home', 'Model Tester', 'About', 'About Us'],
                    #    ['Home', 'Model Tester', 'Live Audio', 'About'],
                       icons=['house', 'mic', 'record-circle', 'info-circle'],
                       default_index=0)

# Home Page
if page == "Home":
    st.title("Speech Emotion Recognition (SER) Model")
    st.image(r"img/voice-recognition-technology-conceptual-image-39055870.webp", use_container_width=False)
    st.write("""
    Welcome to the **Speech Emotion Recognition (SER) Model Tester**!

    This app uses a deep learning model to detect emotions from speech audio files. The model can identify the following emotions:
    
    - 😠 Angry
    - 🤢 Disgust
    - 😨 Fear
    - 😊 Happy
    - 😐 Neutral
    - 😢 Sad
    - 😲 Surprise
    
    **Use Cases:**
    - Improving customer service by detecting emotions in call center conversations
    - Mental health monitoring through emotional state analysis
    - Enhancing human-computer interaction by recognizing user emotions
    - Real-time emotion detection in virtual assistants
    - Content creation for emotionally adaptive games or applications

    Upload an audio file on the **Model Tester** page to see the prediction in action!
    """)

# Model Tester Page
elif page == "Model Tester":
    st.title("Speech Recognition Model Tester")
    audio_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])

    if audio_file is not None:
        audio_data, sample_rate = librosa.load(audio_file, sr=16000)
        processed_data = preprocess_audio(audio_data)
        prediction = model.predict(processed_data)
        predicted_label = np.argmax(prediction)
        emotion_name = emotion_mapping[predicted_label]
        st.write("🎙️ **Prediction Result:** ", emotion_name)

# # Live Audio Page
# elif page == "Live Audio":
#     st.title("🎙️ Live Audio Emotion Recognition")

#     class EmotionAudioProcessor(AudioProcessorBase):
#         def __init__(self):
#             self.model = model

#         def recv(self, frame):
#             audio_data = frame.to_ndarray().flatten().astype(np.float32)
#             processed_data = preprocess_audio(audio_data)
#             prediction = self.model.predict(processed_data)
#             predicted_label = np.argmax(prediction)
#             emotion_name = emotion_mapping[predicted_label]
#             return emotion_name

#     webrtc_ctx = webrtc_streamer(
#         key="emotion-recognition",
#         mode=WebRtcMode.SENDRECV,
#         audio_processor_factory=EmotionAudioProcessor,
#         media_stream_constraints={"audio": True, "video": False},
#         async_processing=True,
#     )

# About Page
elif page == "About":
    st.title("About This App")
    st.write("""
    This app was created to demonstrate how a Speech Emotion Recognition model works.

    **Technologies Used:**
    - Python
    - Streamlit
    - TensorFlow
    - Librosa

    **Developer:** Mulsaniya Harsh, Mohd Anas
    """)

# Data for team members
member_info = {
    "Mulsaniya Harsh": {
        "role": "Data Analyst and Scientist",
        "image": "img/avatar.png"
    },
    "Narendra Singh Rathore": {
        "role": "Data Analyst and Scientist",
        "image": "img/avatar.png"
    },
    
}

# Team members in a list
team_members = list(member_info.keys())

if page == 'About Us':
    st.title("About Us")
    st.write("This is a Multiple Disease Prediction System developed by a student from the Department of Information Technology Arya college of engineering , Jaipur.")
    st.write("Our team consists of:")

    for name, info in member_info.items():
        col1, col2, col3 = st.columns([1, 2, 1])  # Create a layout with columns
        
        with col2:  # The image will be in the center column
            st.image(info["image"], width=150)

        with col1:  # Text appears when hovered over or clicked on
            with st.expander(name):
                st.write(info["role"])
