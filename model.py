import numpy as np
import librosa
import tensorflow as tf

# Load your trained model
model = tf.keras.models.load_model('best_model.keras')  # Ensure the path is correct

def load_audio(file):
    """Load an audio file using librosa."""
    audio_data, sample_rate = librosa.load(file, sr=None)
    return audio_data, sample_rate

def preprocess_audio(audio_data):
    """Preprocess the audio data for your model (modify as necessary)."""
    audio_data = np.expand_dims(audio_data, axis=0)  # Adjust shape for model input
    return audio_data

def predict_audio(file):
    """Load, preprocess, and predict audio."""
    audio_data, sample_rate = load_audio(file)
    processed_data = preprocess_audio(audio_data)

    # Make predictions using your model
    prediction = model.predict(processed_data)
    
    # Process the prediction result
    predicted_label = np.argmax(prediction)  # Modify based on your model output
    return predicted_label

# Test the model with an example audio file
if __name__ == "__main__":
    audio_file = 'test_files\OAF_base_angry.wav'  # Replace with your test audio file path
    prediction = predict_audio(audio_file)
    print("Predicted Label:", prediction)
