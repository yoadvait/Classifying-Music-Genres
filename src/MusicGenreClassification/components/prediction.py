import numpy as np
import librosa
import logging
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.saving import saved_model

logging.basicConfig(level=logging.INFO)

class GenrePredictor:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = self.load_model()

    def load_model(self):
        logging.info("Loading the model...")
        model = load_model(self.model_path)
        return model

    def process_track(self, track_path, num_mfcc=13, n_fft=2048, hop_length=512, ):
        logging.info(f"Processing track: {track_path}")
        signal, sr = librosa.load(track_path, sr=22050)
        mfcc = librosa.feature.mfcc(signal, sr=sr, n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)
        mfcc = mfcc.T  # Transpose to fit model input shape
        mfcc_input = mfcc[np.newaxis, ..., np.newaxis]  
        return mfcc_input

    def make_prediction(self, track_path):
        mfcc_input = self.process_track(track_path)
        prediction = self.model.predict(mfcc_input)
        predicted_class = np.argmax(prediction, axis=1)
        return predicted_class

if __name__ == "__main__":
    MODEL_PATH = "saved_models/saved_model.pb"
    TEST_TRACK_PATH = "path/to/test_track.wav"

    genre_predictor = GenrePredictor(model_path=MODEL_PATH)
    
    genre_prediction = genre_predictor.make_prediction(TEST_TRACK_PATH)
    logging.info(f"Predicted genre class: {genre_prediction}")
