import numpy as np
import logging
import os
import sys
from src.MusicGenreClassification.logger import setup_logger
from src.MusicGenreClassification.components.data_preprocessing import DataPreprocessor
from src.MusicGenreClassification.models.hybrid import HybridModel
from src.MusicGenreClassification.components.prediction import GenrePredictor

logger = setup_logger("MusicGenreClassification")

def main():
    logging.basicConfig(level=logging.INFO)

    dataset_path = "Data"
    mfccs_path = "Data/mfccs.npy"
    labels_path = "Data/labels.npy"
    model_save_path_hybrid = "saved_models/hybrid_model"

    num_classes = 10
    input_shape = (259, 13, 1)

    try:
        logging.info("Music genre classification pipeline started...")

        if not os.path.exists(mfccs_path) or not os.path.exists(labels_path):
            logging.info("Extracting MFCCs and labels...")
            preprocessor = DataPreprocessor(dataset_path)
            mfccs, labels = preprocessor.extract_mfcc()
            preprocessor.save_mfccs_and_labels(mfccs, labels, mfccs_path, labels_path)
            logging.info(f"MFCCs and labels saved to {mfccs_path} and {labels_path}.")
        else:
            logging.info(f"Using existing MFCCs and labels from {mfccs_path} and {labels_path}.")

        logging.info("Initiating model training...")
        trainer = HybridModel(input_shape=input_shape, num_classes=num_classes)
        trainer.train_model(mfccs_path, labels_path, model_save_path_hybrid)
        logging.info("Model training complete.")

        logging.info("Making predictions on test data...")
        test_file_path = "country.00092.wav"
        predictor = GenrePredictor(model_save_path_hybrid)
        predicted_genre = predictor.make_prediction(test_file_path)
        logging.info(f"Predicted Genre: {predicted_genre}")

    except Exception as e:
        logging.error(f"An error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()