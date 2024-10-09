import numpy as np
import librosa
import math
import os
import logging

class DataPreprocessor:
    def __init__(self, dataset_path, num_mfcc=13, n_fft=2048, hop_length=512, num_segments=5, sr=22050, tracklen=30):
        self.dataset_path = dataset_path
        self.num_mfcc = num_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.num_segments = num_segments
        self.sr = sr
        self.tracklen = tracklen

    def extract_mfcc(self):
        mfccs = []
        labels = []

        samples_per_segment = int((self.sr * self.tracklen) / self.num_segments)
        num_mfcc_vectors_per_segment = math.ceil(samples_per_segment / self.hop_length)

        for i, (dirpath, dirnames, filenames) in enumerate(os.walk(self.dataset_path)):
            if dirpath is not self.dataset_path:
                semantic_label = dirpath.split("/")[-1]

                for f in filenames:
                    file_path = os.path.join(dirpath, f)

                    try:
                        signal, sample_rate = librosa.load(file_path, sr=self.sr)

                        for d in range(self.num_segments):
                            start = samples_per_segment * d
                            finish = start + samples_per_segment

                            mfcc = librosa.feature.mfcc(y=signal[start:finish], sr=sample_rate, n_mfcc=self.num_mfcc, n_fft=self.n_fft, hop_length=self.hop_length)
                            mfcc = mfcc.T

                            if len(mfcc) == num_mfcc_vectors_per_segment:
                                mfccs.append(mfcc)
                                labels.append(semantic_label)

                    except Exception as e:
                        logging.error(f"Error processing file {file_path}: {e}")

        mfccs = np.array(mfccs)
        labels = np.array(labels)

        return mfccs, labels

    def save_mfccs_and_labels(self, mfccs, labels, mfccs_path="mfccs.npy", labels_path="labels.npy"):
        np.save(mfccs_path, mfccs)
        np.save(labels_path, labels)
        logging.info("MFCCs and labels saved to disk.")

def main():
    dataset_path = "Data/genres_original"
    
    logging.info("Starting MFCC extraction process...")
    preprocessor = DataPreprocessor(dataset_path)
    mfccs, labels = preprocessor.extract_mfcc()
    preprocessor.save_mfccs_and_labels(mfccs, labels)

if __name__ == "__main__":
    main()
