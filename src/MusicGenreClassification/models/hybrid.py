import numpy as np
import logging

from keras._tf_keras.keras.optimizers import Adam
from keras._tf_keras.keras.layers import Dense, Dropout, Flatten, Reshape, MaxPooling2D, Conv2D, LSTM, BatchNormalization
from keras._tf_keras.keras.utils import to_categorical
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

logging.basicConfig(level=logging.INFO)

class HybridModel:
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self.build_model()

    def load_data(self, mfccs_path, labels_path):
        logging.info("Loading MFCCs and labels...")
        mfccs = np.load(mfccs_path)
        labels = np.load(labels_path)
        return mfccs, labels

    def build_model(self):
        logging.info("Building the hybrid CNN-LSTM model...")
        model = Sequential([
            Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=self.input_shape),
            MaxPooling2D((2, 2), padding='same'),
            BatchNormalization(),
            Dropout(0.3),

            Conv2D(64, (3, 3), activation='relu', padding='same'),
            MaxPooling2D((2, 2), padding='same'),
            BatchNormalization(),
            Dropout(0.3),

            Conv2D(64, (3, 3), activation='relu', padding='same'),
            MaxPooling2D((2, 2), padding='same'),
            BatchNormalization(),
            Dropout(0.3),

            Flatten(),

            Reshape((128,33)),

            LSTM(128, return_sequences=True),
            Dropout(0.3),

            LSTM(64, return_sequences=False),
            Dropout(0.3),

            Dense(64, activation='relu'),
            Dropout(0.3),

            Dense(32, activation='relu'),
            Dropout(0.3),

            Dense(self.num_classes, activation='softmax')
        ])
        return model


    def train_model(self, mfccs_path, labels_path, model_save_path):
        mfccs, labels = self.load_data(mfccs_path, labels_path)

        if mfccs.size == 0 or labels.size == 0:
            logging.error("No data found in the loaded MFCCs or labels. Please check the input files.")

        mfccs = mfccs.reshape(-1, 259, 13, 1)
            
        label_encoder = LabelEncoder()
        labels_encoded = label_encoder.fit_transform(labels)
        labels_categorical = to_categorical(labels_encoded, self.num_classes)
        if len(mfccs) > 0:
            X_train, X_test, y_train, y_test = train_test_split(mfccs, labels_categorical, test_size=0.2, random_state=42)
        else:
            logging.error("No samples to split. Check the input data.")
            return

        adam1 = Adam(learning_rate=0.001)
        self.model.compile(optimizer=adam1, loss='categorical_crossentropy', metrics=['accuracy'])
        early_stopping = EarlyStopping(monitor='val_accuracy', patience=20, restore_best_weights=True)

        logging.info("Training the model...")
        self.model.fit(X_train, y_train, epochs=300, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stopping])

        self.model.save(model_save_path)

        self.model.sa
        logging.info(f"Model saved to {model_save_path}")

if __name__ == "__main__":
    MFCCS_PATH = "Data/mfccs.npy"
    LABELS_PATH = "Data/labels.npy"
    MODEL_SAVE_PATH = "saved_models.hybrid.h5"
    NUM_CLASSES = 10

    hybrid_model = HybridModel(input_shape=(259, 13, 1), num_classes=NUM_CLASSES)
    hybrid_model.train_model(MFCCS_PATH, LABELS_PATH, MODEL_SAVE_PATH)