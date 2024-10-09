import numpy as np
import logging
from keras import Sequential, LSTM, Dense, Dropout, BatchNormalization, Adam, EarlyStopping, to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

logging.basicConfig(level=logging.INFO)

def load_data(mfccs_path, labels_path):

    logging.info("Loading MFCCs and labels...")
    mfccs = np.load(mfccs_path)
    labels = np.load(labels_path)
    return mfccs, labels

def build_model(input_shape, num_classes):

    logging.info("Building the LSTM model...")
    
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        BatchNormalization(),
        
        LSTM(64, return_sequences=False),
        Dropout(0.2),
        BatchNormalization(),
        
        Dense(128, activation='relu'),
        Dropout(0.2),
        
        Dense(64, activation='relu'),
        Dropout(0.2),
        
        Dense(num_classes, activation='softmax')
    ])
    
    return model

def train_model(mfccs_path, labels_path, model_save_path, num_classes):

    mfccs, labels = load_data(mfccs_path, labels_path)
    
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)
    labels_categorical = to_categorical(labels_encoded, num_classes)

    X_train, X_test, y_train, y_test = train_test_split(mfccs, labels_categorical, test_size=0.2, random_state=42)

    model = build_model(input_shape=(130, 13), num_classes=num_classes)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

    early_stopping = EarlyStopping(monitor='val_accuracy', patience=15, restore_best_weights=True)

    logging.info("Training the model...")
    model.fit(X_train, y_train, epochs=200, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stopping])
    
    model.save(model_save_path)
    logging.info(f"Model saved to {model_save_path}")

if __name__ == "__main__":

    MFCCS_PATH = "Data/mfccs.npy"
    LABELS_PATH = "Data/labels.npy"
    MODEL_SAVE_PATH = "saved_models/lstm_model.h5"
    NUM_CLASSES = 10 

    train_model(MFCCS_PATH, LABELS_PATH, MODEL_SAVE_PATH, NUM_CLASSES)
