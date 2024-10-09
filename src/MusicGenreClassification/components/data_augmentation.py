# not included in the pipeline (data not augmentated)

import os
import soundfile as sf
import librosa
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift, Gain
from tqdm import tqdm

os.makedirs('augmented_data', exist_ok=True)

augment = Compose([
    AddGaussianNoise(min_amplitude=0.005, max_amplitude=0.025, p=0.7),
    TimeStretch(min_rate=0.9, max_rate=1.2, p=0.5),
    PitchShift(min_semitones=-3, max_semitones=3, p=0.6),
    Shift(min_shift=-0.4, max_shift=0.4, p=0.6),
    Gain(min_gain_in_db=-10, max_gain_in_db=10, p=0.75)
])

def apply_n_keep(file, save_dir='augmented_data', augmentations=5):

    augmented_files = []
    
    try:
        y, sr = librosa.load(file, res_type='kaiser_fast')
        
        for i in range(augmentations):
            augmented_samples = augment(samples=y, sample_rate=sr)
            augmented_file_path = os.path.join(save_dir, f"aug_{i}_{os.path.basename(file)}")
            sf.write(augmented_file_path, augmented_samples, sr)
            augmented_files.append(augmented_file_path)
    
    except Exception as e:
        print(f"Error processing {file}: {e}")
    
    return augmented_files

def augment_dataset(data, save_dir='augmented_data'):

    augmented_file_paths = []

    for index_num, row in tqdm(data.iterrows(), total=data.shape[0]):
        file_name = os.path.join(os.path.abspath("Data/genres_original"), row['label'], str(row['filename']))
        augmented_files = apply_n_keep(file_name, save_dir)
        augmented_file_paths.extend([(aug_file, row['label']) for aug_file in augmented_files])
    
    return augmented_file_paths