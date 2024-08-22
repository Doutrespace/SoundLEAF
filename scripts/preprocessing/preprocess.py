# Preprocessing data
import os
import numpy as np
import librosa
import soundfile as sf
from tqdm import tqdm
import tensorflow as tf
import tensorflow_datasets as tfds

# Import LEAF (make sure you have installed it: pip install leaf-audio)
from leaf_audio import frontend as leaf_frontend

def mel_spectrogram(audio_file, n_fft=2048, hop_length=512, n_mels=128):
    y, sr = librosa.load(audio_file, sr=None)
    S = librosa.feature.melspectrogram(y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    S_dB = librosa.power_to_db(S, ref=np.max)
    return S_dB

def leaf_representation(audio_file, sample_rate=16000, window_length_seconds=0.025, window_stride_seconds=0.010):
    # Load audio file
    y, sr = librosa.load(audio_file, sr=sample_rate)
    
    # Create LEAF frontend
    leaf = leaf_frontend.Leaf(
        n_filters=40,
        sample_rate=sample_rate,
        window_len=window_length_seconds,
        window_stride=window_stride_seconds,
        compression_fn="log",
    )
    
    # Preprocess audio
    audio_tensor = tf.convert_to_tensor(y[np.newaxis, :], dtype=tf.float32)
    leaf_features = leaf(audio_tensor)
    
    return leaf_features.numpy().squeeze()

def preprocess_dataset(dataset, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    mel_dir = os.path.join(output_dir, 'mel_spectrograms')
    leaf_dir = os.path.join(output_dir, 'leaf_representations')
    os.makedirs(mel_dir, exist_ok=True)
    os.makedirs(leaf_dir, exist_ok=True)

    for idx in tqdm(range(len(dataset)), desc="Preprocessing audio files"):
        audio_file, target = dataset[idx]
        file_id = os.path.splitext(os.path.basename(audio_file))[0]

        # Generate and save Mel spectrogram
        mel_spec = mel_spectrogram(audio_file)
        np.save(os.path.join(mel_dir, f"{file_id}_mel.npy"), mel_spec)

        # Generate and save LEAF representation
        leaf_rep = leaf_representation(audio_file)
        np.save(os.path.join(leaf_dir, f"{file_id}_leaf.npy"), leaf_rep)

        # Save target
        np.save(os.path.join(output_dir, f"{file_id}_target.npy"), target)

if __name__ == "__main__":
    from usm_dataset_loader import USMDataset  # Make sure this is in the same directory

    data_directory = '/Users/nilskarges/Documents/PhD/Dissertation/WuerzburgSoundscape/USMnet/data'
    subset = 'train'  # Change to 'val' or 'eval' as needed
    class_map_file = '/Users/nilskarges/Documents/PhD/Dissertation/WuerzburgSoundscape/USMnet/metadata/class_labels.csv'
    output_directory = '/Users/nilskarges/Documents/PhD/Dissertation/WuerzburgSoundscape/SoundLEAF/data/output'

    dataset = USMDataset(data_directory, subset, class_map_file)
    preprocess_dataset(dataset, output_directory)

    print(f"Preprocessing completed. Output saved to {output_directory}")