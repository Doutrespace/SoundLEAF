import os
import numpy as np
import pandas as pd
from tqdm import tqdm

class USMDataset:
    def __init__(self, data_dir, subset, class_map_path):
        self.data_dir = os.path.join(data_dir, subset)
        self.class_map = self.load_class_map(class_map_path)
        self.audio_files, self.targets = self.load_data()

    def load_class_map(self, class_map_path):
        df = pd.read_csv(class_map_path)
        return {class_name: idx for idx, class_name in enumerate(df['class_name'])}

    def load_data(self):
        audio_files = []
        targets = []
        csv_file = os.path.join(os.path.dirname(self.data_dir), 'metadata', f'usm_{os.path.basename(self.data_dir)}.csv')
        df = pd.read_csv(csv_file)

        for _, row in tqdm(df.iterrows(), total=df.shape[0], desc=f"Loading {os.path.basename(self.data_dir)} data"):
            audio_file = f"{row['usm_id']}_mix.wav"
            target_file = f"{row['usm_id']}_mix_targets.npy"
            audio_path = os.path.join(self.data_dir, audio_file)
            target_path = os.path.join(self.data_dir, target_file)

            if os.path.exists(audio_path) and os.path.exists(target_path):
                audio_files.append(audio_path)
                targets.append(np.load(target_path))
            else:
                print(f"Missing file: {audio_file} or {target_file}")

        return audio_files, np.array(targets)

    def __getitem__(self, idx):
        return self.audio_files[idx], self.targets[idx]

    def __len__(self):
        return len(self.audio_files)

if __name__ == "__main__":
    data_directory = '/Users/nilskarges/Documents/PhD/Dissertation/WuerzburgSoundscape/USMnet/data'
    subset = 'train'  # Change to 'val' or 'eval' as needed
    class_map_file = '/Users/nilskarges/Documents/PhD/Dissertation/WuerzburgSoundscape/USMnet/metadata/class_labels.csv'
    dataset = USMDataset(data_directory, subset, class_map_file)
    print(f"Loaded {len(dataset)} items.")
    
    # Example: Print the first item
    audio_file, target = dataset[0]
    print(f"First audio file: {audio_file}")
    print(f"First target shape: {target.shape}")