import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import scipy.io


class DataUtilEEG:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.class_weights = None  # Class weights
        self.global_mean = None
        self.global_std = None

    def load_data(self):
        """Naloži podatke iz `.mat` datoteke."""
        mat_data = scipy.io.loadmat(self.file_path)
        if 'ALLEEG' in mat_data:
            self.data = mat_data['ALLEEG']
        else:
            raise ValueError("Ključ 'ALLEEG' ni najden v .mat datoteki. Preveri strukturo!")

    def process_data(self):
        """Naloži EEG podatke in vrne standardizirane vrednosti."""
        if self.data is None:
            raise ValueError("Podatki niso naloženi. Najprej pokliči `load_data()`.")

        eeg_data = self.data
        first_record = eeg_data[0][0]
        fixed_length = first_record['pnts'][0][0]  # Dolžina posnetka v točkah

        X_all = []
        y_all = []

        for record in eeg_data[0]:
            data = record['data']  # Shape: (channels, timepoints, epochs)
            condition = record['condition'][0]
            
            condition_str = str(condition) if isinstance(condition, np.str_) else condition.decode()
            label = 1 if condition_str == "After" else 0

            for trial_idx in range(data.shape[2]):
                trial_data = data[:, :, trial_idx]

                if trial_data.shape[1] < fixed_length:
                    padding = fixed_length - trial_data.shape[1]
                    trial_data = np.pad(trial_data, ((0, 0), (0, padding)), mode='constant')
                elif trial_data.shape[1] > fixed_length:
                    trial_data = trial_data[:, :fixed_length]

                X_all.append(trial_data)
                y_all.append(label)

        self.X = np.array(X_all, dtype=np.float32)
        self.X = np.transpose(self.X, (0, 2, 1))  # (samples, timepoints, channels)
        self.y = np.array(y_all, dtype=np.int32)

        # Izračunaj globalno povprečje in standardni odklon
        self.compute_global_standardization()

        # Uporabi globalno standardizacijo na vseh podatkih
        self.X = (self.X - self.global_mean) / (self.global_std + 1e-8)

    def compute_global_standardization(self):
        """Izračuna globalno povprečje in std za standardizacijo EEG podatkov."""
        self.global_mean = np.mean(self.X, axis=(0, 1), keepdims=True)  # Povprečje čez vse vzorce in čas
        self.global_std = np.std(self.X, axis=(0, 1), keepdims=True)  # Standardni odklon

    def split_data(self, test_size=0.3, val_size=0.2, random_state=42, shuffle=True):
        """Razdeli podatke na train, val in test in izračuna class weights."""
        if self.X is None or self.y is None:
            raise ValueError("Podatki niso procesirani. Pokliči `process_data()`.")

        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state, shuffle=shuffle, stratify=self.y
        )

        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=val_size_adjusted, random_state=random_state, shuffle=shuffle, stratify=y_train
        )

        self.X_train, self.X_val, self.X_test = X_train, X_val, X_test
        self.y_train, self.y_val, self.y_test = y_train, y_val, y_test

        # Izračunaj class weights
        self.compute_class_weights()

    def compute_class_weights(self):
        """Izračuna class weights za uravnoteženje razreda v modelu."""
        class_weights = compute_class_weight(
            class_weight='balanced', classes=np.unique(self.y_train), y=self.y_train
        )
        self.class_weights = {i: class_weights[i] for i in range(len(class_weights))}

    def create_tf_datasets(self):
        """Ustvari `tf.data.Dataset` brez batch-ov in augmentacije."""
        if self.X_train is None or self.X_val is None or self.X_test is None:
            raise ValueError("Data splits are not initialized. Run `split_data()` first.")

        self.train_dataset = tf.data.Dataset.from_tensor_slices((self.X_train, self.y_train))
        self.val_dataset = tf.data.Dataset.from_tensor_slices((self.X_val, self.y_val))
        self.test_dataset = tf.data.Dataset.from_tensor_slices((self.X_test, self.y_test))

        print(f"Train: {len(self.X_train)} samples, Val: {len(self.X_val)} samples, Test: {len(self.X_test)} samples")

    def get_class_weights(self):
        """Vrne class weights, ki jih lahko uporabiš pri modelu."""
        return self.class_weights

    def get_summary(self):
        """Prints dataset information."""
        print(f"Train shape: {self.X_train.shape}, Val shape: {self.X_val.shape}, Test shape: {self.X_test.shape}")
        print(f"Class weights: {self.class_weights}")
