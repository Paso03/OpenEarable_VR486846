import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold, StratifiedKFold
import os

class CSVDataModule(pl.LightningDataModule):
    def __init__(self, root_dir, batch_size=32, num_workers=4, k_folds=3, window_size=1, overlap=1, sample_rate=30):
        super().__init__()
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.k = k_folds

        self.window_size = window_size
        self.overlap = overlap
        self.sample_rate = sample_rate

        self.legend = None
        self.x, self.y = None, None

        self.train_indices = []
        self.val_indices = []

    def setup(self, stage=None):
        # Creazione del dataset e preparazione degli split per la cross validation
        self.x, self.y = self.create_dataset()
        k_fold = KFold(n_splits=self.k, shuffle=True, random_state=42)

        for train_idx, val_idx in k_fold.split(self.x):
            self.train_indices.append(train_idx)
            self.val_indices.append(val_idx)

    def windowing(self, data):
        windows, labels = [], []

        # Genera finestre di dati basate su window_size e overlap
        for i in range(0, len(data) - self.window_size * self.sample_rate, int(self.overlap * self.sample_rate)):
            window = data.iloc[i:i + self.window_size * self.sample_rate]
            label = window["label"].iloc[0]
            # Rimuove la colonna label; così la finestra contiene solo dati dei sensori
            window = window.drop(columns=["label"])
            windows.append(window.values)
            labels.append(label)

        return np.array(windows), np.array(labels)

    def create_dataset(self):
        x, y = [], []
        for file in os.listdir(self.root_dir):
            if file.endswith(".csv"):
                df = pd.read_csv(
                    os.path.join(self.root_dir, file),
                    index_col=0,
                    header=None,
                    # Vengono assegnati nomi a tutte le colonne, ma si utilizzeranno solo le sensibili
                    names=["time", "acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z",
                           "mag_x", "mag_y", "mag_z", "roll", "pitch", "yaw"],
                    # Seleziona solo le colonne di accelerometro e giroscopio, escludendo time e le altre
                    usecols=["time", "acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"],
                )
                # Aggiunge la colonna label estratta dal nome del file (ad esempio "Testa_giù-su" o altro)
                df["label"] = file.split("_")[1].split(".")[0]
                x_i, y_i = self.windowing(df)
                x.extend(x_i)
                y.extend(y_i)

        x, y = np.array(x), np.array(y)

        # Label encoding per ottenere una codifica one-hot
        self.legend, y = self.labels_encoding(y)

        return x, y

    @staticmethod
    def labels_encoding(y):
        categories, inverse = np.unique(y, return_inverse=True)
        # Crea la matrice one-hot
        one_hot = np.zeros((y.size, categories.size))
        one_hot[np.arange(y.size), inverse] = 1
        return categories, one_hot.astype(np.float64)

    def prepare_dataset(self, indexes, fold):
        x = torch.tensor(self.x[indexes[fold]], dtype=torch.float32)
        y = torch.tensor(self.y[indexes[fold]], dtype=torch.float32)
        return torch.utils.data.TensorDataset(x, y)

    def train_dataloader(self, fold=0):
        return DataLoader(
            self.prepare_dataset(self.train_indices, fold),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            persistent_workers=True
        )

    def val_dataloader(self, fold=0):
        return DataLoader(
            self.prepare_dataset(self.val_indices, fold),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            persistent_workers=True
        )
