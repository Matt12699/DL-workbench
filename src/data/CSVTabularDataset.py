import torch
from torch.utils.data import Dataset
import pandas as pd

class CSVTabularDataset(Dataset):

    def __init__(self, csv_path, target_column):

        # Carico il dataset preprocessato
        df = pd.read_csv(csv_path)

        # Salvo le feature (X) e la variabile target (y)
        self.X = df.drop(columns=[target_column]).values
        self.y = df[target_column].values

    def __len__(self):
        # Numero totale di esempi
        return len(self.X)
    
    def __getitem__(self, idx):
        # Restituisce l'esempio idx-esimo come tensore
        features = torch.tensor(self.X[idx], dtype=torch.float32)
        label = torch.tensor(self.y[idx], dtype=torch.long)
        return features, label