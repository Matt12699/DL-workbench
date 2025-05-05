import sys
import logging

import pandas as pd
import numpy as np
import torch

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from utilities.logging_config import setup_logging
from utilities.argument_parser import ArgumentParser
from utilities.config_manager import ConfigManager
from data.log_min_max_scaler import LogMinMaxScaler
from data.custom_imputer import CustomImputer
from data.frequency_encoder import FrequencyEncoder
from data.custom_binner import CustomBinner
from data.CSVTabularDataset import CSVTabularDataset
from torch.utils.data import random_split, DataLoader
from torch import optim
from torch import nn
from model.IDSModel import IDSModel

setup_logging()
logger = logging.getLogger(__name__)


def prepare_data(input_path: str, output_path: str, dataset_name: str):
    logger.info("Preparing data...")

    # Carico la configurazione dal JSON
    config_path= "config/dataset.json"
    config_manager = ConfigManager()
    config_manager.load_config(config_path)

    # Ottengo le colonne dal JSON
    numeric_columns = config_manager.get_value(dataset_name, "numeric_columns")
    categorical_columns = config_manager.get_value(dataset_name, "categorical_columns")
    target_column = config_manager.get_value(dataset_name, "target_column")

    # Leggo il DataFrame da un file csv
    df = pd.read_csv(input_path, nrows=1000000) 
    
    # Pipeline per le feature numeriche
    # CustomImputer mi permette di gestire i valori NaN e infiniti
    # Log-Min-Max Scaler: Faccio il logaritmo delle feature numeriche e 
    # poi riscalo tutte le feature numeriche in modo tale che si trovino tutte in un range che va da -1 a 1
    num_pipeline = Pipeline([
        ("Impute", CustomImputer()),
        ("Binning", CustomBinner())
       # ("Scaling", LogMinMaxScaler()), (commentato per analizzare i dati)
    ])

    # Pipeline per le feature categoriche 
    # FrequencyEncoder mappa i valori delle feature categoriche a un valore in base alla loro frequenza
    # OneHotEncoder crea una colonna per ogni valore della feature 
    cat_pipeline = Pipeline([
        ("FrequencyEncoder", FrequencyEncoder(soglia=0.5)),
        ("1hot", OneHotEncoder(sparse_output = False)),
    ])

    # ColumnTransformer prende tutte le colonne e fa le trasformazioni giuste su feature numeriche e categoriche
    preProcessing = ColumnTransformer([
        ("num", num_pipeline, numeric_columns),
        ("cat", cat_pipeline, categorical_columns),
    ])

    # Dataset processato
    df_prepared = preProcessing.fit_transform(df)
    
    # Rimetto i nomi delle feature che si sono persi nel processing

    # Genero i nomi delle colonne categoriche (ottenuti tramite get_feature_names_out)
    one_hot_column_names = preProcessing.transformers_[1][1].named_steps["1hot"].get_feature_names_out(categorical_columns)

    # Combino i nomi delle colonne numeriche e quelle generate dal OneHotEncoder
    all_columns = numeric_columns + list(one_hot_column_names)

    # Crea il DataFrame con i nomi delle colonne
    df_prepared_df = pd.DataFrame(df_prepared, columns=all_columns)

    train_ratio = 0.9

    # Aggiungo la colonna target al DataFrame processato
    df_prepared_df[target_column] = df[target_column]

    # Suddivido il dataset in sottogruppi 
    train_df, test_df = train_test_split(df_prepared_df, test_size=(1 - train_ratio), random_state=None)

    if output_path!=None:
        train_df.to_csv(output_path + "_Train.csv", index=False)
        test_df.to_csv(output_path + "_Test.csv", index=False)
        logging.info("Saved Datasets")

def train_model(processed_csv_path: str, dataset_name: str):

    logging.info("Training model...")

    # Carico la configurazione dal JSON
    config_path= "config/dataset.json"
    config_manager = ConfigManager()
    config_manager.load_config(config_path)

    target_column = config_manager.get_value(dataset_name, "target_column")

    full_dataset = CSVTabularDataset(processed_csv_path, target_column=target_column)

    # Ottengo il numero di feature dal dataset 
    num_features = full_dataset.X.shape[1]

    train_ratio = 0.9

    n_total = len(full_dataset)
    n_train = int (train_ratio * n_total)
    n_val = n_total - n_train
    
    # Suddivido il dataset in due sottogruppi in modo casuale
    train_dataset , val_dataset = random_split(full_dataset, [n_train, n_val])

    # Divido i campioni in batch
    train_dataLoader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_dataLoader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    # Selezioniamo il dispositivo da usare: GPU, CPU...
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

    model = IDSModel(num_features=num_features).to(device)

    # Definisco la Loss Function
    criterion = nn.CrossEntropyLoss()

    # Definisco l'algoritmo di ottimizzazione
    optimizer = optim.SGD(model.parameters(),
                          lr=0.001,
                          momentum=0.9)
    
    # Definisco il training Loop
    # Numero di epoche
    N_EPOCHS = 15

    # Training
    # Processiamo l'intero training set in 10 epoche
    for epoch in range(N_EPOCHS):

        # Per ogni epoca capiamo quanta è la loss
        train_loss= 0.0
        model.train()
        for inputs, labels in train_dataLoader:

            # Sposta input e label alla GPU se è disponibile
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Azzero i gradienti accumulati dai passsaggi precedenti
            # Serve a far si che l'ottimizzatore aggiorni i parametri del modello correttamente
            optimizer.zero_grad()

            # Passo l'input al modello per ottenere l'output
            outputs = model(inputs)

            # Calcolo la loss
            loss=criterion(outputs, labels)

            # Backpropagation: calcolo i gradienti
            loss.backward()

            # Aggiusto i parametri basati sui gradienti
            optimizer.step()

            train_loss += loss.item()

        # Validation
        val_loss = 0.0
        model.eval()
        for inputs, labels in val_dataLoader:

            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)

            loss = criterion(outputs, labels)

            val_loss += loss.item()

        print(f"Epoch: {epoch} Train Loss: {train_loss/len(train_dataLoader):.4f} Val Loss: {val_loss/len(val_dataLoader):.4f}")
    
    # Salvo il modello alla fine dell'allenamento
    save_path = "src/model/trained_model.pth"
    torch.save(model.state_dict(), save_path)
    logging.info("Saved model")

    
def evaluate_model(model_path: str, processed_csv_path: str, dataset_name: str):

    logging.info("Evaluating model...")

    # Carico la configurazione dal JSON
    config_path= "config/dataset.json"
    config_manager = ConfigManager()
    config_manager.load_config(config_path)

    target_column = config_manager.get_value(dataset_name, "target_column")

    test_dataset = CSVTabularDataset(processed_csv_path, target_column=target_column)

    # Ottengo il numero di feature dal dataset 
    num_features = test_dataset.X.shape[1]

    # Divido i campioni in batch
    test_dataLoader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Selezioniamo il dispositivo da usare: GPU, CPU...
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

    model = IDSModel(num_features=num_features).to(device)

    # Carico lo stato del modello
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except FileNotFoundError:
        print(f"Errore: File del modello non trovato in: {model_path}")
        # return None
    except RuntimeError as e:
        print(f"Errore durante il caricamento del modello: {e}")
        # return None

    num_correct = 0.0
    for x_test_batch, y_test_batch in test_dataLoader:

        model.eval()
        y_test_batch = y_test_batch.to(device)
        x_test_batch = x_test_batch.to(device)

        # Output per ogni batch
        y_pred_batch = model(x_test_batch)

        # Per ogni batch, estraggo la classe di appartenenza con la probabilità più alta
        _, predicted = torch.max(y_pred_batch, 1)

        # Calcolo quante sono le predizioni corrette
        num_correct += (predicted == y_test_batch).float().sum()
    
    # Calcolo l'accuratezza del mio modello
    accuracy = num_correct / (len(test_dataLoader) * test_dataLoader.batch_size)

    print(f"Test Accuracy: {accuracy:.4f}")

    


if __name__ == "__main__":
    parser = ArgumentParser("fl-ids")

    parser.register_subcommands(
        "prepare",
        ["--input", "--output", "--dataset"],
        ["The input path for the data.", "The output path for the prepared data.", "The name of the dataset"],
    )

    parser.register_subcommands(
        "train",
        ["--input", "--dataset"],
        ["The input path for the processed data.", "The name of the dataset"],
    )

    parser.register_subcommands(
        "evaluate",
        ["--model", "--input", "--dataset"],
        ["The path for the processed model", "The input path for the processed data.", "The name of the dataset"],
    )

    args = parser.parse_arguments(sys.argv[1:])

    if args.subcommand == "prepare":
        prepare_data(args.input, args.output, args.dataset)
    elif args.subcommand == "train":
        train_model(args.input, args.dataset)
    elif args.subcommand == "evaluate":
        evaluate_model(args.model, args.input, args.dataset)

    

