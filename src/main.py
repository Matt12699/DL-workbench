import sys
import logging

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pandas.plotting import scatter_matrix
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

    # Salvo il dataset processato nel file CSV specificato
    if output_path!=None: 
        df_prepared_df.to_csv(output_path, index=False)


if __name__ == "__main__":
    parser = ArgumentParser("fl-ids")

    parser.register_subcommands(
        "prepare",
        ["--input", "--output", "--dataset"],
        ["The input path for the data.", "The output path for the prepared data.", "The name of the dataset"],
    )

    args = parser.parse_arguments(sys.argv[1:])

    if args.subcommand == "prepare":
        prepare_data(args.input, args.output, args.dataset)
