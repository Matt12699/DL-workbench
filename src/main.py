import sys
import logging

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from utilities.logging_config import setup_logging
from utilities.argument_parser import ArgumentParser
from utilities.config_manager import ConfigManager
from data.log_min_max_scaler import LogMinMaxScaler


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
    # Log-Min-Max Scaler: Faccio il logaritmo delle feature numeriche e 
    # poi riscalo tutte le feature numeriche in modo tale che si trovino tutte in un range che va da -1 a 1
    num_pipeline = Pipeline([
        ("Scaling", LogMinMaxScaler())
    ])

    # Pipeline per le feature categoriche
    # cat_pipeline = Pipeline([

    # ])

    logger.info("Column Transformer...")

    # ColumnTransformer prende tutte le colonne e fa le trasformazioni giuste su feature numeriche e categoriche
    preProcessing = ColumnTransformer([
        ("num", num_pipeline, numeric_columns),
       # ("cat", cat_pipeline, categorical_columns),
    ])

    # Dataset processato
    df_prepared = preProcessing.fit_transform(df)

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
