import sys
import logging
import os

import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import f1_score, confusion_matrix, recall_score, precision_score, precision_recall_curve, auc
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
from data.arrayToDataFrame import ArrayToDataFrame
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
    df = pd.read_csv(input_path) 
    
    # Pipeline per le feature numeriche
    # CustomImputer mi permette di gestire i valori NaN e infiniti
    # Log-Min-Max Scaler: Faccio il logaritmo delle feature numeriche e 
    # poi riscalo tutte le feature numeriche in modo tale che si trovino tutte in un range che va da -1 a 1
    num_pipeline = Pipeline([
        ("Impute", CustomImputer()),
        ("Scaling", LogMinMaxScaler()),
        ("ToDataFrame", ArrayToDataFrame()),  # ← step intermedio
        ("Binning", CustomBinner(method="Uniform"))
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

def train_model(processed_csv_path: str, dataset_name: str, positive_label_value: int, plots_dir_train: str , early_stopping_metric: str, model_config: str):

    logging.info("Training model...")

    if positive_label_value == None:
        positive_label_value=1

    if plots_dir_train == None:
        plots_dir_train = "plots\Training_progress"

    if early_stopping_metric == None:
        early_stopping_metric = 'val_loss'

    if model_config == None:
        model_config = 'small'

    # Creo la directory per i plot se non esiste
    if not os.path.exists(plots_dir_train):
        os.makedirs(plots_dir_train)
        logging.info(f"Created directory for plots: {plots_dir_train}")
    

    # Carico la configurazione del dataset dal JSON
    config_path= "config/dataset.json"
    config_manager = ConfigManager()
    config_manager.load_config(config_path)

    target_column = config_manager.get_value(dataset_name, "target_column")

    full_dataset = CSVTabularDataset(processed_csv_path, target_column=target_column)

    # Ottengo il numero di feature dal dataset 
    num_features = full_dataset.X.shape[1]

    # Carico la configurazione degli iperparametri dal JSON
    config_path= "config/hyperparameters.json"
    config_manager.load_config(config_path)
    encoder_config = config_manager.get_value(model_config, "encoder")
    decoder_config = config_manager.get_value(model_config, "decoder")
    classifier_config = config_manager.get_value(model_config, "classifier")

    learningRate = config_manager.get_value(model_config, "lr")

    encoderDropout = encoder_config["dropout"]
    encoderHidden_layers = encoder_config["hidden_layers"]

    decoderDropout = decoder_config["dropout"]
    decoderHidden_layers = decoder_config["hidden_layers"]

    classifierDropout = classifier_config["dropout"]
    classifierHidden_layers = classifier_config["hidden_layers"]

    # Stabilisco input e output
    
    encoderInput_dim = num_features
    encoderOutput_dim = encoder_config["output_dim"]

    decoderInput_dim = encoderOutput_dim
    decoderOutput_dim = num_features

    classifierInput_dim = encoderOutput_dim
    classifierOutput_dim = classifier_config["output_dim"]

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
    logging.info(f"Using device: {device}")

    # Creo i modelli
    encoder = IDSModel(dropout=encoderDropout, hidden_layers=encoderHidden_layers, input_dim=encoderInput_dim, output_dim=encoderOutput_dim).to(device)
    decoder = IDSModel(dropout=decoderDropout, hidden_layers=decoderHidden_layers, input_dim=decoderInput_dim ,output_dim=decoderOutput_dim).to(device)
    classifier = IDSModel(dropout=classifierDropout, hidden_layers=classifierHidden_layers, input_dim=classifierInput_dim ,output_dim=classifierOutput_dim).to(device)

    # Loss per capire la ricostruzione dell'input
    criterion_autoEncoder = nn.MSELoss()
    
    # Definisco il training Loop
    # Numero di epoche
    N_EPOCHS = 100
 
    history = {
        'epochAutoEncoder': [],
        'epochClassifier': [],
        'autoEncoder_train_loss':[],
        'classifier_train_loss':[],
        'autoEncoder_val_loss': [],
        'classifier_val_loss': [],
        'val_f1': [],
        'val_precision': [],
        'val_recall': [],
        'val_pr_auc':[]
    }  
    

    # Parametri per Early Stopping
    early_stopping_patience = 10
    early_stopping_min_delta = 0.0001 # Miglioramento minimo per considerarlo tale

    # --- Inizializzazione Variabili per Early Stopping e Model Checkpointing ---
    AutoEncoder_best_metric_val = float('inf') 
    classifier_best_metric_val = -float('inf') if early_stopping_metric != 'val_loss' else float('inf')
    epochs_no_improve = 0

    Encoder_save_path = r"src\model\trained_encoder.pth"
    Classifier_save_path = r"src\model\trained_model.pth"

    classifier_save_dir = os.path.dirname(Classifier_save_path)
    if not os.path.exists(classifier_save_dir):
        os.makedirs(classifier_save_dir)
        logging.info(f"Created directory for model: {Classifier_save_path}")

    encoder_save_dir = os.path.dirname(Encoder_save_path)
    if not os.path.exists(encoder_save_dir):
        os.makedirs(encoder_save_dir)
        logging.info(f"Created directory for encoder: {Encoder_save_path}")
    
    logging.info(f"AutoEncoder Early stopping enabled: monitoring 'val_loss', patience={early_stopping_patience}, min_delta={early_stopping_min_delta}")
    logging.info(f"Best model will be saved to: {Encoder_save_path}")

    # Definisco l'algoritmo di ottimizzazione per l'Auto-Encoder
    params_AutoEncoder = list(encoder.parameters()) + list(decoder.parameters()) 

    optimizer = optim.Adam(params_AutoEncoder,
                          lr=learningRate)
    
    logging.info("Auto-Encoder training is starting")

    # Training Auto-Encoder
    for epoch in range(N_EPOCHS):

        # Per ogni epoca capiamo quanta è la loss
        total_autoEncoder_train_loss= 0.0
        encoder.train()
        decoder.train()
        for inputs, _ in train_dataLoader:

            # Sposta input alla GPU se disponibile
            inputs = inputs.to(device)

            # Azzero i gradienti accumulati dai passsaggi precedenti
            # Serve a far si che l'ottimizzatore aggiorni i parametri del modello correttamente
            optimizer.zero_grad()

            # -- Auto-Encoder --
            encoded_representation = encoder(inputs)
            reconstructed_output = decoder(encoded_representation)

            # Calcolo della loss
            loss_autoEncoder = criterion_autoEncoder(reconstructed_output, inputs)

            # Backpropagation: calcolo i gradienti
            loss_autoEncoder.backward()

            # Aggiusto i parametri basati sui gradienti
            optimizer.step()

            total_autoEncoder_train_loss += loss_autoEncoder.item()

        # Validation Auto-Encoder
        total_autoEncoder_val_loss = 0.0
        encoder.eval()
        decoder.eval()

        for inputs, _ in val_dataLoader:

            inputs = inputs.to(device)

            # -- Auto-Encoder --
            encoded_representation = encoder(inputs)
            reconstructed_output = decoder(encoded_representation)

            # Calcolo delle loss
            loss_autoEncoder = criterion_autoEncoder(reconstructed_output, inputs)

            total_autoEncoder_val_loss += loss_autoEncoder.item()

        # Calcolo le metriche dell'Auto-Encoder
        avg_autoEncoder_train_loss = total_autoEncoder_train_loss/len(train_dataLoader)
        avg_autoEncoder_val_loss = total_autoEncoder_val_loss/len(val_dataLoader)

        print("\n====================")
        print(f"Epoch: {epoch+1}/{N_EPOCHS}")
        print(f"Auto-Encoder Training Loss: {avg_autoEncoder_train_loss:.4f}")
        print(f"Auto-Encoder Validation Loss: {avg_autoEncoder_val_loss:.4f} ")
        print("====================\n")

        # Popolo il dizionario history per quanto riguarda l'autoEncoder
        history['epochAutoEncoder'].append(epoch + 1)
        history['autoEncoder_train_loss'].append(total_autoEncoder_train_loss/len(train_dataLoader))
        history['autoEncoder_val_loss'].append(total_autoEncoder_val_loss/len(val_dataLoader))

        # --- Logica di Early Stopping e Model Checkpointing ---
        current_metric_to_check = avg_autoEncoder_val_loss
        improved = (AutoEncoder_best_metric_val - current_metric_to_check) > early_stopping_min_delta


        if improved:
            AutoEncoder_best_metric_val = current_metric_to_check
            epochs_no_improve = 0
            model_checkpoint = { 
                'dropout_rate': encoderDropout, 
                'hidden_layers_config': encoderHidden_layers,
                'output_dim': encoderOutput_dim, 
                'input_dim': encoderInput_dim,
                'model_state_dict': encoder.state_dict()
            }
            torch.save(model_checkpoint, Encoder_save_path)
            logging.info(f"Epoch {epoch+1}: val_loss improved to {AutoEncoder_best_metric_val:.4f}. Model saved to {Encoder_save_path}")
        else:
            epochs_no_improve += 1
            logging.info(f"Epoch {epoch+1}: val_loss did not improve from {AutoEncoder_best_metric_val:.4f}. Patience: {epochs_no_improve}/{early_stopping_patience}")

        if epochs_no_improve >= early_stopping_patience:
            logging.info(f"Early stopping triggered after {epoch+1} epochs. Best {early_stopping_metric}: {AutoEncoder_best_metric_val:.4f}")
            break # Esce dal loop delle epoche
    
    # Carico da file l'encoder migliore
    try:
        checkpoint = torch.load(Encoder_save_path, map_location=device)

        # Estraggo i parametri di configurazione dal checkpoint
        dropout_rate_loaded = checkpoint['dropout_rate']
        hidden_layers_config_loaded = checkpoint['hidden_layers_config']
        input_dim_loaded = checkpoint['input_dim']
        output_dim_loaded = checkpoint['output_dim']

        # Creo l'istanza dell'encoder con i parametri caricati
        encoder = IDSModel(dropout=dropout_rate_loaded, hidden_layers=hidden_layers_config_loaded, input_dim=input_dim_loaded, output_dim=output_dim_loaded).to(device) 

        # Carica i pesi
        encoder.load_state_dict(checkpoint['model_state_dict'])
        logging.info(f"Encoder loaded successfully from checkpoint: {Encoder_save_path}")

    except FileNotFoundError:
        logging.warning(f"Error: Encoder file not found in: {Encoder_save_path}")
        exit()
    except RuntimeError as e:
        logging.warning(f"Error during the encoder loading: {e}")
        exit()

    # --- Calcolo pos_weight per BCEWithLogitsLoss ---
    # Raccolgo tutte le etichette dal DataLoader di training

    all_train_labels = []
    logging.info("Collecting labels from train_dataLoader...")
    for _, labels_batch in train_dataLoader:
        all_train_labels.append(labels_batch) # Aggiungi il tensore del batch alla lista

    # Concateno tutti i tensori dei batch in un unico tensore
    labels_tensor = torch.cat(all_train_labels).long() 

    # Calcolo i pesi

    num_total_train = len(labels_tensor)

    num_positives_train = torch.sum(labels_tensor == positive_label_value).item()
    num_negatives_train = num_total_train - num_positives_train

    # Calcolo il peso solo se entrambe le classi sono presenti
    if num_positives_train > 0 and num_negatives_train > 0 :
        # Calcola il peso per la classe positiva
        pos_weight_value = num_negatives_train / num_positives_train
        logging.info(f"pos_weight for BCEWithLogitsLoss: {pos_weight_value:.4f}")
        pos_weight_tensor = torch.tensor([pos_weight_value], device=device) # Sposta sul device corretto
        criterion_classification = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    else:
        logging.warning("Could not calculate pos_weight (only one class present). Using unweighted BCEWithLogitsLoss.")
        criterion_classification = nn.BCEWithLogitsLoss()

    # --- Fine Calcolo pos_weight ---
    
    logging.info(f"Classifier Early stopping enabled: monitoring '{early_stopping_metric}', patience={early_stopping_patience}, min_delta={early_stopping_min_delta}")
    logging.info(f"Best model will be saved to: {Classifier_save_path}")

    # Definisco l'algoritmo di ottimizzazione per il classificatore

    optimizer = optim.Adam(classifier.parameters(),
                          lr=learningRate)
    
    # Riporto il parametro che indica le epoche senza miglioramenti a 0
    epochs_no_improve = 0

    logging.info("Classification training is starting")
    # Training Encoder+Classificatore
    for epoch in range(N_EPOCHS):

        # Per ogni epoca capiamo quanta è la loss
        total_classifier_train_loss= 0.0
        encoder.train()
        classifier.train()

        for inputs, labels in train_dataLoader:

            # Sposta input alla GPU se disponibile
            inputs = inputs.to(device)

            # Aggiunge una dimensione e converte a float
            labels = labels.float().unsqueeze(1).to(device) 

            # Azzero i gradienti accumulati dai passsaggi precedenti
            # Serve a far si che l'ottimizzatore aggiorni i parametri del modello correttamente
            optimizer.zero_grad()

            # -- Encoder --
            encoded_representation = encoder(inputs)

            # Passo l'output dell'encoder al modello
            outputs = classifier(reconstructed_output)

            # Calcolo della loss
            loss_classification=criterion_classification(outputs, labels)

            # Backpropagation: calcolo i gradienti
            loss_classification.backward()

            # Aggiusto i parametri basati sui gradienti
            optimizer.step()

            total_classifier_train_loss += loss_classification.item()

        # Inizializzo le liste che conterranno le etichette predette e quelle vere
        all_predictions = []
        all_true_labels = []

        # Validation
        total_classifier_val_loss = 0.0
        encoder.eval()
        classifier.eval()

        for inputs, labels in val_dataLoader:

            inputs = inputs.to(device)
            # Aggiunge una dimensione e converte a float
            labels = labels.float().unsqueeze(1).to(device) 

            # -- Auto-Encoder --
            encoded_representation = encoder(inputs)

            outputs = classifier(encoded_representation)

            # Calcolo delle loss
            loss_classification = criterion_classification(outputs, labels)

            total_classifier_val_loss += loss_classification.item()

            # Applico la sigmoide per convertire in probabilità
            probs_val = torch.sigmoid(outputs).cpu()

            # Applico una soglia (0.5) alle probabilità per ottenere la classe predetta
            predicted = (probs_val > 0.5).float().squeeze()

            # Aggiungo le predizioni del batch (convertite in NumPy e spostate su CPU)
            all_predictions.extend(predicted.cpu().numpy())
            # Aggiungo le etichette vere del batch (convertite in NumPy, già su CPU)
            all_true_labels.extend(labels.cpu().numpy())
        
        y_true_np = np.array(all_true_labels)
        y_pred_np = np.array(all_predictions)

        # Calcolo le metriche
        avg_classifier_train_loss = total_classifier_train_loss/len(train_dataLoader)
        avg_classifier_val_loss = total_classifier_val_loss/len(val_dataLoader)
        f1 = f1_score(y_true_np, y_pred_np, pos_label=positive_label_value, average='binary', zero_division=0)
        precision = precision_score(y_true_np, y_pred_np, pos_label=positive_label_value, average='binary', zero_division=0)
        recall = recall_score(y_true_np, y_pred_np, pos_label=positive_label_value, average='binary', zero_division=0)
        precision_curve, recall_curve, _ = precision_recall_curve(y_true_np, y_pred_np, pos_label=positive_label_value)
        pr_auc = auc(recall_curve, precision_curve) # Area sotto la curva PR

        print("\n====================")
        print(f"Epoch: {epoch+1}/{N_EPOCHS}")
        print(f"Classifier Training Loss: {avg_classifier_train_loss:.4f}")
        print(f"Classifier Validation Loss: {avg_classifier_val_loss:.4f} ")
        print(f"F1-score: {f1}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"PR AUC: {pr_auc}")
        print("====================\n")

        # Popolo il dizionario history
        history['epochClassifier'].append(epoch + 1)
        history['classifier_train_loss'].append(total_classifier_train_loss/len(train_dataLoader))
        history['classifier_val_loss'].append(total_classifier_train_loss/len(train_dataLoader)) 
        history['val_f1'].append(f1); 
        history['val_precision'].append(precision)
        history['val_recall'].append(recall); 
        history['val_pr_auc'].append(pr_auc)

        # --- Logica di Early Stopping e Model Checkpointing ---
        # La differenza è principalmente tra la loss che deve diminuire e le altre metriche
        current_metric_to_check = 0.0
        if early_stopping_metric == 'val_loss':
            current_metric_to_check = avg_classifier_val_loss
            improved = (classifier_best_metric_val - current_metric_to_check) > early_stopping_min_delta
        elif early_stopping_metric == 'val_f1':
            current_metric_to_check = f1
            improved = (current_metric_to_check - classifier_best_metric_val) > early_stopping_min_delta
        elif early_stopping_metric == 'val_pr_auc':
            current_metric_to_check = pr_auc
            improved = (current_metric_to_check - classifier_best_metric_val) > early_stopping_min_delta
        elif early_stopping_metric == 'val_precision':
            current_metric_to_check = precision
            improved = (current_metric_to_check - classifier_best_metric_val) > early_stopping_min_delta
        elif early_stopping_metric == 'val_recall':
            current_metric_to_check = recall
            improved = (current_metric_to_check - classifier_best_metric_val) > early_stopping_min_delta
        else: # Default a val_loss se la metrica non è riconosciuta
            logging.warning(f"Unknown early_stopping_metric: {early_stopping_metric}. Defaulting to val_loss.")
            current_metric_to_check = avg_classifier_val_loss
            improved = (classifier_best_metric_val - current_metric_to_check) > early_stopping_min_delta
            early_stopping_metric = 'classifier_val_loss' # Aggiorna per coerenza nel logging


        if improved:
            classifier_best_metric_val = current_metric_to_check
            epochs_no_improve = 0
            model_checkpoint = { 
                'dropout_rate': classifierDropout, 
                'hidden_layers_config': classifierHidden_layers,
                'output_dim': classifierOutput_dim, 
                'input_dim': classifierInput_dim,
                'model_state_dict': classifier.state_dict()
            }
            torch.save(model_checkpoint, Classifier_save_path)
            logging.info(f"Epoch {epoch+1}: {early_stopping_metric} improved to {classifier_best_metric_val:.4f}. Model saved to {Classifier_save_path}")
        else:
            epochs_no_improve += 1
            logging.info(f"Epoch {epoch+1}: {early_stopping_metric} did not improve from {classifier_best_metric_val:.4f}. Patience: {epochs_no_improve}/{early_stopping_patience}")

        if epochs_no_improve >= early_stopping_patience:
            logging.info(f"Early stopping triggered after {epoch+1} epochs. Best {early_stopping_metric}: {classifier_best_metric_val:.4f}")
            break # Esce dal loop delle epoche

    # --- SEZIONE GRAFICI ---
    # 1. Grafico Training Loss vs Validation Loss (Auto-Encoder)
    plt.figure(figsize=(10, 6))
    plt.plot(history['epochAutoEncoder'], history['autoEncoder_train_loss'], label='Auto-Encoder Training Loss', marker='o')
    plt.plot(history['epochAutoEncoder'], history['autoEncoder_val_loss'], label='Auto-Encoder Validation Loss', marker='o')
    plt.title(f'Auto-Encoder Training & Validation Loss Over Epochs ({dataset_name})')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.grid(True)
    plt.savefig(os.path.join(plots_dir_train, f'AutoEncoder_loss_curve_{dataset_name}.png'))
    plt.close()
    logging.info(f"Loss curve plot saved to {os.path.join(plots_dir_train, f'AutoEncoder_loss_curve_{dataset_name}.png')}")

    # 1. Grafico Training Loss vs Validation Loss (Classificatore)
    plt.figure(figsize=(10, 6))
    plt.plot(history['epochClassifier'], history['classifier_train_loss'], label='Classifier Training Loss', marker='o')
    plt.plot(history['epochClassifier'], history['classifier_val_loss'], label='Classifier Validation Loss', marker='o')
    plt.title(f'Classifier Training & Validation Loss Over Epochs ({dataset_name})')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.grid(True)
    plt.savefig(os.path.join(plots_dir_train, f'Classifier_loss_curve_{dataset_name}.png'))
    plt.close()
    logging.info(f"Loss curve plot saved to {os.path.join(plots_dir_train, f'Classifier_loss_curve_{dataset_name}.png')}")

    # 2. Grafico Metriche di Validazione (F1, PR AUC, Precision, Recall)
    plt.figure(figsize=(12, 7))
    plt.plot(history['epochClassifier'], history['val_f1'], label='Validation F1-Score', marker='s')
    plt.plot(history['epochClassifier'], history['val_pr_auc'], label='Validation PR AUC', marker='^')
    plt.plot(history['epochClassifier'], history['val_precision'], label='Validation Precision', marker='.')
    plt.plot(history['epochClassifier'], history['val_recall'], label='Validation Recall', marker='.')
    plt.title(f'Validation Metrics Over Epochs ({dataset_name})')
    plt.xlabel('Epoch'); plt.ylabel('Score'); plt.legend(); plt.grid(True); plt.ylim(0,1.05)
    plt.savefig(os.path.join(plots_dir_train, f'validation_metrics_curve_{dataset_name}.png'))
    plt.close()
    logging.info(f"Validation metrics curve plot saved to {os.path.join(plots_dir_train, f'validation_metrics_curve_{dataset_name}.png')}")


    
def evaluate_model(model_path: str, processed_csv_path: str, dataset_name: str, positive_label_value: int, plots_dir: str):

    logging.info("Evaluating model...")

    if positive_label_value == None:
        positive_label_value=1

    if plots_dir == None:
        plots_dir = "plots\evaluation"

    # Creo la directory per i plot se non esiste
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
        logging.info(f"Created directory for plots: {plots_dir}")

    # Carico la configurazione dal JSON
    config_path= "config/dataset.json"
    config_manager = ConfigManager()
    config_manager.load_config(config_path)

    target_column = config_manager.get_value(dataset_name, "target_column")

    test_dataset = CSVTabularDataset(processed_csv_path, target_column=target_column)

    # Divido i campioni in batch
    test_dataLoader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Selezioniamo il dispositivo da usare: GPU, CPU...
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

    if model_path == None:
        logging.warning(f"Error: Model file cannot be None")
        exit()

    # Carico lo stato del modello
    try:
        checkpoint = torch.load(model_path, map_location=device)

        # Estraggo i parametri di configurazione dal checkpoint
        dropout_rate_loaded = checkpoint['dropout_rate']
        hidden_layers_config_loaded = checkpoint['hidden_layers_config']
        input_dim_loaded = checkpoint['input_dim']
        output_dim_loaded = checkpoint['output_dim']

        # Creo l'istanza del modello con i parametri caricati
        model = IDSModel(dropout=dropout_rate_loaded, hidden_layers=hidden_layers_config_loaded, input_dim=input_dim_loaded, output_dim=output_dim_loaded).to(device) 

        # Carica i pesi
        model.load_state_dict(checkpoint['model_state_dict'])
        logging.info(f"Model loaded successfully from checkpoint: {model_path}")

    except FileNotFoundError:
        logging.warning(f"Error: Model file not found in: {model_path}")
        exit()
    except RuntimeError as e:
        logging.warning(f"Error during the model loading: {e}")
        exit()

    # Inizializzo le liste che conterranno le etichette predette e quelle vere
    all_predictions = []
    all_true_labels = []

    for x_test_batch, y_test_batch in test_dataLoader:

        model.eval()
        y_test_batch = y_test_batch.float().unsqueeze(1).to(device) 
        x_test_batch = x_test_batch.to(device)

        # Output per ogni batch
        y_pred_batch = model(x_test_batch)

        # Applico la sigmoide per convertire in probabilità
        probs_val = torch.sigmoid(y_pred_batch).cpu()

        # Applico una soglia (0.5) alle probabilità per ottenere la classe predetta
        predicted = (probs_val > 0.5).float().squeeze()

        # Aggiungo le predizioni del batch (convertite in NumPy e spostate su CPU)
        all_predictions.extend(predicted.cpu().numpy())
        # Aggiungo le etichette vere del batch (convertite in NumPy, già su CPU)
        all_true_labels.extend(y_test_batch.cpu().numpy())

    # Converto le liste in array NumPy per sklearn.metrics
    y_true_np = np.array(all_true_labels)
    y_pred_np = np.array(all_predictions)

    # Calcolo le metriche
    f1 = f1_score(y_true_np, y_pred_np, pos_label=positive_label_value, average='binary', zero_division=0)
    precision = precision_score(y_true_np, y_pred_np, pos_label=positive_label_value, average='binary', zero_division=0)
    recall = recall_score(y_true_np, y_pred_np, pos_label=positive_label_value, average='binary', zero_division=0)

    # Serve ad assicurarci che la matrice di confusione abbia la dimensione giusta
    cm_labels = [0,1]

    # Per mostrare dopo il grafico
    cm_display_labels = ['Benign (0)', 'Malign (1)']


    # Calcolo della matrice di confusione
    # labels=[negative_label_value, positive_label_value] assicura che
    # la prima riga/colonna sia per la classe negativa, la seconda per la positiva.
    # Questo dà: [[TN, FP], [FN, TP]]
    cm = confusion_matrix(y_true_np, y_pred_np, labels=cm_labels)


    print(f"\n--- Evaluation Metrics (Positive Class: {positive_label_value}) ---")
    print(f"Test F1-Score: {f1:.4f}")
    print(f"Test Precision: {precision:.4f}")
    print(f"Test Recall (Sensitivity): {recall:.4f}")
    print("\n")
    print(f"Confusion Matrix (labels: {cm_labels}):")


    # Formato standard Confusion Matrix (con labels=[negativo, positivo]):
    # Riga 0: Veri Negativi (TN), Falsi Positivi (FP)
    # Riga 1: Falsi Negativi (FN), Veri Positivi (TP)
    #   Pred N  Pred P
    # [[TN,      FP],    <- True N
    #  [FN,      TP]]    <- True P
    print(cm)
    print("\n")
    if cm.shape == (2,2):
        # Appiattisco la matrice in un array
        tn, fp, fn, tp = cm.ravel()
        print(f"  True Negatives (TN): {tn}")
        print(f"  False Positives (FP): {fp}")
        print(f"  False Negatives (FN): {fn}")
        print(f"  True Positives (TP): {tp}")

    # --- SEZIONE GRAFICI ---
    # 1. GRAFICO: MATRICE DI CONFUSIONE
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=cm_display_labels, yticklabels=cm_display_labels)
    plt.title(f'Confusion Matrix - Test Set ({dataset_name})')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

    # Salvo il grafico
    plot_filename = f"confusion_matrix_{dataset_name}.png"
    plot_save_path = os.path.join(plots_dir, plot_filename)
    try:
        plt.savefig(plot_save_path)
        logging.info(f"Confusion matrix plot saved to {plot_save_path}")
    except Exception as e:
        logging.error(f"Failed to save confusion matrix plot: {e}")
    plt.close() # Chiude la figura per liberare memoria

    # 2. GRAFICO: BAR CHART DELLE METRICHE PRINCIPALI
    metrics_names = ['F1-Score', 'Precision', 'Recall']
    metrics_values = [f1, precision, recall]
    plt.figure(figsize=(10, 6))
    bars = plt.bar(metrics_names, metrics_values, color=['lightcoral', 'lightgreen', 'gold'])
    plt.title(f'Performance Metrics - Test Set ({dataset_name})\nPositive Class: {cm_display_labels[1]}')
    plt.ylabel('Score')
    plt.ylim(0, 1.1) # Le metriche sono tra 0 e 1 (metto 1.1 per evitare che le scritte si sovrappongano)
    for bar in bars: # Aggiunge il valore sopra ogni barra
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01, f'{yval:.4f}', ha='center', va='bottom')
    plot_metrics_filename = f"metrics_barchart_{dataset_name}.png"
    plot_metrics_save_path = os.path.join(plots_dir, plot_metrics_filename)
    try:
        plt.savefig(plot_metrics_save_path)
        logging.info(f"Metrics bar chart saved to {plot_metrics_save_path}")
    except Exception as e:
        logging.error(f"Failed to save metrics bar chart: {e}")
    plt.close()

    # 3. GRAFICO: PRECISION-RECALL CURVE
    # 'precision' e 'recall' qui sono array di valori per la curva, non la metrica singola
    precision_curve, recall_curve, _ = precision_recall_curve(y_true_np, y_pred_np, pos_label=positive_label_value)
    pr_auc = auc(recall_curve, precision_curve) # Area sotto la curva PR

    
    print(f"\nTest PR AUC (Area Under Precision-Recall Curve): {pr_auc:.4f}\n") # Stampa PR AUC

    plt.figure(figsize=(8, 6))
    plt.plot(recall_curve, precision_curve, marker='.', label=f'PR Curve (AUC = {pr_auc:.4f})')
    # Serve a rappresentare un classificatore senza abilità. La curva precision/recall dovrebbe trovarsi sempre al di sopra
    # Numero di campioni positivi reali / numero totale di campioni
    no_skill = len(y_true_np[y_true_np==positive_label_value]) / len(y_true_np)
    plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label=f'No Skill (Prevalence = {no_skill:.2f})')
    plt.title(f'Precision-Recall Curve - Test Set ({dataset_name})\nPositive Class: {cm_display_labels[1]}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    plt.grid(True)
    plot_pr_filename = f"precision_recall_curve_{dataset_name}.png"
    plot_pr_save_path = os.path.join(plots_dir, plot_pr_filename)
    try:
        plt.savefig(plot_pr_save_path)
        logging.info(f"Precision-Recall curve plot saved to {plot_pr_save_path}")
    except Exception as e:
        logging.error(f"Failed to save Precision-Recall curve plot: {e}")
    plt.close()

    logging.info(f"Evaluation complete. F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, PR AUC: {pr_auc:.4f}")
    
   

    


if __name__ == "__main__":
    parser = ArgumentParser("fl-ids")

    parser.register_subcommands(
        "prepare",
        ["--input", "--output", "--dataset"],
        ["The input path for the data.", "The output path for the prepared data.", "The name of the dataset"],
    )

    parser.register_subcommands(
        "train",
        ["--input", "--dataset", "--positiveLabel", "--plotsDir", "--earlyMetric", "--config"],
        ["The input path for the processed data.", "The name of the dataset", "The value of the positive label", "The path for the plots", "The early stopping metric", "The name of the model configuration"],
    )

    parser.register_subcommands(
        "evaluate",
        ["--model", "--input", "--dataset", "--positiveLabel", "--plotsDir"],
        ["The path for the processed model", "The input path for the processed data.", "The name of the dataset", "The value of the positive label", "The path for the plots"],
    )

    args = parser.parse_arguments(sys.argv[1:])

    if args.subcommand == "prepare":
        prepare_data(args.input, args.output, args.dataset)
    elif args.subcommand == "train":
        train_model(args.input, args.dataset, args.positiveLabel, args.plotsDir, args.earlyMetric, args.config)
    elif args.subcommand == "evaluate":
        evaluate_model(args.model, args.input, args.dataset, args.positiveLabel, args.plotsDir)

    

