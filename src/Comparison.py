def plot_comparison(self, data_dict, metric_name, custom_title=None, save=True, show=True):
        """
        Crea un grafico di confronto tra diversi paradigmi per la stessa metrica.
        
        Args:
            data_dict: dizionario con formato {paradigm: {'percentages': [...], 'values': [...]}}
            metric_name: nome della metrica
            custom_title: titolo personalizzato
            save: se salvare il grafico
            show: se mostrare il grafico
        """
        
        config = self.metric_configs[metric_name]
        
        plt.figure(figsize=(12, 8))
        
        for paradigm, data in data_dict.items():
            percentages = np.array(data['percentages'])
            values = np.array(data['values'])
            
            color = self.colors.get(paradigm, '#1f77b4')
            label = paradigm.replace('_', ' ').title()
            
            plt.plot(percentages, values, marker="o", linewidth=4, markersize=12,
                    label=label, color=color, markerfacecolor='white',
                    markeredgewidth=3, markeredgecolor=color)
        
        # Impostazioni comuni con testo molto grande
        plt.xscale('log')
        plt.gca().invert_xaxis()
        plt.xlabel('Percentage of Labeled Data (%)', fontsize=20, fontweight='bold')
        plt.ylabel(config['ylabel'], fontsize=20, fontweight='bold')
        
        title = custom_title or f"Comparison of {config['ylabel']} across Paradigms"
        plt.title(title, fontsize=22, fontweight='bold', pad=25)
        
        # Tick labels più grandi
        plt.xticks(fontsize=16, fontweight='bold')
        plt.yticks(fontsize=16, fontweight='bold')
        
        plt.grid(True, which="major", ls="-", alpha=0.4, linewidth=1.2)
        plt.grid(True, which="minor", ls="--", alpha=0.3, linewidth=0.8)
        plt.legend(fontsize=16, frameon=True, fancybox=True, shadow=True)
        plt.tight_layout()
        
        if save:
            filename = f"{metric_name}_comparison.png"
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Grafico di confronto salvato in: {filepath}")
        
        if show:
            plt.show()
        else:
            plt.close()
import numpy as np
import matplotlib.pyplot as plt
import os

class MetricsPlotter:
    """
    Classe generalizzata per creare grafici di metriche ML in funzione 
    della percentuale di dati etichettati per diversi paradigmi di apprendimento.
    """
    
    def __init__(self, output_dir='NF-UNSW-NB15-v3/Architectures_Trends', figsize=(10, 6)):
        self.output_dir = output_dir
        self.figsize = figsize
        # Assicurati che la directory esista
        os.makedirs(output_dir, exist_ok=True)
        
        # Colori predefiniti per diversi paradigmi - palette più professionale per tesi
        self.colors = {
            'supervised': '#2E4A62',        # Blu navy professionale
            'unsupervised': '#8B0000',      # Rosso scuro
            'semi_supervised': '#2F4F2F',   # Verde scuro foresta  
            'self_supervised': '#4A0E4E',   # Viola scuro
            'transfer_learning': '#8B4513', # Marrone saddle
            'few_shot': '#1C5F2C'          # Verde pino scuro
        }
        
        # Colori specifici per metriche diverse nello stesso grafico
        self.metric_colors = {
            'f1_score': '#2E4A62',     # Blu navy
            'precision': '#8B0000',    # Rosso scuro  
            'recall': '#2F4F2F'        # Verde scuro
        }
        
        # Configurazioni specifiche per metriche
        self.metric_configs = {
            'f1_score': {
                'ylabel': 'F1 Score',
                'title_template': 'F1 Score across Different Amounts of Labeled Data',
                'filename_template': 'f1_score_{paradigm}.png',
                'ylim_padding': 0.005,
                'invert_better': False,  # Più alto è meglio
                'annotation_format': '{:.4f}'
            },
            'misclassified_samples': {
                'ylabel': 'Misclassified Samples (Count)',
                'title_template': 'Misclassified Samples across Different Amounts of Labeled Data', 
                'filename_template': 'misclassified_{paradigm}.png',
                'ylim_padding': 50,  # Padding per numeri assoluti
                'invert_better': True,  # Più basso è meglio
                'annotation_format': '{:.0f}'  # Numeri interi
            },
            'accuracy': {
                'ylabel': 'Accuracy',
                'title_template': 'Accuracy across Different Amounts of Labeled Data',
                'filename_template': 'accuracy_{paradigm}.png', 
                'ylim_padding': 0.005,
                'invert_better': False,
                'annotation_format': '{:.4f}'
            },
            'precision': {
                'ylabel': 'Precision',
                'title_template': 'Precision across Different Amounts of Labeled Data',
                'filename_template': 'precision_{paradigm}.png',
                'ylim_padding': 0.005,
                'invert_better': False,
                'annotation_format': '{:.4f}'
            },
            'recall': {
                'ylabel': 'Recall', 
                'title_template': 'Recall across Different Amounts of Labeled Data',
                'filename_template': 'recall_{paradigm}.png',
                'ylim_padding': 0.005,
                'invert_better': False,
                'annotation_format': '{:.4f}'
            }
        }
    
    def plot_metric(self, percentages, values, paradigm, metric_name, 
                   custom_title=None, custom_color=None, save=True, show=True):
        """
        Crea un grafico per una specifica metrica e paradigma.
        
        Args:
            percentages: array delle percentuali di dati etichettati
            values: array dei valori della metrica
            paradigm: nome del paradigma (es. 'supervised', 'unsupervised')
            metric_name: nome della metrica (es. 'f1_score', 'misclassified_samples')
            custom_title: titolo personalizzato (opzionale)
            custom_color: colore personalizzato (opzionale)
            save: se salvare il grafico
            show: se mostrare il grafico
        """
        
        # Verifica che la metrica sia supportata
        if metric_name not in self.metric_configs:
            raise ValueError(f"Metrica '{metric_name}' non supportata. "
                           f"Metriche disponibili: {list(self.metric_configs.keys())}")
        
        config = self.metric_configs[metric_name]
        
        # Converti in numpy arrays
        percentages = np.array(percentages)
        values = np.array(values)
        
        # Creazione del grafico
        plt.figure(figsize=self.figsize)
        
        # Determina colore e label
        color = custom_color or self.colors.get(paradigm, '#1f77b4')
        paradigm_label = paradigm.replace('_', ' ').title() + ' Learning Approach'
        
        # Plot principale con linee e markers più spessi
        plt.plot(percentages, values, marker="o", linewidth=4, markersize=12, 
                label=paradigm_label, color=color, markerfacecolor='white',
                markeredgewidth=3, markeredgecolor=color)
        
        # Impostazioni assi con testo molto grande e grassetto
        plt.xscale('log')
        plt.gca().invert_xaxis()
        plt.xlabel('Labeled Data', fontsize=20, fontweight='bold')
        plt.ylabel(config['ylabel'], fontsize=20, fontweight='bold')
        
        # Titolo molto grande
        title = custom_title or config['title_template']
        plt.title(title, fontsize=22, fontweight='bold', pad=40)
        
        # Zoom intelligente sull'asse Y
        padding = config['ylim_padding']
        ymin = values.min() - padding
        ymax = values.max() + padding
        plt.ylim(ymin, ymax)
        
        # Limiti asse X per migliore visualizzazione
        plt.xlim(percentages.max() * 1.2, percentages.min() * 0.8)
        
        # Griglia migliorata
        plt.grid(True, which="major", ls="-", alpha=0.4, linewidth=1.2)
        plt.grid(True, which="minor", ls="--", alpha=0.3, linewidth=0.8)
        
        # Legenda con styling e testo grande
        plt.legend(fontsize=16, frameon=True, fancybox=True, shadow=True)
        
        # Tick labels più grandi e grassetti
        plt.xticks(fontsize=30, fontweight='bold')
        plt.yticks(fontsize=30, fontweight='bold')
        
        # Annotazioni sui punti per migliore leggibilità con testo più grande
        annotation_format = config.get('annotation_format', '{:.2f}')
        for i, (x, y) in enumerate(zip(percentages, values)):
            plt.annotate(annotation_format.format(y), (x, y), 
                       textcoords="offset points", xytext=(0,15), ha='center', 
                       fontsize=30, fontweight='bold',
                       bbox=dict(boxstyle="round,pad=0.4", facecolor='white', 
                               alpha=0.9, edgecolor='gray', linewidth=1))
        
        plt.tight_layout()
        
        # Salvataggio
        if save:
            filename = config['filename_template'].format(paradigm=paradigm)
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            print(f"Grafico salvato in: {filepath}")
        
        # Visualizzazione
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_multiple_metrics(self, percentages, metrics_data, paradigm, 
                              custom_title=None, save=True, show=True):
        """
        Crea un grafico con multiple metriche per lo stesso paradigma.
        
        Args:
            percentages: array delle percentuali di dati etichettati
            metrics_data: dizionario con formato {'metric_name': values_array}
            paradigm: nome del paradigma
            custom_title: titolo personalizzato (opzionale)
            save: se salvare il grafico
            show: se mostrare il grafico
        """
        
        # Converti percentages in numpy array
        percentages = np.array(percentages)
        
        # Creazione del grafico
        plt.figure(figsize=self.figsize)
        
        # Plot per ogni metrica
        for metric_name, values in metrics_data.items():
            if metric_name not in self.metric_configs:
                print(f"Attenzione: Metrica '{metric_name}' non riconosciuta, salto...")
                continue
                
            values = np.array(values)
            color = self.metric_colors.get(metric_name, '#1f77b4')
            label = metric_name.replace('_', ' ').title()
            
            # Plot con stili diversi per ogni metrica
            if metric_name == 'f1_score':
                plt.plot(percentages, values, marker="o", linewidth=4, markersize=12,
                        label=label, color=color, markerfacecolor='white',
                        markeredgewidth=3, markeredgecolor=color)
            elif metric_name == 'precision':
                plt.plot(percentages, values, marker="s", linewidth=4, markersize=11,
                        label=label, color=color, markerfacecolor='white',
                        markeredgewidth=3, markeredgecolor=color)
            elif metric_name == 'recall':
                plt.plot(percentages, values, marker="^", linewidth=4, markersize=12,
                        label=label, color=color, markerfacecolor='white',
                        markeredgewidth=3, markeredgecolor=color)
        
        # Impostazioni assi con testo molto grande e grassetto
        plt.xscale('log')
        plt.gca().invert_xaxis()
        plt.xlabel('Labeled Data', fontsize=20, fontweight='bold')
        plt.ylabel('Score', fontsize=20, fontweight='bold')
        
        # Titolo
        paradigm_formatted = paradigm.replace('_', ' ').title()
        title = custom_title or f'{paradigm_formatted} Learning: F1-Score, Precision & Recall'
        plt.title(title, fontsize=22, fontweight='bold', pad=40)
        
        # Calcola limiti Y basati su tutti i valori
        all_values = np.concatenate([np.array(values) for values in metrics_data.values()])
        padding = 0.01
        ymin = all_values.min() - padding
        ymax = all_values.max() + padding
        plt.ylim(ymin, ymax)
        
        # Limiti asse X per migliore visualizzazione
        plt.xlim(percentages.max() * 1.2, percentages.min() * 0.8)
        
        # Griglia migliorata
        plt.grid(True, which="major", ls="-", alpha=0.4, linewidth=1.2)
        plt.grid(True, which="minor", ls="--", alpha=0.3, linewidth=0.8)
        
        # Legenda con styling e testo grande
        plt.legend(fontsize=16, frameon=True, fancybox=True, shadow=True, loc='lower left')
        
        # Tick labels più grandi e grassetti
        plt.xticks(fontsize=30, fontweight='bold')
        plt.yticks(fontsize=30, fontweight='bold')
        
        # Annotazioni sui punti per migliore leggibilità
        for metric_name, values in metrics_data.items():
            values = np.array(values)
            for i, (x, y) in enumerate(zip(percentages, values)):
                plt.annotate(f'{y:.4f}', (x, y), 
                           textcoords="offset points", xytext=(0,15), ha='center', 
                           fontsize=30, fontweight='bold',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor='white', 
                                   alpha=0.8, edgecolor='gray', linewidth=1))
        
        plt.tight_layout()
        
        # Salvataggio
        if save:
            filename = f"multiple_metrics_{paradigm}.png"
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            print(f"Grafico multi-metrica salvato in: {filepath}")
        
        # Visualizzazione
        if show:
            plt.show()
        else:
            plt.close()


# ===== ESEMPI DI UTILIZZO =====

def example_usage():
    # Inizializza il plotter
    plotter = MetricsPlotter()

    # I tuoi dati F1-Score esistenti
    percentages = [2000000, 1000000, 2000, 200]
    f1_values = [0.9991, 0.9984, 0.9831, 0.9391]

    # Aggiungi i tuoi dati reali per Precision e Recall
    precision_values = [0.9982, 0.9969, 0.9671, 0.8931]  
    recall_values = [0.9999, 0.9999, 0.9997, 0.9902]   

    # Dati di esempio per Misclassified Samples (numeri assoluti invece di percentuali)
    misclassified_values = [24, 41, 442, 1651]  

    # Grafico combinato con tutte e 3 le metriche
    metrics_data = {
        'f1_score': f1_values,
        'precision': precision_values, 
        'recall': recall_values
    }
    plotter.plot_multiple_metrics(percentages, metrics_data, 'Supervised')  
    
    # Plot Misclassified Samples
    plotter.plot_metric(percentages, misclassified_values, 'Supervised', 'misclassified_samples')
    
    # Esempio di confronto tra paradigmi (inserisci i tuoi dati)
    comparison_data = {
        'supervised': {
            'percentages': [100, 50, 0.1, 0.01],
            'values': [0.9991, 0.9984, 0.9831, 0.9391]
        },
        'semi_supervised': {
            'percentages': [100, 50, 0.1, 0.01], 
            'values': [0.9985, 0.9990, 0.9850, 0.9420]  # Valori di esempio
        }
    }
    
    # Plot di confronto
   # plotter.plot_comparison(comparison_data, 'f1_score')

# Esegui gli esempi (decommenta per testare)
example_usage()