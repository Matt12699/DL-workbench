import numpy as np
import matplotlib.pyplot as plt
import os

class BoundsComparisonPlotter:
    """
    Classe per creare grafici di confronto tra approcci semi-supervisionati
    con upper bound (supervised) e lower bound (unsupervised).
    """
    
    def __init__(self, output_dir='NF-UNSW-NB15-v3/Architectures_Trends', figsize=(12, 8)):
        self.output_dir = output_dir
        self.figsize = figsize
        os.makedirs(output_dir, exist_ok=True)
        
        # Colori professionali per tesi
        self.colors = {
            'supervised': '#2E4A62',           # Blu navy - Upper bound
            'unsupervised': '#8B0000',         # Rosso scuro - Lower bound
            'semi_autoencoder': '#2F4F2F',     # Verde foresta - Semi-sup AutoEncoder
            'semi_contrastive': '#4A0E4E'      # Viola scuro - Semi-sup Contrastive
        }
        
        # Stili di linea per differenziare bounds da approcci
        self.line_styles = {
            'supervised': '--',        # Linea tratteggiata per upper bound
            'unsupervised': '--',      # Linea tratteggiata per lower bound
            'semi_autoencoder': '-',   # Linea continua per approccio
            'semi_contrastive': '-'    # Linea continua per approccio
        }
        
        # Markers diversi
        self.markers = {
            'supervised': 's',         # Quadrato per supervised
            'unsupervised': 'v',       # Triangolo giù per unsupervised
            'semi_autoencoder': 'o',   # Cerchio per autoencoder
            'semi_contrastive': '^'    # Triangolo su per contrastive
        }
    
    def plot_f1_bounds_comparison(self, labeled_samples, supervised_f1, unsupervised_f1, 
                                  semi_autoencoder_f1, semi_contrastive_f1,
                                  custom_title=None, save=True, show=True):
        """
        Crea un grafico di confronto F1 con upper/lower bounds.
        
        Args:
            labeled_samples: array del numero di campioni etichettati
            supervised_f1: valore F1 costante per supervised (upper bound)
            unsupervised_f1: valore F1 costante per unsupervised (lower bound)
            semi_autoencoder_f1: array valori F1 per semi-supervised autoencoder
            semi_contrastive_f1: array valori F1 per semi-supervised contrastive
            custom_title: titolo personalizzato
            save: se salvare il grafico
            show: se mostrare il grafico
        """
        
        labeled_samples = np.array(labeled_samples)
        semi_autoencoder_f1 = np.array(semi_autoencoder_f1)
        semi_contrastive_f1 = np.array(semi_contrastive_f1)
        
        plt.figure(figsize=self.figsize)
        
        # Upper bound (Supervised) - linea orizzontale costante
        plt.axhline(y=supervised_f1, color=self.colors['supervised'], 
                   linestyle=self.line_styles['supervised'], linewidth=8,
                   label='Supervised Learning (Upper Bound)', alpha=0.8)
        
        # Lower bound (Unsupervised) - linea orizzontale costante
        plt.axhline(y=unsupervised_f1, color=self.colors['unsupervised'],
                   linestyle=self.line_styles['unsupervised'], linewidth=8,
                   label='Unsupervised Learning (Lower Bound)', alpha=0.8)
        
        # Semi-supervised AutoEncoder Based
        plt.plot(labeled_samples, semi_autoencoder_f1, 
                marker=self.markers['semi_autoencoder'], linewidth=8, markersize=14,
                color=self.colors['semi_autoencoder'], markerfacecolor='white',
                markeredgewidth=3, markeredgecolor=self.colors['semi_autoencoder'],
                linestyle=self.line_styles['semi_autoencoder'],
                label='Semi-Supervised AutoEncoder Based')
        
        # Semi-supervised Contrastive
        plt.plot(labeled_samples, semi_contrastive_f1,
                marker=self.markers['semi_contrastive'], linewidth=8, markersize=14,
                color=self.colors['semi_contrastive'], markerfacecolor='white', 
                markeredgewidth=3, markeredgecolor=self.colors['semi_contrastive'],
                linestyle=self.line_styles['semi_contrastive'],
                label='Semi-Supervised Contrastive')
        
        # Impostazioni assi
        plt.xscale('log')
        # plt.gca().invert_xaxis()  # Rimuovo l'inversione per avere ordine crescente
        plt.xlabel('Number of Labeled Samples', fontsize=30, fontweight='bold')
        plt.ylabel('F1 Score', fontsize=30, fontweight='bold')
        
        # Titolo
        title = custom_title or 'F1 Score Comparison'
        plt.title(title, fontsize=32, fontweight='bold', pad=40)
        
        # Calcola limiti Y intelligenti con più margine sopra
        all_values = np.concatenate([semi_autoencoder_f1, semi_contrastive_f1, 
                                   [supervised_f1, unsupervised_f1]])
        padding = 0.05
        ymin = max(0, all_values.min() - padding)
        ymax = min(1.08, all_values.max() + 0.04)
        plt.ylim(ymin, ymax)
        
        # Limiti asse X (ora non più invertito)
        plt.xlim(labeled_samples.min() * 0.8, labeled_samples.max() * 1.2)
        
        # Griglia
        plt.grid(True, which="major", ls="-", alpha=0.4, linewidth=1.2)
        plt.grid(True, which="minor", ls="--", alpha=0.3, linewidth=0.8)
        
        # Legenda spostata in basso a destra
        plt.legend(fontsize=22, frameon=True, fancybox=True, shadow=True, 
                  loc='lower right')
        
        # Tick labels
        plt.xticks(fontsize=24, fontweight='bold')
        plt.yticks(fontsize=24, fontweight='bold')
        
        plt.tight_layout()
        
        # Salvataggio
        if save:
            filename = 'f1_bounds_comparison.png'
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            print(f"Grafico bounds comparison salvato in: {filepath}")
        
        if show:
            plt.show()
        else:
            plt.close()


# Esempio di utilizzo
def example_bounds_comparison():
    plotter = BoundsComparisonPlotter()
    
    # Dati di esempio - SOSTITUISCI CON I TUOI VALORI REALI
    labeled_samples = [2000000, 1000000, 2000, 200]
    
    supervised_f1_bound = 0.9991        # Upper bound - il migliore possibile
    unsupervised_f1_bound = 0.9262      # Lower bound - il peggiore ragionevole
    
    # Approcci semi-supervisionati (curve che variano)
    semi_autoencoder_f1 = [0.9400, 0.9105, 0.6490, 0.4660]    
    semi_contrastive_f1 = [0.9994, 0.9988, 0.9860, 0.9679]    
    
    # Crea il grafico
    plotter.plot_f1_bounds_comparison(
        labeled_samples=labeled_samples,
        supervised_f1=supervised_f1_bound,
        unsupervised_f1=unsupervised_f1_bound,
        semi_autoencoder_f1=semi_autoencoder_f1,
        semi_contrastive_f1=semi_contrastive_f1
    )


example_bounds_comparison()