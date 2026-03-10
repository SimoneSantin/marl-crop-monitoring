import numpy as np
import gstools as gs
from utils.constants import COUNT_MARKER

class FieldGenerator:
    def __init__(self, size=20, len_scale=5.0, var=1.0):
        """
        size: Dimensione della griglia (size x size)
        len_scale: "Correlation Length". Più è alto, più le zone sono ampie e omogenee.
        var: Varianza del campo gaussiano.
        """
        self.size = size
        self.x = np.arange(size)
        self.y = np.arange(size)
        
        # Modello di covarianza Gaussiano (garantisce transizioni lisce)
        self.model = gs.Gaussian(dim=2, var=var, len_scale=len_scale)

    def generate_field(self, seed=None):
        if seed is None:
            seed = np.random.randint(0, 100000)
            
        # Generiamo due componenti vettoriali indipendenti (U, V)
        # Seed diversi per U e V garantiscono che non siano identici
        srf_u = gs.SRF(self.model, seed=seed)
        field_u = srf_u.structured([self.x, self.y])
        
        srf_v = gs.SRF(self.model, seed=seed + 1)
        field_v = srf_v.structured([self.x, self.y])
        
        # --- 1. Calcolo COUNT (dal Modulo) ---
        # Il modulo segue una distribuzione di Rayleigh. 
        # Lo normalizziamo per stare nel range [0, COUNT_MARKER-1]
        magnitude = np.sqrt(field_u**2 + field_v**2)
        
        # Normalizzazione Min-Max locale per sfruttare tutto il range 0-9
        mag_norm = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min() + 1e-9)
        true_counts = np.floor(mag_norm * COUNT_MARKER).astype(int)
        true_counts = np.clip(true_counts, 0, COUNT_MARKER - 1)

        # --- 2. Calcolo ANGLE (dall'Orientamento) ---
        # Arctan2 restituisce valori in [-pi, pi]
        true_angles = np.arctan2(field_v, field_u) 

        return {
            'field_u': field_u,
            'field_v': field_v,
            'true_counts': true_counts, # Matrice (Size, Size) interi
            'true_angles': true_angles  # Matrice (Size, Size) float [-pi, pi]
        }