import numpy as np
from scipy.stats import vonmises, norm
from utils.constants import COUNT_MARKER, COUNT_VECTOR

import numpy as np
from scipy.stats import norm
from utils.constants import COUNT_MARKER


class ScalarSensor:
    """
    Sensore per il conteggio (Ni).
    Il rumore dipende dinamicamente dall'allineamento (Occlusione).
    """

    def __init__(self, num_noise_bins=21):
        self.num_classes = COUNT_MARKER
        self.all_classes = np.arange(self.num_classes, dtype=np.float32)

        self.min_std = 0.1
        self.max_std = 5.0
        self.num_noise_bins = num_noise_bins

        # table[true_val, noise_bin, class]
        self.table = np.zeros(
            (self.num_classes, self.num_noise_bins, self.num_classes),
            dtype=np.float32
        )

        for true_val in range(self.num_classes):
            for noise_bin in range(self.num_noise_bins):
                intensity = noise_bin / (self.num_noise_bins - 1)
                current_std = self.min_std + intensity * (self.max_std - self.min_std)

                pdf_vals = norm.pdf(self.all_classes, loc=true_val, scale=current_std)
                pdf_vals /= (pdf_vals.sum() + 1e-9)

                self.table[true_val, noise_bin] = pdf_vals.astype(np.float32)

    def _quantize_noise(self, noise_intensity):
        intensity = float(np.clip(noise_intensity, 0.0, 1.0))
        return int(round(intensity * (self.num_noise_bins - 1)))

    def observe(self, true_val, noise_intensity):
        true_val = int(true_val)
        noise_bin = self._quantize_noise(noise_intensity)
        return self.table[true_val, noise_bin]


class VectorSensor:
    """
    Sensore per l'orientamento (Alpha).
    Anche qui il rumore può dipendere da fattori esterni.
    """
    def __init__(self):
        self.num_classes = COUNT_VECTOR
        
        # Kappa (Concentrazione Von Mises):
        # 15.0 = Molto preciso (evitiamo 50 per aliasing)
        # 0.0 = Uniforme (non so nulla)
        self.max_kappa = 15.0
        
        # Setup angoli discreti
        step = 2 * np.pi / self.num_classes
        self.class_angles = np.linspace(-np.pi, np.pi - step, self.num_classes)

    def observe(self, true_angle_rad, noise_intensity):
        """
        Ritorna P(classe | true_angle).
        
        Args:
            true_angle_rad (float): Angolo vero in radianti.
            noise_intensity (float): 0.0 (Preciso) -> 1.0 (Rumore Totale).
        """
        intensity = np.clip(noise_intensity, 0.0, 1.0)
        
        # Mappiamo intensity in Kappa (Inverso)
        # Intensity 0 -> Kappa Max
        # Intensity 1 -> Kappa 0
        current_kappa = self.max_kappa * (1.0 - intensity)
        
        if current_kappa < 1e-3:
            # Caso limite: Uniforme
            return np.ones(self.num_classes, dtype=np.float32) / self.num_classes
        
        # Von Mises PDF
        pdf_vals = vonmises.pdf(self.class_angles, current_kappa, loc=true_angle_rad)
        pdf_vals /= (pdf_vals.sum() + 1e-9)
        
        return pdf_vals.astype(np.float32)