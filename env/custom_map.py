import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
import torch

from utils.constants import (
    COUNT_MARKER, 
    COUNT_VECTOR, 
    MAX_STEPS,
)
from env.field_generator import FieldGenerator
from env.sensor import ScalarSensor, VectorSensor

class CustomMapEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, 
                 field_size=20, 
                 max_steps=MAX_STEPS, 
                 device='cpu'):
        
        super(CustomMapEnv, self).__init__()
        
        self.field_size = field_size
        self.max_steps = max_steps
        self.device = device
        
        self.generator = FieldGenerator(size=self.field_size, len_scale=5.0, var=1.0)
        
        # Inizializziamo i sensori senza noise fisso
        self.scalar_sensor = ScalarSensor()
        self.vector_sensor = VectorSensor()
        
        self.action_space = spaces.Discrete(4) 

        self.obs_dim = 9 + COUNT_MARKER 
       
        self.observation_space = spaces.Box(low=-1, high=1, shape=(self.obs_dim,), dtype=np.float32)

        # Stato interno
        self.agent_pos = None 
        self.visited_mask = None
        self.current_step = 0
        self.grid_counts = None
        self.grid_angles = None
        
        # Variabile per tracciare l'ultimo movimento (Drone Vector)
        # Formato (dx, dy) ovvero (variazione righe, variazione colonne)
        self.last_move_vector = np.array([0.0, 0.0])
        
        self.fig = None
        self.ax = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Genera mappa
        field_data = self.generator.generate_field(seed=seed)
        self.grid_counts = field_data['true_counts']
        self.grid_angles = field_data['true_angles']
        
        self.agent_pos = [np.random.randint(0, self.field_size), np.random.randint(0, self.field_size)]
        self.visited_mask = np.zeros((self.field_size, self.field_size), dtype=bool)
        self.visited_mask[self.agent_pos[0], self.agent_pos[1]] = True
        self.current_step = 0
        
        # Reset del vettore movimento.
        # All'inizio il drone è fermo/appena atterrato.
        # Assumiamo noise massimo (1.0) finché non si muove, o un valore neutro.
        # Impostiamo [0,0] così il calcolo dell'allineamento darà 0 (Max Noise).
        # Questo sprona l'agente a muoversi subito.
        self.last_move_vector = np.array([0.0, 0.0])
        
        return self._get_obs(), {}

    def step(self, action):
        x, y = self.agent_pos
        
        # Calcolo nuova posizione e Vettore Movimento
        # Mapping Azioni: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT
        # Coordinate Griglia: x=Righe (0 in alto), y=Colonne (0 a sinistra)
        
        dx, dy = 0, 0
        
        if action == 0:   # UP (Riga diminuisce)
            x = max(0, x - 1)
            dx = -1
        elif action == 1: # DOWN (Riga aumenta)
            x = min(self.field_size - 1, x + 1)
            dx = 1
        elif action == 2: # LEFT (Colonna diminuisce)
            y = max(0, y - 1)
            dy = -1
        elif action == 3: # RIGHT (Colonna aumenta)
            y = min(self.field_size - 1, y + 1)
            dy = 1
        
        # Aggiorniamo il vettore movimento del drone
        # Se sbatte contro il muro (dx=0, dy=0), il noise salirà al massimo.
        self.last_move_vector = np.array([dx, dy])
        
        is_new_cell = not self.visited_mask[x, y]
        self.agent_pos = [x, y]
        self.visited_mask[x, y] = True
        self.current_step += 1
        
        # Ottieni osservazione (qui viene calcolato il noise dinamico)
        obs = self._get_obs()
        
        # Penalità Tempo
        reward = -0.05
                    
        truncated = False
        terminated = self.current_step >= self.max_steps
        
        return obs, reward, terminated, truncated, {}

    def _get_obs(self):
        """
        Calcola l'osservazione applicando il noise dinamico basato sull'allineamento.
        """
        x, y = self.agent_pos
        
        # --- 1. POV (Visione Locale 3x3) ---
        pov = np.zeros((3, 3), dtype=np.float32)
        k_r = 0
        for i in range(-1, 2):
            k_c = 0
            for j in range(-1, 2):
                nx, ny = x + i, y + j
                if 0 <= nx < self.field_size and 0 <= ny < self.field_size:
                    pov[k_r, k_c] = 1.0 if self.visited_mask[nx, ny] else 0.0
                else:
                    pov[k_r, k_c] = -1.0
                k_c += 1
            k_r += 1
        pov_flat = pov.flatten() 
        
        # --- 2. CALCOLO NOISE DINAMICO (ALLINEAMENTO) ---
        
        # Recuperiamo l'angolo vero della pianta (in radianti)
        alpha = self.grid_angles[x, y]
        
        # Convertiamo l'angolo pianta in vettore (u, v)
        # u = componente colonne (orizzontale), v = componente righe (verticale)
        # Nota: gstools usa arctan2(v, u). 
        # Assumiamo standard matematico: u=cos(alpha), v=sin(alpha)
        # Ma attenzione alle coordinate dell'array (row, col).
        # Mappiamo: U -> dy (colonne), V -> dx (righe).
        # Verifichiamo il sistema di riferimento:
        # Se alpha=0 (Est) -> cos=1, sin=0 -> dy=1, dx=0 (Movimento a Destra)
        # Se alpha=pi/2 (Sud/Giù) -> cos=0, sin=1 -> dy=0, dx=1 (Movimento Giù)
        # Questo mapping (u->dy, v->dx) è coerente con action 3 (dy=1) e action 1 (dx=1).
        
        plant_vec_dy = np.cos(alpha) # U (horizontal change)
        plant_vec_dx = np.sin(alpha) # V (vertical change)
        
        drone_dx = self.last_move_vector[0]
        drone_dy = self.last_move_vector[1]
        
        # Calcolo allineamento (Prodotto Scalare Normalizzato)
        # Dato che cos e sin sono già normalizzati e il drone muove di 1 o 0...
        # Normalizziamo il vettore drone se non è zero
        norm_drone = np.sqrt(drone_dx**2 + drone_dy**2)
        
        if norm_drone > 0:
            # Prodotto scalare
            dot_product = (drone_dx * plant_vec_dx) + (drone_dy * plant_vec_dy)
            alignment = abs(dot_product / norm_drone) 
            # alignment è [0, 1]. 1=Parallelo, 0=Perpendicolare.
        else:
            # Drone fermo (es. reset o muro) -> Noise Massimo
            alignment = 0.0
            
        # Calcolo Intensità Rumore (Inverso dell'allineamento)
        # Allineato (1.0) -> Noise 0.0
        # Perpendicolare (0.0) -> Noise 1.0
        noise_intensity = 1.0 - alignment

        true_val = self.grid_counts[x, y]
        # Passiamo l'intensità del rumore calcolata
        sensor_dist = self.scalar_sensor.observe(true_val, noise_intensity)

        return np.concatenate((pov_flat, sensor_dist)).astype(np.float32)

    def render(self):
        if self.render_mode != 'human': return
        if self.fig is None:
            plt.ion()
            self.fig, self.ax = plt.subplots(figsize=(5,5))
        self.ax.clear()
        self.ax.imshow(self.visited_mask.T, origin='lower', cmap='Blues', vmin=0, vmax=1)
        self.ax.plot(self.agent_pos[1], self.agent_pos[0], 'ro', markersize=10)
        self.ax.set_title(f"Step: {self.current_step}")
        plt.draw()
        plt.pause(0.001)

    def set_noise_levels(self, scalar=None, vector=None):
        pass