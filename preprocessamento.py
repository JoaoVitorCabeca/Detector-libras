import numpy as np


# Função para normalizar as coordenadas
def normalizar_landmarks(landmarks):
    # Normaliza os valores de x, y, z para estarem entre 0 e 1
    landmarks = np.array(landmarks)
    max_val = np.max(landmarks)
    min_val = np.min(landmarks)
    normalized_landmarks = (landmarks - min_val) / (max_val - min_val)
    return normalized_landmarks
