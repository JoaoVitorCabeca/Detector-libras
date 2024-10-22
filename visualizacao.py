import cv2

# Função para exibir landmarks e conexões na imagem
def exibir_landmarks(image, hand_landmarks, mp_drawing, mp_hands):
    mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    return image