import cv2
import face_recognition

image = face_recognition.load_image_file("/home/theking/Documents/infosM1/projet2/easy_facial_recognition/ROBERTO/INPUT_DIR/CRYSTALE/20241014_153214.jpg")
face_locations = face_recognition.face_locations(image)

print(face_locations)  # Affiche les coordonnées des visages détectés
