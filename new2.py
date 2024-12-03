import os
import cv2
import face_recognition
import pickle
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
from tqdm import tqdm
import numpy as np

# Fichier pour sauvegarder les encodages
ENCODINGS_FILE = "face_encodings.pkl"

def load_encodings():
    """Charger les encodages existants depuis un fichier."""
    if os.path.exists(ENCODINGS_FILE):
        with open(ENCODINGS_FILE, "rb") as f:
            return pickle.load(f)
    return {}

def save_encodings(encodings):
    """Sauvegarder les encodages dans un fichier."""
    with open(ENCODINGS_FILE, "wb") as f:
        pickle.dump(encodings, f)

def is_valid_image(file_path):
    """Vérifier si le fichier est une image valide."""
    try:
        img = cv2.imread(file_path)
        return img is not None
    except:
        return False

def encode_faces(dataset_path):
    """Encoder les visages depuis un répertoire de dataset."""
    encodings = load_encodings()

    # Lister les personnes dans le dataset
    people = [person for person in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, person))]
    total_images = sum([len(os.listdir(os.path.join(dataset_path, person))) for person in people])

    print(f"Total images to process: {total_images}")
    progress = tqdm(total=total_images, desc="Encoding Faces")

    for person_name in people:
        person_path = os.path.join(dataset_path, person_name)
        image_files = os.listdir(person_path)

        for image_file in image_files:
            image_path = os.path.join(person_path, image_file)

            # Vérifier la validité de l'image
            if not is_valid_image(image_path):
                print(f"Skipping invalid image: {image_path}")
                progress.update(1)
                continue

            try:
                image = face_recognition.load_image_file(image_path)

                # Redimensionner l'image pour réduire la charge mémoire
                if image.shape[1] > 800:
                    scale_percent = 800 / image.shape[1]  # Échelle pour largeur max de 800 px
                    new_width = int(image.shape[1] * scale_percent)
                    new_height = int(image.shape[0] * scale_percent)
                    image = cv2.resize(image, (new_width, new_height))

                face_locations = face_recognition.face_locations(image)
                face_encodings = face_recognition.face_encodings(image, face_locations)

                if face_encodings:
                    encodings[person_name] = encodings.get(person_name, []) + face_encodings
                else:
                    print(f"Warning: No face detected in {image_path}")
            except Exception as e:
                print(f"Error processing {image_path}: {e}")

            progress.update(1)

    progress.close()
    save_encodings(encodings)
    print("Encoding complete!")

def recognize_faces():
    """Reconnaître les visages en temps réel avec la webcam."""
    encodings = load_encodings()
    known_names = list(encodings.keys())
    known_encodings = [enc for name in known_names for enc in encodings[name]]

    video_capture = cv2.VideoCapture(0)

    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("Erreur : Impossible de lire la vidéo.")
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.6)
            name = "Unknown"

            if True in matches:
                first_match_index = matches.index(True)
                name = known_names[first_match_index]

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

def verify_identity(input_name):
    """Vérifier si la personne correspond à un nom donné."""
    encodings = load_encodings()
    known_encodings = encodings.get(input_name, [])
    if not known_encodings:
        print(f"No encodings found for {input_name}")
        return

    video_capture = cv2.VideoCapture(0)
    print("Veuillez regarder la webcam pour vérification...")
    verified = False

    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("Erreur : Impossible de lire la vidéo.")
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.6)
            if True in matches:
                verified = True
                break

        if verified:
            print(f"Identity verified for {input_name}")
            break

        cv2.imshow('Verification', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

def create_gui():
    """Créer une interface graphique avec Tkinter."""
    root = tk.Tk()
    root.title("Système de Reconnaissance Faciale")

    tk.Label(root, text="Système de Reconnaissance Faciale", font=("Helvetica", 16)).pack(pady=10)

    def train_model():
        dataset_path = filedialog.askdirectory(title="Sélectionner le Répertoire de Dataset")
        if dataset_path:
            encode_faces(dataset_path)
            messagebox.showinfo("Formation Terminée", "Modèle entraîné avec succès!")

    def start_recognition():
        recognize_faces()

    def verify_person():
        input_name = simpledialog.askstring("Vérification d'Identité", "Entrez le Nom:")
        if input_name:
            verify_identity(input_name)

    tk.Button(root, text="Entraîner le Modèle", command=train_model, width=25).pack(pady=5)
    tk.Button(root, text="Démarrer la Reconnaissance", command=start_recognition, width=25).pack(pady=5)
    tk.Button(root, text="Vérifier une Identité", command=verify_person, width=25).pack(pady=5)

    root.mainloop()

if __name__ == "__main__":
    create_gui()
