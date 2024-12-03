# import os
# import shutil
# from PIL import Image
# import face_recognition

# # Input and output directory paths
# INPUT_DIR = "INPUT_DIR"
# OUTPUT_DIR = "OUTPUT_DIR"

# # Ensure the output directory exists
# os.makedirs(OUTPUT_DIR, exist_ok=True)

# # Initialize a report dictionary
# report = {}

# def process_images(input_dir, output_dir):
#     """
#     Process images from the input directory, checking for detectable faces, resizing them,
#     and saving valid images into the output directory.

#     :param input_dir: Path to the directory containing subfolders of images.
#     :param output_dir: Path to the directory where processed images will be saved.
#     """
#     for person in os.listdir(input_dir):
#         person_input_path = os.path.join(input_dir, person)
#         person_output_path = os.path.join(output_dir, person)

#         # Skip non-directory entries
#         if not os.path.isdir(person_input_path):
#             continue

#         # Create a subfolder in the output directory for this person
#         os.makedirs(person_output_path, exist_ok=True)

#         # Initialize counts for the report
#         accepted = 0
#         rejected = 0

#         print(f"Processing folder: {person}...")  # Feedback for folder being processed

#         for image_name in os.listdir(person_input_path):
#             image_path = os.path.join(person_input_path, image_name)
#             print(f"  Processing image: {image_name}...")  # Feedback for each image

#             try:
#                 # Load the image using face_recognition
#                 image = face_recognition.load_image_file(image_path)
#                 face_locations = face_recognition.face_locations(image)

#                 if face_locations:
#                     # Resize the first detected face to 224x224
#                     top, right, bottom, left = face_locations[0]
#                     face_image = image[top:bottom, left:right]
#                     pil_image = Image.fromarray(face_image)
#                     pil_image = pil_image.resize((224, 224))

#                     # Save the processed image
#                     output_path = os.path.join(person_output_path, image_name)
#                     pil_image.save(output_path)
#                     accepted += 1
#                 else:
#                     # No face detected
#                     print(f"    No face detected in {image_name}. Skipping...")
#                     rejected += 1
#             except Exception as e:
#                 print(f"    Error processing {image_name}: {e}")
#                 rejected += 1

#         # Add counts to the report
#         report[person] = {"accepted": accepted, "rejected": rejected}
#         print(f"  Folder {person} completed: {accepted} accepted, {rejected} rejected.")

# def generate_report(report):
#     """
#     Generate and print a report based on processed images.

#     :param report: Dictionary containing counts of accepted and rejected images per person.
#     """
#     print("\nProcessing Report:")
#     for person, counts in report.items():
#         print(f"Person: {person}")
#         print(f"  Accepted: {counts['accepted']}")
#         print(f"  Rejected: {counts['rejected']}")

# # Process the images and generate a report
# process_images(INPUT_DIR, OUTPUT_DIR)
# generate_report(report)





# import os
# import cv2
# from PIL import Image
# import face_recognition

# # Input and output directory paths
# INPUT_DIR = "INPUT_DIR"
# OUTPUT_DIR = "OUTPUT_DIR"

# # Ensure the output directory exists
# os.makedirs(OUTPUT_DIR, exist_ok=True)

# # Initialize a report dictionary
# report = {}

# def detect_faces(image_path):
#     """
#     Detect faces using face_recognition and OpenCV as fallback.
#     """
#     # Try face_recognition first
#     image = face_recognition.load_image_file(image_path)
#     face_locations = face_recognition.face_locations(image, model="cnn")  # Use "cnn" for better accuracy

#     if face_locations:
#         return face_locations

#     # Fallback to OpenCV
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
#     faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

#     # Convert OpenCV format to face_recognition format
#     face_locations = [(y, x+w, y+h, x) for (x, y, w, h) in faces]
#     return face_locations

# def process_images(input_dir, output_dir):
#     for person in os.listdir(input_dir):
#         person_input_path = os.path.join(input_dir, person)
#         person_output_path = os.path.join(output_dir, person)

#         if not os.path.isdir(person_input_path):
#             continue

#         os.makedirs(person_output_path, exist_ok=True)

#         accepted = 0
#         rejected = 0

#         print(f"Processing folder: {person}...")

#         for image_name in os.listdir(person_input_path):
#             image_path = os.path.join(person_input_path, image_name)
#             print(f"  Processing image: {image_name}...")

#             try:
#                 face_locations = detect_faces(image_path)

#                 if face_locations:
#                     top, right, bottom, left = face_locations[0]
#                     image = face_recognition.load_image_file(image_path)
#                     face_image = image[top:bottom, left:right]
#                     pil_image = Image.fromarray(face_image)
#                     pil_image = pil_image.resize((224, 224))

#                     output_path = os.path.join(person_output_path, image_name)
#                     pil_image.save(output_path)
#                     accepted += 1
#                 else:
#                     print(f"    No face detected in {image_name}. Skipping...")
#                     rejected += 1
#             except Exception as e:
#                 print(f"    Error processing {image_name}: {e}")
#                 rejected += 1

#         report[person] = {"accepted": accepted, "rejected": rejected}
#         print(f"  Folder {person} completed: {accepted} accepted, {rejected} rejected.")

# def generate_report(report):
#     print("\nProcessing Report:")
#     for person, counts in report.items():
#         print(f"Person: {person}")
#         print(f"  Accepted: {counts['accepted']}")
#         print(f"  Rejected: {counts['rejected']}")

# process_images(INPUT_DIR, OUTPUT_DIR)
# generate_report(report)




import os
import cv2
import face_recognition
import numpy as np
from PIL import Image, ExifTags
from concurrent.futures import ThreadPoolExecutor


def correct_image_orientation(image_path):
    """
    Corrige l'orientation de l'image à partir des métadonnées EXIF.
    
    Args:
        image_path (str): Chemin de l'image.
    
    Returns:
        np.ndarray: Image corrigée.
    """
    try:
        pil_image = Image.open(image_path)
        
        # Identifier les métadonnées EXIF pour corriger l'orientation
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == "Orientation":
                break

        exif = pil_image._getexif()
        if exif is not None:
            orientation = exif.get(orientation)
            if orientation == 3:
                pil_image = pil_image.rotate(180, expand=True)
            elif orientation == 6:
                pil_image = pil_image.rotate(270, expand=True)
            elif orientation == 8:
                pil_image = pil_image.rotate(90, expand=True)
        
        # Convertir l'image corrigée en format NumPy
        return np.array(pil_image)
    except Exception as e:
        print(f"Erreur lors de la correction de l'orientation pour {image_path}: {e}")
        return None


def preprocess_image(image_path, output_size=(128, 128)):
    """
    Prétraitement d'une image pour la reconnaissance faciale.
    """
    try:
        # Corriger l'orientation de l'image
        corrected_image = correct_image_orientation(image_path)
        if corrected_image is None:
            return None
        
        # Détection des visages
        face_locations = face_recognition.face_locations(corrected_image)
        if face_locations:
            # Prendre le premier visage détecté
            top, right, bottom, left = face_locations[0]
            face_image = corrected_image[top:bottom, left:right]
        else:
            # Utiliser l'image entière si aucun visage détecté
            face_image = corrected_image

        # Redimensionner et normaliser l'image
        face_image = cv2.resize(face_image, output_size)
        face_image = face_image / 255.0  # Normalisation

        return face_image
    except Exception as e:
        print(f"Erreur lors du traitement de {image_path}: {e}")
        return None


def preprocess_and_save(image_path, output_path, output_size=(128, 128)):
    """
    Prétraite une image et la sauvegarde.
    """
    processed_image = preprocess_image(image_path, output_size)
    if processed_image is not None:
        processed_image_uint8 = (processed_image * 255).astype(np.uint8)
        cv2.imwrite(output_path, cv2.cvtColor(processed_image_uint8, cv2.COLOR_RGB2BGR))
        print(f"Image traitée et sauvegardée : {output_path}")


def preprocess_dataset(input_dir, output_dir, output_size=(128, 128)):
    """
    Prétraite toutes les images d'un dossier.
    """
    os.makedirs(output_dir, exist_ok=True)

    tasks = []
    with ThreadPoolExecutor() as executor:
        for person_name in os.listdir(input_dir):
            person_dir = os.path.join(input_dir, person_name)
            output_person_dir = os.path.join(output_dir, person_name)
            
            if not os.path.isdir(person_dir):
                continue
            
            os.makedirs(output_person_dir, exist_ok=True)
            
            for image_name in os.listdir(person_dir):
                if not image_name.lower().endswith(('jpg', 'jpeg', 'png', 'webp')):
                    continue
                
                image_path = os.path.join(person_dir, image_name)
                output_path = os.path.join(output_person_dir, image_name)
                
                # Ajouter une tâche pour traitement parallèle
                tasks.append(executor.submit(preprocess_and_save, image_path, output_path, output_size))
        
        # Attendre la fin de toutes les tâches
        for task in tasks:
            task.result()


# Exécuter le prétraitement



input_directory = 'INPUT_DIR'  # Chemin vers les images originales
output_directory = 'OUTPUT_DIR'  # Chemin de sauvegarde
preprocess_dataset(input_directory, output_directory, output_size=(128, 128))
