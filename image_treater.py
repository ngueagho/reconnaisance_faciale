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





# import os
# import cv2
# import face_recognition
# import numpy as np
# from PIL import Image, ExifTags
# from concurrent.futures import ThreadPoolExecutor


# def correct_image_orientation(image_path):
#     """
#     Corrige l'orientation de l'image à partir des métadonnées EXIF.
    
#     Args:
#         image_path (str): Chemin de l'image.
    
#     Returns:
#         np.ndarray: Image corrigée.
#     """
#     try:
#         pil_image = Image.open(image_path)
        
#         # Identifier les métadonnées EXIF pour corriger l'orientation
#         for orientation in ExifTags.TAGS.keys():
#             if ExifTags.TAGS[orientation] == "Orientation":
#                 break

#         exif = pil_image._getexif()
#         if exif is not None:
#             orientation = exif.get(orientation)
#             if orientation == 3:
#                 pil_image = pil_image.rotate(180, expand=True)
#             elif orientation == 6:
#                 pil_image = pil_image.rotate(270, expand=True)
#             elif orientation == 8:
#                 pil_image = pil_image.rotate(90, expand=True)
        
#         # Convertir l'image corrigée en format NumPy
#         return np.array(pil_image)
#     except Exception as e:
#         print(f"Erreur lors de la correction de l'orientation pour {image_path}: {e}")
#         return None


# def preprocess_image(image_path, output_size=(128, 128)):
#     """
#     Prétraitement d'une image pour la reconnaissance faciale.
#     """
#     try:
#         # Corriger l'orientation de l'image
#         corrected_image = correct_image_orientation(image_path)
#         if corrected_image is None:
#             return None
        
#         # Détection des visages
#         face_locations = face_recognition.face_locations(corrected_image)
#         if face_locations:
#             # Prendre le premier visage détecté
#             top, right, bottom, left = face_locations[0]
#             face_image = corrected_image[top:bottom, left:right]
#         else:
#             # Utiliser l'image entière si aucun visage détecté
#             face_image = corrected_image

#         # Redimensionner et normaliser l'image
#         face_image = cv2.resize(face_image, output_size)
#         face_image = face_image / 255.0  # Normalisation

#         return face_image
#     except Exception as e:
#         print(f"Erreur lors du traitement de {image_path}: {e}")
#         return None


# def preprocess_and_save(image_path, output_path, output_size=(128, 128)):
#     """
#     Prétraite une image et la sauvegarde au format PNG.
#     """
#     processed_image = preprocess_image(image_path, output_size)
#     if processed_image is not None:
#         processed_image_uint8 = (processed_image * 255).astype(np.uint8)
        
#         # Convertir le chemin de sortie au format PNG
#         output_path_png = os.path.splitext(output_path)[0] + '.png'
        
#         # Sauvegarder en PNG
#         cv2.imwrite(output_path_png, cv2.cvtColor(processed_image_uint8, cv2.COLOR_RGB2BGR))
#         print(f"Image traitée et sauvegardée : {output_path_png}")


# def preprocess_dataset(input_dir, output_dir, output_size=(128, 128)):
#     """
#     Prétraite toutes les images d'un dossier.
#     """
#     os.makedirs(output_dir, exist_ok=True)

#     tasks = []
#     with ThreadPoolExecutor() as executor:
#         for person_name in os.listdir(input_dir):
#             person_dir = os.path.join(input_dir, person_name)
#             output_person_dir = os.path.join(output_dir, person_name)
            
#             if not os.path.isdir(person_dir):
#                 continue
            
#             os.makedirs(output_person_dir, exist_ok=True)
            
#             for image_name in os.listdir(person_dir):
#                 if not image_name.lower().endswith(('jpg', 'jpeg', 'png', 'webp')):
#                     continue
                
#                 image_path = os.path.join(person_dir, image_name)
#                 output_path = os.path.join(output_person_dir, image_name)
                
#                 # Ajouter une tâche pour traitement parallèle
#                 tasks.append(executor.submit(preprocess_and_save, image_path, output_path, output_size))
        
#         # Attendre la fin de toutes les tâches
#         for task in tasks:
#             task.result()


# input_directory = 'INPUT_DIR'  # Chemin vers les images originales
# output_directory = 'OUTPUT_DIR'  # Chemin de sauvegarde
# preprocess_dataset(input_directory, output_directory, output_size=(128, 128))
