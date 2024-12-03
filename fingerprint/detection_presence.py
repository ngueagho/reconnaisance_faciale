import pyudev

# Identifiants du capteur d'empreintes digitales
ID_VENDOR = "2541"  # Chipsailing
ID_PRODUCT = "0236"  # CS9711 Fingerprint

def detect_usb_device():
    try:
        context = pyudev.Context()
        monitor = pyudev.Monitor.from_netlink(context)
        monitor.filter_by(subsystem='usb')

        print("En attente du branchement ou du débranchement du capteur d'empreintes digitales...")

        for device in iter(monitor.poll, None):
            vendor_id = device.get('ID_VENDOR_ID')
            product_id = device.get('ID_MODEL_ID')

            if device.action == 'add' and vendor_id == ID_VENDOR and product_id == ID_PRODUCT:
                print(f"[INFO] Capteur d'empreintes digitales branché : {device.device_path}")
            elif device.action == 'remove' and vendor_id == ID_VENDOR and product_id == ID_PRODUCT:
                print("[INFO] Capteur d'empreintes digitales débranché")
    except Exception as e:
        print(f"[ERREUR] Une erreur est survenue : {e}")

if __name__ == "__main__":
    detect_usb_device()




# import pyudev
# import os
# import time
# from fingerprint_sdk import FingerprintSensor  # Remplacez par la bibliothèque réelle

# # Identifiants du capteur d'empreintes digitales
# ID_VENDOR = "2541"  # Chipsailing
# ID_PRODUCT = "0236"  # CS9711 Fingerprint

# # Répertoire pour enregistrer les images d'empreintes
# OUTPUT_DIR = "./empreintes"

# def capture_fingerprint_image(sensor):
#     """Capture une image de l'empreinte et la sauvegarde."""
#     try:
#         print("[INFO] Capture de l'image de l'empreinte...")
#         image_data = sensor.capture_image()  # Fonction dépendant du SDK réel
#         timestamp = time.strftime("%Y%m%d_%H%M%S")
#         filename = os.path.join(OUTPUT_DIR, f"empreinte_{timestamp}.bmp")
        
#         with open(filename, "wb") as image_file:
#             image_file.write(image_data)
        
#         print(f"[INFO] Image de l'empreinte enregistrée : {filename}")
#     except Exception as e:
#         print(f"[ERREUR] Échec de la capture : {e}")

# def detect_usb_device():
#     context = pyudev.Context()
#     monitor = pyudev.Monitor.from_netlink(context)
#     monitor.filter_by(subsystem='usb')

#     print("En attente du branchement ou du débranchement du capteur d'empreintes digitales...")

#     for device in iter(monitor.poll, None):
#         vendor_id = device.get('ID_VENDOR_ID')
#         product_id = device.get('ID_MODEL_ID')

#         if device.action == 'add' and vendor_id == ID_VENDOR and product_id == ID_PRODUCT:
#             print(f"[INFO] Capteur d'empreintes digitales branché : {device.device_path}")
#             # Initialiser le capteur
#             try:
#                 sensor = FingerprintSensor(device.device_node)
#                 sensor.initialize()
#                 print("[INFO] Capteur initialisé avec succès.")
#                 capture_fingerprint_image(sensor)
#             except Exception as e:
#                 print(f"[ERREUR] Impossible d'initialiser le capteur : {e}")
#         elif device.action == 'remove' and vendor_id == ID_VENDOR and product_id == ID_PRODUCT:
#             print("[INFO] Capteur d'empreintes digitales débranché")

# if __name__ == "__main__":
#     # Créer le dossier de sortie si nécessaire
#     os.makedirs(OUTPUT_DIR, exist_ok=True)
#     detect_usb_device()
