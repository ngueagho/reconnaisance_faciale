import os
import cv2
import face_recognition
import pickle
import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
from tqdm import tqdm  # Import tqdm for the progress bar


# Directory to save encodings
ENCODINGS_FILE = "face_encodings.pkl"

def load_encodings():
    """Load existing face encodings from a file."""
    if os.path.exists(ENCODINGS_FILE):
        with open(ENCODINGS_FILE, "rb") as f:
            return pickle.load(f)
    return {}

def save_encodings(encodings):
    """Save face encodings to a file."""
    with open(ENCODINGS_FILE, "wb") as f:
        pickle.dump(encodings, f)




def encode_faces(dataset_path):
    """Encode faces from the dataset directory with a progress bar."""
    encodings = load_encodings()
    # Get the list of people and images
    people = [person for person in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, person))]
    total_images = sum(len(os.listdir(os.path.join(dataset_path, person))) for person in people)
    print(f"Total images to process: {total_images}")

    # Initialize the progress bar
    with tqdm(total=total_images, desc="Encoding Faces", ncols=80) as progress:
        for person_name in people:
            person_path = os.path.join(dataset_path, person_name)
            for image_file in os.listdir(person_path):
                image_path = os.path.join(person_path, image_file)

                # Skip directories or non-image files
                if not os.path.isfile(image_path) or not image_file.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                    continue

                try:
                    # Process the image
                    image = face_recognition.load_image_file(image_path)
                    face_locations = face_recognition.face_locations(image)
                    face_encodings = face_recognition.face_encodings(image, face_locations)

                    if face_encodings:
                        encodings[person_name] = encodings.get(person_name, []) + face_encodings
                    else:
                        print(f"Warning: No face detected in {image_path}")
                except Exception as e:
                    print(f"Error processing {image_path}: {e}")

                # Update the progress bar
                progress.update(1)

    save_encodings(encodings)
    print("Encoding complete!")






def recognize_faces():
    """Recognize faces in real-time using the webcam."""
    encodings = load_encodings()
    if not encodings:
        print("No face encodings found. Please train the model first.")
        return

    known_names = list(encodings.keys())
    known_encodings = [enc for name in known_names for enc in encodings[name]]

    video_capture = cv2.VideoCapture(0)

    while True:
        ret, frame = video_capture.read()
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
    """Verify if the person in front of the webcam matches the input name."""
    encodings = load_encodings()
    known_encodings = encodings.get(input_name, [])
    if not known_encodings:
        print(f"No encodings found for {input_name}")
        return

    video_capture = cv2.VideoCapture(0)
    print("Please look at the webcam for verification...")
    verified = False

    while True:
        ret, frame = video_capture.read()
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

def add_new_person(person_name, images_path=None):
    """Add a new person by capturing images or using an existing folder."""
    dataset_path = "dataset"
    person_path = os.path.join(dataset_path, person_name)
    os.makedirs(person_path, exist_ok=True)

    if images_path:
        for image_file in os.listdir(images_path):
            src_path = os.path.join(images_path, image_file)
            dest_path = os.path.join(person_path, image_file)
            os.rename(src_path, dest_path)
    else:
        print("Capturing images for new person. Press 'q' to quit.")
        video_capture = cv2.VideoCapture(0)
        count = 0
        while True:
            ret, frame = video_capture.read()
            cv2.imshow("Capture Images", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            img_path = os.path.join(person_path, f"{person_name}_{count}.jpg")
            cv2.imwrite(img_path, frame)
            count += 1
        video_capture.release()
        cv2.destroyAllWindows()

    encode_faces(dataset_path)

# GUI using tkinter
def create_gui():
    root = tk.Tk()
    root.title("Facial Recognition System")

    tk.Label(root, text="Facial Recognition System", font=("Helvetica", 16)).pack(pady=10)

    def train_model():
        dataset_path = filedialog.askdirectory(title="Select Dataset Directory")
        if dataset_path:
            encode_faces(dataset_path)
            messagebox.showinfo("Training Complete", "Model trained successfully!")

    def start_recognition():
        recognize_faces()

    def verify_person():
        input_name = tk.simpledialog.askstring("Verify Identity", "Enter Name:")
        if input_name:
            verify_identity(input_name)

    def add_person():
        person_name = tk.simpledialog.askstring("Add Person", "Enter Person's Name:")
        if person_name:
            add_new_person(person_name)

    tk.Button(root, text="Train Model", command=train_model, width=25).pack(pady=5)
    tk.Button(root, text="Start Recognition", command=start_recognition, width=25).pack(pady=5)
    tk.Button(root, text="Verify Identity", command=verify_person, width=25).pack(pady=5)
    tk.Button(root, text="Add New Person", command=add_person, width=25).pack(pady=5)

    root.mainloop()

if __name__ == "__main__":
    create_gui()
