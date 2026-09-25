import cv2
import numpy as np
from datetime import datetime
import os
from dotenv import load_dotenv
import face_recognition

class FaceDetector:
    def __init__(self):
        load_dotenv()
        self.camera_source = int(os.getenv('CAMERA_SOURCE', 0))
        self.cap = None
        self.known_face_encodings = []
        self.known_face_names = []
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.load_known_faces()

    def load_known_faces(self, faces_dir='known_faces'):
        """Load known faces from the faces directory"""
        if not os.path.exists(faces_dir):
            os.makedirs(faces_dir)
            print(f"Created {faces_dir} directory. Please add family members' photos.")
            return

        for filename in os.listdir(faces_dir):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                try:
                    # Load the image file
                    file_path = os.path.join(faces_dir, filename)
                    image = face_recognition.load_image_file(file_path)
                    
                    # Get face encodings (features)
                    face_encodings = face_recognition.face_encodings(image)
                    
                    if len(face_encodings) > 0:
                        # Take the first face found in the image
                        face_encoding = face_encodings[0]
                        # Get the name from the filename (without extension)
                        name = os.path.splitext(filename)[0]
                        
                        self.known_face_encodings.append(face_encoding)
                        self.known_face_names.append(name)
                        print(f"Loaded known face: {name}")
                    else:
                        print(f"No face found in {filename}")
                except Exception as e:
                    print(f"Error loading {filename}: {str(e)}")

        if not self.known_face_encodings:
            print("No known faces loaded. Please add family photos to the known_faces directory.")

    def start_camera(self):
        """Initialize the camera"""
        self.cap = cv2.VideoCapture(self.camera_source)
        if not self.cap.isOpened():
            raise RuntimeError("Could not start camera")

    def stop_camera(self):
        """Release the camera"""
        if self.cap:
            self.cap.release()

    def detect_faces(self):
        """Detect and recognize faces in the current frame"""
        if not self.cap:
            self.start_camera()

        ret, frame = self.cap.read()
        if not ret:
            return None, []

        # Process every other frame to reduce lag
        if hasattr(self, 'frame_count'):
            self.frame_count += 1
        else:
            self.frame_count = 0

        if self.frame_count % 2 == 0:  # Only process every other frame
            # Resize frame for faster face recognition processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            
            # Convert from BGR to RGB
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

            # Find face locations and encodings using HOG model (faster than CNN)
            face_locations = face_recognition.face_locations(rgb_small_frame, model="hog")
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            
            # Store the results for next frame
            self.last_face_locations = face_locations
            self.last_face_encodings = face_encodings
        else:
            # Use the results from previous frame
            face_locations = getattr(self, 'last_face_locations', [])
            face_encodings = getattr(self, 'last_face_encodings', [])

        detected_faces = []
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # Scale back up face locations
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Check if the face matches any known faces with stricter tolerance
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding, tolerance=0.45)  # Stricter tolerance
            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            name = "Unknown"
            face_color = (0, 0, 255)  # Red for unknown faces

            if True in matches and len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index] and face_distances[best_match_index] < 0.45:  # Double check with distance
                    name = self.known_face_names[best_match_index]
                    face_color = (0, 255, 0)  # Green for known faces

            detected_faces.append({
                'name': name,
                'location': (top, right, bottom, left),
                'timestamp': datetime.now()
            })

            # Draw rectangle around face
            cv2.rectangle(frame, (left, top), (right, bottom), face_color, 2)
            
            # Draw name below face
            cv2.putText(frame, name, (left, bottom + 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.75, face_color, 2)

        return frame, detected_faces

    def save_intruder_face(self, frame, face_location, intruders_dir='intruder_faces'):
        """Save detected intruder face"""
        if not os.path.exists(intruders_dir):
            os.makedirs(intruders_dir)

        top, right, bottom, left = face_location
        face_image = frame[top:bottom, left:right]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(intruders_dir, f"intruder_{timestamp}.jpg")
        cv2.imwrite(filename, face_image)
        return filename