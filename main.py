import asyncio
import os
from face_detector import FaceDetector
from face_tracker import FaceTracker
from sound_detector import SoundDetector
from alert_system import AlertSystem
import cv2

async def process_frame(frame, detected_faces, face_detector, alert_system):
    """Process detected faces and send alerts if needed"""
    for face in detected_faces:
        if face['name'] == "Unknown":
            # Save intruder's face
            image_path = face_detector.save_intruder_face(
                frame, 
                face['location'],
            )
            
            try:
                # Send alert
                await alert_system.send_alert(
                    "⚠️ Intruder detected! Unknown person in the premises.",
                    image_path
                )
            except Exception as e:
                print(f"Failed to send alert: {str(e)}")

async def monitor_sound(sound_detector, alert_system, face_detector):
    """Monitor for abnormal sounds and send Telegram alerts. Suppress console noise while camera is open."""
    try:
        while True:
            camera_open = False
            try:
                camera_open = bool(face_detector.cap and face_detector.cap.isOpened())
            except Exception:
                camera_open = False

            # If camera is open, suppress verbose prints from start_monitoring
            events = sound_detector.start_monitoring(verbose=not camera_open)

            if events:
                for sound_event in events:
                    try:
                        stype = sound_event.get('type')
                        confidence = sound_event.get('confidence', 0.0)
                        audio_path = sound_event.get('audio_path')
                        message = f"🔊 Abnormal sound detected!\nType: {stype}\nConfidence: {confidence:.2%}"
                        # Attach audio clip if available by sending as document
                        await alert_system.send_alert(message, image_path=None, is_intruder=False)
                        # Try sending audio file as document if available
                        if audio_path:
                            try:
                                await alert_system.send_audio_clip(audio_path)
                            except Exception as e:
                                print(f"Failed to send audio clip: {e}")
                    except Exception as e:
                        print(f"Failed to process sound event: {str(e)}")

            await asyncio.sleep(0.1)
    except asyncio.CancelledError:
        return

async def main():
    # Initialize components
    face_detector = FaceDetector()
    sound_detector = SoundDetector()
    alert_system = AlertSystem()

    print("System initialized. Starting surveillance...")

    try:
        # Start the camera
        face_detector.start_camera()

        # Start sound monitoring in a separate task
        sound_monitoring_task = asyncio.create_task(monitor_sound(sound_detector, alert_system, face_detector))

        # Per-face tracker (debounce specific unknown faces)
        PER_FACE_THRESHOLD = int(os.getenv('PER_FACE_THRESHOLD', 3))
        PER_FACE_EXPIRY = float(os.getenv('PER_FACE_EXPIRY', 2.0))
        PER_FACE_DISTANCE = float(os.getenv('PER_FACE_DISTANCE', 60.0))
        tracker = FaceTracker(threshold=PER_FACE_THRESHOLD, expiry_seconds=PER_FACE_EXPIRY, distance_pixels=PER_FACE_DISTANCE)

        # Main loop for face detection
        while True:
            frame, detected_faces = face_detector.detect_faces()
            
            if frame is None:
                print("Failed to get frame from camera")
                await asyncio.sleep(1)
                continue

            # Per-face debounce: find confirmed unknowns
            confirmed = tracker.update(detected_faces)
            if confirmed:
                # For each confirmed unknown, save the cropped face and send an alert
                for item in confirmed:
                    loc = item.get('location')
                    try:
                        image_path = face_detector.save_intruder_face(frame, loc)
                        message = "⚠️ Intruder detected! Unknown person in the premises."
                        await alert_system.send_alert(message, image_path)
                    except Exception as e:
                        print(f"Failed to send per-face alert: {str(e)}")

            # Display the frame
            cv2.imshow('Security Camera', frame)
            
            # Check for quit command
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # Small delay to prevent CPU overload
            await asyncio.sleep(0.01)

    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        # Cleanup
        face_detector.stop_camera()
        sound_detector.stop_monitoring()
        cv2.destroyAllWindows()
        if 'sound_monitoring_task' in locals():
            sound_monitoring_task.cancel()
            try:
                await sound_monitoring_task
            except asyncio.CancelledError:
                pass

if __name__ == "__main__":
    asyncio.run(main())