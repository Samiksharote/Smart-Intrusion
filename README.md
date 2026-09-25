# Smart Intrusion Detection System

A machine learning-based security system that detects intruders using face recognition, monitors for abnormal sounds, and sends real-time alerts to your phone via Telegram.

## Features

- Face detection and recognition using OpenCV and face_recognition
- Abnormal sound detection using audio processing
- Real-time alerts via Telegram
- Saves intruder faces and abnormal sound clips
- Easy configuration through environment variables

## Setup Instructions

1. Install required dependencies:
```bash
pip install -r requirements.txt
```

2. Configure the system:
   - Copy `.env.example` to `.env`
   - Get a Telegram bot token from BotFather
   - Add your Telegram chat ID
   - Update the environment variables in `.env`

3. Add known faces:
   - Create a `known_faces` directory
   - Add photos of known people (one face per image)
   - Name the files with the person's name (e.g., `john.jpg`)

## Usage

Run the system:
```bash
python main.py
```

The system will:
- Monitor for unknown faces using your camera
- Detect abnormal sounds using your microphone
- Send alerts to your Telegram when intruders or abnormal sounds are detected
- Save intruder faces and sound clips for review

## Directory Structure

```
Smart Intrusion/
├── main.py              # Main application
├── face_detector.py     # Face detection module
├── sound_detector.py    # Sound detection module
├── alert_system.py      # Alert system module
├── requirements.txt     # Dependencies
├── .env                 # Configuration
├── known_faces/         # Known face images
├── intruder_faces/      # Saved intruder faces
└── sound_clips/         # Saved abnormal sounds
```

## Configuration

Edit the `.env` file to customize:
- Telegram bot token and chat ID
- Camera source
- Sound detection threshold
- Audio sampling settings

## Requirements

- Python 3.8+
- OpenCV
- face_recognition
- TensorFlow
- sounddevice
- librosa
- python-telegram-bot
- Other dependencies listed in requirements.txt