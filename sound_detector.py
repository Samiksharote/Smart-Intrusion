import sounddevice as sd
import numpy as np
import librosa
from datetime import datetime
import os
from dotenv import load_dotenv

class SoundDetector:
    def __init__(self):
        load_dotenv()
        self.sample_rate = int(os.getenv('SAMPLE_RATE', 44100))
        self.duration = int(os.getenv('DURATION', 2))  # Increased to 2 seconds for better detection
        self.is_recording = True  # Start as True by default
        # Ambient noise tracking for adaptive thresholds
        self.ambient_level = float(os.getenv('INITIAL_AMBIENT', 1e-6))
        self.ambient_alpha = float(os.getenv('AMBIENT_ALPHA', 0.85))  # EMA smoothing
        # Relative factor above ambient to consider an event
        self.ambient_factor = float(os.getenv('AMBIENT_FACTOR', 1.8))
        
        # Thresholds for different types of sounds
        self.thresholds = {
            'silence': 0.001,        # Much lower to detect more sounds
            'normal_activity': 0.05,  # Very low for better sensitivity
            'loud_activity': 0.1,    # Lowered significantly
            'abnormal': 0.2,         # More sensitive
            'dangerous': 0.3         # Much more sensitive for better detection
        }
        
        # Frequency ranges for different sound types (in Hz)
        self.freq_ranges = {
            'footsteps': (20, 200),     # Low frequency thuds
            'glass_breaking': (1000, 8000),  # Broader range for glass breaking
            'screaming': (800, 3000),   # Broader range for human vocal range
            'dog_barking': (400, 2000),  # Broader range for barking
            'gunshot': (100, 8000),     # Extended range for better gunshot detection
        }
        
        # Duration thresholds (in seconds) for different sounds
        self.duration_thresholds = {
            'gunshot': 0.2,  # Increased to catch slightly longer sounds
            'glass_breaking': 0.3,  # Short duration
            'screaming': 0.5,  # Longer duration
            'dog_barking': 0.4,  # Medium duration
        }
        
        # Peak detection settings
        self.peak_thresholds = {
            'gunshot': 0.4,  # Lower threshold for gunshot detection
            'glass_breaking': 0.6,
            'screaming': 0.5,
            'dog_barking': 0.5
        }

    def start_monitoring(self, verbose=True):
        """Record once and analyze. If verbose is False, suppress debug prints."""
        if not self.is_recording:
            return []

        try:
            # Record audio
            recording = sd.rec(
                int(self.duration * self.sample_rate),
                samplerate=self.sample_rate,
                channels=1,
                dtype=np.float32
            )
            sd.wait()

            # Get sound level
            current_level = np.sqrt(np.mean(np.square(recording)))
            # update ambient level (exponential moving average)
            try:
                self.ambient_level = self.ambient_alpha * self.ambient_level + (1.0 - self.ambient_alpha) * current_level
            except Exception:
                pass
            if verbose:
                print(f"Sound level: {current_level:.4f}")

            # Skip if too quiet
            if current_level <= self.thresholds['silence']:
                return []

            # Analyze sound (pass current_level and ambient for adaptive thresholds)
            sound_type, confidence = self._analyze_sound(recording, current_level=current_level, ambient_level=self.ambient_level)

            # Print debug info
            if sound_type and verbose:
                print(f"Detected: {sound_type} (confidence: {confidence:.2f})")

            # Check if we should alert
            if sound_type and confidence > 0:
                # Special case for footsteps: only alert during night hours
                if sound_type == 'footsteps':
                    current_hour = datetime.now().hour
                    if not (22 <= current_hour or current_hour < 6):
                        return []

                # Save a short audio clip centered on the detected onset (best-effort)
                audio_path = None
                try:
                    # try to compute onset envelope to find the event time
                    try:
                        onset_env = librosa.onset.onset_strength(y=recording.flatten(), sr=self.sample_rate)
                    except Exception:
                        onset_env = None
                    audio_path = self.save_event_clip(recording.flatten(), onset_env)
                except Exception:
                    audio_path = None

                # Return the detection with optional audio path
                return [{
                    'timestamp': datetime.now(),
                    'type': sound_type,
                    'confidence': confidence,
                    'audio_path': audio_path
                }]

        except Exception as e:
            print(f"Sound monitoring error: {str(e)}")

        return []

    def stop_monitoring(self):
        """Stop monitoring sounds"""
        self.is_recording = False

    def _analyze_sound(self, audio_data, current_level=None, ambient_level=None):
        """Analyze the type of sound and return its type and confidence score

        Accepts current_level and ambient_level for adaptive thresholds.
        """
        try:
            # Flatten audio data
            audio_data = audio_data.flatten()
            
            # Calculate basic features
            energy = np.sum(np.square(audio_data)) / len(audio_data)
            zero_crossings = librosa.zero_crossings(audio_data).sum()
            
            # Calculate frequency features
            spectral = np.abs(librosa.stft(audio_data))
            frequencies = librosa.fft_frequencies(sr=self.sample_rate)
            
            # Calculate onset strength (sudden changes in sound)
            onset_env = librosa.onset.onset_strength(y=audio_data, sr=self.sample_rate)
            onset_times = librosa.times_like(onset_env, sr=self.sample_rate)
            
            # Short-term / peak measures (helps detect impulsive sounds like gunshots)
            short_window = int(0.05 * self.sample_rate)  # 50 ms window
            if short_window < 1:
                short_window = 1
            # rolling RMS
            kernel = np.ones(short_window) / short_window
            rms_short = np.sqrt(np.convolve(np.square(audio_data), kernel, mode='valid'))
            short_rms_max = float(np.max(rms_short)) if rms_short.size > 0 else 0.0
            peak = float(np.max(np.abs(audio_data)))

            # Ambient-aware thresholds
            ambient = ambient_level if ambient_level is not None else getattr(self, 'ambient_level', 1e-6)
            rel_factor = getattr(self, 'ambient_factor', 3.0)
            rel_short_thresh = max(self.thresholds['abnormal'], ambient * rel_factor)
            rel_peak_thresh = max(self.thresholds['loud_activity'], ambient * rel_factor)

            # Quick silence/ambient check
            if energy < self.thresholds['silence'] and short_rms_max < (self.thresholds['silence'] * 10) and peak < (self.thresholds['silence'] * 10):
                return None, 0.0  # Silence, no alert needed

            if energy < self.thresholds['normal_activity'] and short_rms_max < self.thresholds['normal_activity'] and peak < self.thresholds['normal_activity']:
                return None, 0.0  # Normal activity, no alert needed

            # Analyze multiple frequency bands for gunshot characteristics first (to avoid mislabeling)
            freq_bands = [
                (100, 500),    # Low frequency impact
                (500, 2000),   # Mid-range blast
                (2000, 8000)   # High frequency crack
            ]

            band_energies = []
            for low, high in freq_bands:
                freq_mask = (frequencies >= low) & (frequencies <= high)
                if not np.any(freq_mask):
                    band_energies.append(0.0)
                else:
                    band_energies.append(np.mean(spectral[freq_mask]))

            # compute HF dominance
            low_e, mid_e, hf_e = band_energies[0], band_energies[1], band_energies[2]
            band_sum = sum(band_energies)
            hf_ratio = hf_e / (low_e + mid_e + 1e-9)

            # Onset metrics
            onset_max = float(np.max(onset_env)) if onset_env.size > 0 else 0.0
            onset_count = int(np.sum(onset_env > 0.3))
            n_onset_frames = len(onset_env) if len(onset_env) > 0 else 1
            duration_seconds = len(audio_data) / float(self.sample_rate)
            max_onset_duration = onset_count * (duration_seconds / n_onset_frames)

            # Gunshot heuristics: impulsive, HF presence, short onset duration, and above ambient
            if band_sum > 0 and (short_rms_max > max(rel_short_thresh, ambient * 0.5) or peak > max(rel_peak_thresh, ambient * 0.8) or onset_max > 0.4):
                if hf_ratio > 0.25 and (peak > ambient * rel_factor or short_rms_max > ambient * rel_factor or onset_max > 0.35):
                    # allow slightly relaxed onset duration but still short
                    if max_onset_duration < max(0.8, self.duration_thresholds.get('gunshot', 0.2) * 4):
                        conf = min(max(short_rms_max, peak) * 2.0, 1.0)
                        return 'gunshot', conf

            # Next check other specific sounds (glass, screaming, footsteps, barking)
            for sound_type, (freq_min, freq_max) in self.freq_ranges.items():
                freq_mask = (frequencies >= freq_min) & (frequencies <= freq_max)
                if not np.any(freq_mask):
                    freq_energy = 0.0
                else:
                    freq_energy = np.mean(spectral[freq_mask])

                # Glass: high-frequency dominant and reasonably above ambient, and not extremely impulsive
                if sound_type == 'glass_breaking':
                    if hf_ratio > 0.5 and (short_rms_max > rel_short_thresh or peak > rel_peak_thresh) and max_onset_duration >= 0.02:
                        return 'glass_breaking', min(max(short_rms_max, energy) * 1.2, 1.0)

                elif sound_type == 'screaming' and (short_rms_max > rel_short_thresh or energy > self.thresholds['abnormal']) and freq_energy > 0.4:
                    return 'screaming', min(max(short_rms_max, energy), 1.0)

                elif sound_type == 'footsteps' and (short_rms_max > self.thresholds['loud_activity'] or energy > self.thresholds['loud_activity']) and freq_energy > 0.25:
                    return 'footsteps', min(max(short_rms_max, energy), 1.0)

                elif sound_type == 'dog_barking' and (short_rms_max > rel_short_thresh or energy > self.thresholds['abnormal']) and freq_energy > 0.3:
                    return 'dog_barking', min(max(short_rms_max, energy), 1.0)

            # If no specific sound type is identified but energy is high
            if short_rms_max > self.thresholds['dangerous'] or (peak > 0.6 and short_rms_max > (self.thresholds['abnormal'] * 0.1)):
                return 'loud_crash', min(max(short_rms_max, energy), 1.0)
            elif short_rms_max > self.thresholds['abnormal'] or energy > self.thresholds['abnormal']:
                return 'abnormal_sound', min(max(short_rms_max, energy), 1.0)

            return None, 0.0
                
        except Exception as e:
            print(f"Error in sound analysis: {str(e)}")
            return None, 0.0

    def save_audio_clip(self, audio_data, clips_dir='sound_clips'):
        """Save the detected abnormal sound"""
        if not os.path.exists(clips_dir):
            os.makedirs(clips_dir)
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(clips_dir, f"abnormal_sound_{timestamp}.wav")
        
        # Save the audio clip
        import scipy.io.wavfile as wav
        # Ensure data is in int16 format for wide compatibility
        try:
            if np.issubdtype(audio_data.dtype, np.floating):
                scaled = np.int16(audio_data / np.max(np.abs(audio_data)) * 32767)
            else:
                scaled = audio_data.astype(np.int16)
        except Exception:
            # fallback: cast directly
            scaled = np.array(audio_data, dtype=np.int16)

        wav.write(filename, self.sample_rate, scaled)
        return filename

    def save_event_clip(self, audio_data, onset_env=None, clips_dir='sound_clips', pre_sec=0.5, post_sec=1.0):
        """Save a short clip around the strongest onset in audio_data.

        - audio_data: 1D numpy array, floats in roughly -1..1 or ints
        - onset_env: optional onset envelope (1D) aligned to frames; if None, a simple RMS-based peak is used
        - pre_sec/post_sec: seconds before/after the detected event to include
        """
        if not os.path.exists(clips_dir):
            os.makedirs(clips_dir)

        sr = self.sample_rate
        n = len(audio_data)

        # determine sample index of event
        event_sample = None
        try:
            if onset_env is not None and len(onset_env) > 0:
                # map onset frame index to sample index
                idx = int(np.argmax(onset_env))
                # onset_env frames correspond to a hop length; approx map using length ratio
                event_sample = int((idx / len(onset_env)) * n)
            else:
                # fallback to short-term RMS peak
                w = int(0.05 * sr)
                if w < 1:
                    w = 1
                rms = np.sqrt(np.convolve(np.square(audio_data), np.ones(w)/w, mode='valid'))
                if rms.size > 0:
                    im = int(np.argmax(rms))
                    event_sample = min(n-1, int((im + w//2)))
        except Exception:
            event_sample = None

        if event_sample is None:
            # default to middle
            event_sample = n // 2

        start = max(0, int(event_sample - pre_sec * sr))
        end = min(n, int(event_sample + post_sec * sr))

        # Debug: log chosen window
        try:
            print(f"Saving event clip: event_sample={event_sample}, start={start}, end={end}, duration={(end-start)/sr:.3f}s")
        except Exception:
            pass

        clip = audio_data[start:end]

        # write clip as 16-bit wav
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(clips_dir, f"abnormal_sound_{timestamp}.wav")
        import scipy.io.wavfile as wav
        try:
            if np.issubdtype(clip.dtype, np.floating):
                if np.max(np.abs(clip)) == 0:
                    scaled = np.int16(clip)
                else:
                    scaled = np.int16(clip / np.max(np.abs(clip)) * 32767)
            else:
                scaled = clip.astype(np.int16)
        except Exception:
            scaled = np.array(clip, dtype=np.int16)

        wav.write(filename, sr, scaled)
        return filename