import numpy as np
from sound_detector import SoundDetector


def make_gunshot(sr, duration=2.0):
    n = int(sr * duration)
    t = np.linspace(0, duration, n, endpoint=False)
    audio = np.random.normal(0, 1e-4, size=n)

    # Short impulsive burst (very short, high amplitude)
    start = int(0.5 * sr)
    burst_len = int(0.01 * sr)  # 10 ms burst
    window = np.hanning(burst_len)

    # Make the burst high-frequency dominant (simulate crack)
    hf_burst = np.sin(2 * np.pi * 6000 * t[:burst_len]) * 8.0
    audio[start:start+burst_len] += window * hf_burst

    # Add a small broad HF component around the burst
    hf = 0.6 * np.sin(2 * np.pi * 3000 * t)
    audio += hf * np.exp(-80 * (t - 0.5)**2)

    # Normalize
    audio = audio / np.max(np.abs(audio))
    return audio.astype(np.float32)


def make_glass_break(sr, duration=2.0):
    n = int(sr * duration)
    t = np.linspace(0, duration, n, endpoint=False)
    audio = np.random.normal(0, 1e-4, size=n)

    # High-frequency burst spread over a slightly longer window
    start = int(0.6 * sr)
    burst_len = int(0.03 * sr)  # 30 ms
    window = np.hanning(burst_len)
    audio[start:start+burst_len] += window * 3.0 * np.sin(2 * np.pi * 4000 * t[:burst_len])

    audio = audio / np.max(np.abs(audio))
    return audio.astype(np.float32)


def make_footsteps(sr, duration=2.0):
    n = int(sr * duration)
    audio = np.random.normal(0, 1e-4, size=n)

    # Simulate repetitive low-frequency thuds
    for k in range(3):
        pos = int((0.3 + 0.3 * k) * sr)
        pulse_len = int(0.02 * sr)
        audio[pos:pos+pulse_len] += np.hanning(pulse_len) * 0.8 * (1 - 0.2 * k)

    audio = audio / np.max(np.abs(audio))
    return audio.astype(np.float32)


def make_background(sr, duration=2.0):
    n = int(sr * duration)
    audio = np.random.normal(0, 0.02, size=n)
    audio = audio / np.max(np.abs(audio)) * 0.02
    return audio.astype(np.float32)


def run_tests():
    det = SoundDetector()
    sr = det.sample_rate

    tests = [
        ("background", make_background(sr)),
        ("footsteps", make_footsteps(sr)),
        ("glass_break", make_glass_break(sr)),
        ("gunshot", make_gunshot(sr)),
    ]

    for name, audio in tests:
        print('\n--- Test:', name)
        sound_type, confidence = det._analyze_sound(audio)
        print('Result:', sound_type, 'confidence:', confidence)

        # Extra debug info for gunshot to inspect spectral content
        if name == 'gunshot':
            import librosa
            sr = det.sample_rate
            energy = np.sum(np.square(audio)) / len(audio)
            # short RMS
            short_window = int(0.05 * sr)
            if short_window < 1:
                short_window = 1
            rms_short = np.sqrt(np.convolve(np.square(audio), np.ones(short_window) / short_window, mode='valid'))
            short_rms_max = float(np.max(rms_short))
            peak = float(np.max(np.abs(audio)))
            spectral = np.abs(librosa.stft(audio))
            frequencies = librosa.fft_frequencies(sr=sr)
            freq_bands = [(100, 500), (500, 2000), (2000, 8000)]
            band_energies = []
            for low, high in freq_bands:
                mask = (frequencies >= low) & (frequencies <= high)
                band_energies.append(np.mean(spectral[mask]) if np.any(mask) else 0.0)
                import librosa
                onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
                print('DEBUG gunshot: energy=', energy, 'peak=', peak, 'short_rms_max=', short_rms_max, 'band_energies=', band_energies, 'onset_max=', float(np.max(onset_env)))


if __name__ == '__main__':
    run_tests()
