import time
from sound_detector import SoundDetector


def run_live_monitor(poll_interval=0.5):
    det = SoundDetector()
    print("Starting live sound monitor. Speak/play test sounds near the mic.")
    try:
        while True:
            events = det.start_monitoring()
            if events:
                for ev in events:
                    ts = ev.get('timestamp')
                    st = ev.get('type')
                    conf = ev.get('confidence')
                    print(f"[{ts}] Detected: {st} (confidence: {conf:.2f})")
            else:
                print("No event")
            time.sleep(poll_interval)
    except KeyboardInterrupt:
        print("Stopping live monitor")


if __name__ == '__main__':
    run_live_monitor()
