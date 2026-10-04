# tts_service.py
import threading
import time
from elevenlabs.client import ElevenLabs
from elevenlabs.play import play
from config import Elevenlabs



class TTSService:
    
    def __init__(self):
        self.client = ElevenLabs(api_key=Elevenlabs.ELEVENLABS_API_KEY)
        self._playing = False
        self._lock = threading.Lock()
        self._stop_flag = threading.Event()
        self._thread = None

    def is_playing(self):
        with self._lock:
            return self._playing

    def stop(self):
        """Immediately stop TTS playback."""
        with self._lock:
            if not self._playing:
                return
        self._stop_flag.set()  # will interrupt stream reading
        # Give thread tiny time to exit
        time.sleep(0.05)

    def speak(self, text: str, on_done=None):
        """
        Speak text asynchronously.
        Automatically supports interruption via stop().
        """
        if not text:
            print("No text received for TTS.")
            return

        # Stop any existing playback
        self.stop()
        self._stop_flag.clear()

        def _run():
            with self._lock:
                self._playing = True

            try:
                # Create a streaming generator
                stream = self.client.text_to_speech.stream(
                    voice_id=Elevenlabs.ELEVENLABS_VOICE_ID,
                    model_id=Elevenlabs.ELEVENLABS_TTS_MODEL,
                    text=text
                )

                # stream audio chunk by chunk
                for chunk in stream:
                    if self._stop_flag.is_set():
                        break
                # if not self._stop_flag.is_set():
                    play(stream)  # ▶ plays audio incrementally

            except Exception as e:
                print("TTS error:", e)

            finally:
                with self._lock:
                    self._playing = False
                if on_done:
                    try:
                        on_done()
                    except:
                        pass

        self._thread = threading.Thread(target=_run, daemon=False)
        self._thread.start()
        return self._thread
    
    
if __name__ == "__main__":
    t = TTSService()
    t.speak("hii rahul")