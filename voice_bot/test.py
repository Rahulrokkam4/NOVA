# orchestrator.py
"""
Orchestrator for on-device AI voice bot.
Wires: VADMicService -> STTService -> (LLM) -> TTSService
Behavior:
 - Keeps audio device open. Uses vad.set_pause(True/False) during TTS (no stream restart).
 - Preemption: user speaking while TTS plays stops TTS and prioritizes user.
 - If new user input arrives while LLM generating, skip speaking previous reply in favor of newest user input.
 - Shutdown after silence_timeout seconds of no user activity.
"""

import queue
import threading
import time
import sys
import traceback

# Import your modules (adjust names if different)
from modules.vad_mic import VADMicService
from modules.stt import STTSerive
from modules.tts import TTSService
from modules.llm import LLMService


class VoiceOrchestrator:
    def __init__(
        self,
        silence_timeout: float = 10.0,
        audio_q_size: int = 6,
        transcript_q_size: int = 8,
        stt_model_size: str = "base",
        stt_device: str = "cpu",
        stt_compute_type: str | None = None,
        stt_language: str | None = "en"
    ):
        # Queues
        self.audio_q = queue.Queue(maxsize=audio_q_size)       # VAD -> STT
        self.transcript_q = queue.Queue(maxsize=transcript_q_size)  # STT -> Orchestrator

        # Components
        self.vad = VADMicService(self.audio_q)  # must accept out_q on init
        self.stt = STTSerive(
            in_q=self.audio_q,
            out_q=self.transcript_q,
            model_size=stt_model_size,
            device=stt_device,
            compute_type=stt_compute_type,
            language=stt_language
        )
        self.tts = TTSService()  # must implement speak(text, on_done=None), stop(), is_playing()
        self.llm = LLMService()
        
        # Orchestration state
        self.silence_timeout = float(silence_timeout)
        self.last_user_activity = time.time()
        self._shutdown = threading.Event()
        self._processing = threading.Event()  # true while handling LLM+TTS for a transcript
        self._watcher = threading.Thread(target=self._silence_watcher, daemon=True)

        # bookkeeping for debugging
        self._main_thread = None

    # ---------- lifecycle ----------
    def start(self):
        """Start components and main loop (non-blocking)."""
        # Start audio stream (keeps device open)
        self.vad.start()
        # Start STT worker thread
        self.stt.start()
        # Start silence watcher
        self._watcher.start()
        # Start main processing loop
        self._main_thread = threading.Thread(target=self._main_loop, daemon=True)
        self._main_thread.start()
        print("VoiceOrchestrator started")

    def stop(self):
        """Graceful shutdown: stop components and threads."""
        if self._shutdown.is_set():
            return
        print("VoiceOrchestrator stopping...")
        self._shutdown.set()
        try:
            # Stop TTS first (interrupt playback)
            if self.tts and getattr(self.tts, "is_playing", None):
                try:
                    self.tts.stop()
                except Exception:
                    pass
        except Exception:
            pass

        # Stop components
        try:
            self.vad.stop()   # safe to stop at shutdown
        except Exception:
            pass
        try:
            self.stt.stop()
        except Exception:
            pass

        print("VoiceOrchestrator stopped")

    # ---------- watcher ----------
    def _silence_watcher(self):
        """
        Shuts down after silence_timeout of user inactivity, but ignores timeout
        while the system is busy processing or TTS is playing.
        """
        while not self._shutdown.is_set():
            # If currently processing or TTS playing, treat as active — don't shutdown.
            processing = self._processing.is_set()
            tts_playing = False
            try:
                if self.tts and getattr(self.tts, "is_playing", None):
                    tts_playing = self.tts.is_playing()
            except Exception:
                tts_playing = False

            if not processing and not tts_playing:
                # Only enforce timeout when idle
                idle = time.time() - self.last_user_activity
                if idle > self.silence_timeout:
                    print(f"No user activity for {self.silence_timeout}s — shutting down.")
                    try:
                        self.stop()
                    except Exception:
                        pass
                    return
            # otherwise, we're busy — reset watcher sleep and continue
            time.sleep(0.5)


    # ---------- main processing ----------
    def _main_loop(self):
        """
        Consume transcripts from STT (self.transcript_q), call LLM, and play TTS.
        Handles preemption and skipping stale replies if new user speech arrives.
        """
        while not self._shutdown.is_set():
            # Wait for a transcript; short timeout so we can check shutdown flag
            try:
                item = self.transcript_q.get(timeout=0.2)
            except queue.Empty:
                continue

            try:
                text = (item.get("text") if isinstance(item, dict) else str(item)).strip()
            except Exception:
                text = ""

            if not text:
                # ignore empty transcripts
                continue

            # Mark user activity
            self.last_user_activity = time.time()
            print(f"[USER @ {time.strftime('%H:%M:%S')}] {text}")

            # If TTS is playing, user interrupted — stop TTS and cancel any in-flight STT work
            if hasattr(self.tts, "is_playing") and self.tts.is_playing():
                print("User interrupted bot while speaking -> stopping TTS and prioritizing user.")
                try:
                    self.tts.stop()
                except Exception:
                    pass
                # best-effort cancel of any in-progress transcription
                try:
                    self.stt.cancel()
                except Exception:
                    pass

            # Set processing flag
            self._processing.set()
            self.last_user_activity = time.time()
            
            # Call LLM synchronously (if your LLM is async, call from thread)
            try:
                # Run LLM in a thread to avoid blocking entire main loop if your LLM is slow,
                # but here we keep it simple and synchronous. For very slow LLMs, wrap in thread.
                reply_text = self.llm.stream_response(user_query=text)
            except Exception as e:
                print("LLM error:", e)
                traceback.print_exc()
                reply_text = "Sorry, I failed to generate a reply."

            # If another user utterance arrived while generating reply, skip speaking this reply
            if not self.transcript_q.empty():
                print("New user input arrived while generating reply -> skipping speaking previous reply.")
                self._processing.clear()
                # continue to next loop to handle fresh user input
                continue

            # Pause VAD processing (do NOT close stream) to avoid TTS re-triggering VAD.
            try:
                self.vad.set_pause(True)
                # self.last_user_activity = time.time()  # keep alive while TTS plays
                # self.tts.speak(reply_text, on_done=_on_tts_done)
            except Exception:
                print("Warning: failed to pause vad (continuing)")

            # Prepare event to wait for TTS completion or preemption
            tts_done = threading.Event()
            def _on_tts_done():
                tts_done.set()

            # Start TTS (non-blocking)
            try:
                # NOTE: TTSService.speak or speak_text signature must accept on_done callback.
                # If your real tts_service uses .speak(text) only, adapt to not block:
                # - Either wrap .speak in a background thread here, or update TTSService to support on_done.
                self.tts.speak(reply_text, on_done=_on_tts_done)
            except TypeError:
                # fallback: .speak(text) without callback, run in thread to wait for completion
                def _play_blocking():
                    try:
                        self.tts.speak(reply_text)
                    except Exception:
                        pass
                    finally:
                        _on_tts_done()
                threading.Thread(target=_play_blocking, daemon=True).start()
            except Exception as e:
                print("TTS error:", e)
                _on_tts_done()

            # Wait for tts to finish unless preempted by new user speech
            while not tts_done.is_set() and not self._shutdown.is_set():
                # If new transcript(s) present, user is speaking -> preempt
                if not self.transcript_q.empty():
                    print("Detected user speech during TTS -> preempting TTS.")
                    try:
                        self.tts.stop()
                    except Exception:
                        pass
                    # Cancel STT in case it's running
                    try:
                        self.stt.cancel()
                    except Exception:
                        pass
                    break
                time.sleep(0.04)

            # Resume VAD processing (keep stream open)
            try:
                self.vad.set_pause(False)
            except Exception:
                pass

            # Reset processing flag and update last activity
            self._processing.clear()
            self.last_user_activity = time.time()

        # exiting main loop -> ensure shutdown
        try:
            self.stop()
        except Exception:
            pass

# ---------- run as script ----------
if __name__ == "__main__":
    orch = VoiceOrchestrator(silence_timeout=10.0,
                             stt_model_size="base",
                             stt_device="cuda",
                             stt_compute_type="float16",
                             stt_language="en")
    orch.start()

    try:
        while not orch._shutdown.is_set():
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("Keyboard interrupt: shutting down")
        orch.stop()
        sys.exit(0)
