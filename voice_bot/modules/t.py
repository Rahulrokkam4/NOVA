import queue
import numpy as np
from faster_whisper import WhisperModel

class STTTranscriber:
    def __init__(self, input_queue):
        self.input_queue = input_queue
        # Load a Faster-Whisper model (e.g., tiny, base, small, medium). 
        # 'tiny' is fast but less accurate. 'small' or 'base' is a good balance.
        # Run on CPU for this real-time pipeline.
        print("⏳ Loading Faster-Whisper model...")
        self.model = WhisperModel("base", device="cuda", compute_type="float16")
        print("✅ Model loaded.")

    def transcribe_loop(self):
        """Pulls finished audio segments from the queue and transcribes them."""
        print("🧠 STT Transcriber is ready to process utterances...")
        while True:
            try:
                # Block and wait for a cleaned audio segment from the VAD module
                audio_segment = self.input_queue.get(timeout=1)
                
                # Check if the segment is empty (can happen if VAD resets too fast)
                if audio_segment.size == 0:
                    print("Skipping empty audio segment.")
                    continue

                # 1. Transcription (Faster-Whisper expects a NumPy array)
                segments, info = self.model.transcribe(
                    audio_segment, 
                    beam_size=5, 
                    language="en" # Specify the language
                )
                
                # 2. Process and Output Result
                full_transcript = " ".join(segment.text for segment in segments)
                
                # print("\n" + "="*50)
                # print(f"👂 Transcribed Utterance (Language: {info.language}):")
                # print(f"📝 {full_transcript.strip()}")
                # print("="*50 + "\n")
                
                self.input_queue.task_done() 
                return full_transcript.strip(),info.language
                # Signal that the item has been processed (important for queue management)
                

            except queue.Empty:
                # Timeout, just continue waiting for the next segment
                continue
            except KeyboardInterrupt:
                break