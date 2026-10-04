import time
import queue
import torch
import numpy as np
import traceback as tb
import sounddevice as sd
from collections import deque
from config import VAD


class VADMicService:
    def __init__(self, out_q, vad_threshold=VAD.VAD_THRESHOLD):
        self.out_q = out_q
        self.vad_threshold = vad_threshold
        self.paused = False # NEW
        
        # Load silero model via torch.hub
        try:
            self.model, self.utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False
            )
            self.model.eval()
        except Exception as e:
            print(f"Failed to load silero vad model : {e}")
        
        torch.set_num_threads(1)
        
        # State variables
        self.speech_buffer = []
        self.is_speech_active = False
        self.silence_frame_count = 0
        
        # Pre-speech circular buffer (captures audio BEFORE speech detected)
        pre_speech_frame = int(VAD.PRE_SPEECH_PADDING_MS / VAD.FRAME_MS)
        self.pre_speech_buffer = deque(maxlen=max(1,pre_speech_frame))
        
        # Calculated limits
        self.max_silence_frames = int(VAD.SILENCE_LIMIT_SEC / (VAD.FRAME_MS / 1000))
        self.min_speech_frames = int(VAD.MIN_SPEECH_SEC / (VAD.FRAME_MS / 1000))
        self.max_speech_frames = int(VAD.MAX_UTTERANCE_SEC / (VAD.FRAME_MS / 1000))    
        
        # Audio stream handle
        self.stream = None
    
    # --Pause, Start, Stop function for control--   
    def set_pause(self, state: bool):
        """Pause or Resume the microphone listener."""
        state = bool(state)
        if state == self.paused:
            return
        self.paused = state
        if state:
            self.reset_state() # Clear buffers to avoid sending older audio after resume
            self.pre_speech_buffer.clear()
            self.dropped_frames = 0
            print("Mic PAUSED")
        else:
            print("Mic LISTENING")
            
    def start(self, device=None): # func for Clear buffers to avoid sending older audio after resume
        if self.stream is not None:
            return
        self.stream =sd.InputStream(
                samplerate=VAD.SAMPLE_RATE,
                blocksize=VAD.FRAME_SIZE,
                dtype='float32',
                channels=1,
                callback=self.callback,
                device=device
            )
        self.stream.start()
        
    def stop(self): # func for Stop and close input stream and emit remaining buffer safely.
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
            self.stream = None
        try:
            self.emit_utterance()
        except Exception:
            pass
  
              
    def callback(self, indata, frames, time_info, status): # func is called by sounddevice stream for each audio block Runs in a separate audio thread - must be fast and non-blocking.
        try:    
            if self.paused:# ✅ NEW:
                self.dropped_frames += 1
                return
            
            if status:
                # print(f"Stream Status: {status}")
                pass

            # prepare frame cast only mono float32
            frame = indata[:, 0].astype(np.float32)
            
            # VAD Inference
            with torch.no_grad():# Disable gradient computation for inference
                frame_tensor = torch.from_numpy(frame)
                speech_prob = float(self.model(frame_tensor, VAD.SAMPLE_RATE).item())
            
            # state machine logic   
            is_speech = speech_prob > self.vad_threshold
            
            if is_speech: # Speech detected
                if not self.is_speech_active:
                    self.is_speech_active = True
                    # print(f"Speech Start (prob: {speech_prob:.2f})")
                    
                    if self.pre_speech_buffer: # Add buffered frames from BEFORE speech started
                        self.speech_buffer.extend(list(self.pre_speech_buffer))
                self.silence_frame_count = 0 # Reset silence counter
                self.speech_buffer.append(frame) # Add current speech frame
                
                if len(self.speech_buffer) >= self.max_speech_frames: # Safety check - prevent infinite utterances
                    # print(f"Max utterance length reached ({MAX_UTTERANCE_SEC}s)")
                    self.emit_utterance()
            else: # Speech not detected
                self.pre_speech_buffer.append(frame) # Always buffer silence (for pre-speech padding)
                
                if self.is_speech_active:
                    self.silence_frame_count += 1 # We're in speech mode but current frame is silence
                    self.speech_buffer.append(frame)
                    
                    if self.silence_frame_count >= self.max_silence_frames: # Check if silence limit reached
                        self.emit_utterance()
        except Exception as e:
            print(f"Error in VAD callback: {e}")
            tb.print_exc()

    # -- Utterance Logic --        
    def reset_state(self):# func for Reset all state variables for next utterance.
        self.speech_buffer = []
        self.is_speech_active = False
        self.silence_frame_count = 0 

    def emit_utterance(self): # func for Process and emit complete utterance(validation, trimming, and safe queue insertion).
        if not self.speech_buffer:
            self.reset_state()
            return
        if len(self.speech_buffer) < self.min_speech_frames:
            # print(f"Utterance too short ({len(self.speech_buffer)} frames), discarding")
            self.reset_state() # Reset State
            return
        trimmed_buffer = self.trim_silence(self.speech_buffer) # Smart Trimming
        
        if not trimmed_buffer:
            self.reset_state() # Reset State
            return
        
        clean_audio = np.concatenate(trimmed_buffer) # Concatenate and prepare audio
        ts = time.time()
        #Send to queue (non-blocking)
        try: # Using put_nowait to avoid blocking audio thread.
            self.out_q.put_nowait({"audio": clean_audio, "sample_rate": VAD.SAMPLE_RATE, "time_stamp":ts})
            # print(f"Utterance sent: {duration:.2f}s ({len(trimmed_buffer)} frames)")
        except queue.Full:
            print("Output queue full - dropping utterance")   
        self.reset_state() # Reset State
        
    def trim_silence(self, buffer): # func for Smart silence trimming using energy-based detection.
        if not buffer:
            return []
        audio = np.concatenate(buffer, axis=0) # Convert to numpy array for analysis
        
        # Calculate energy per frame
        frame_energies = []
        for i in range(0, len(audio), VAD.FRAME_SIZE):
            frame = audio[i:i + VAD.FRAME_SIZE]
            if frame.size == 0:
                continue
            energy = float(np.sqrt(np.mean(frame ** 2))) # RMS enery
            frame_energies.append(energy)
            
        if not frame_energies:
            return buffer
        
        threshold = max(np.percentile(frame_energies, 5), 0.01) # Find threshold (5th percentile = noise floor)
        speech_frame = [i for i, e in enumerate(frame_energies) if e > threshold] # Find speech boundaries (frames above threshold)
        
        if not speech_frame:
            return buffer # No frames above threshold - keep original buffer
        
        start_frame = max(0, speech_frame[0] - 1) # Keep 1 frame before
        end_frame = min(len(buffer), speech_frame[-1] + 2) # Keep 1 frame after
        return buffer[start_frame:end_frame]
            


if __name__ == "__main__":
    import sys, json
    q = queue.Queue()
    v = VADMicService(q)
    print("Starting VAD mic. Speak into your mic.")
    v.start()
    try:
        while True:
            item = q.get()
            audio = item["audio"]
            print("UTTERANCE: duration=", len(audio)/VAD.SAMPLE_RATE, "sec at", item["time_stamp"])
    except KeyboardInterrupt:
        print("Stopping")
    finally:
        v.stop()
        sys.exit(0)
        