import queue, time, threading, numpy as np, traceback as tb
from faster_whisper import WhisperModel
from typing import Optional, Callable


class STTServive:
    def __init__(self, 
                 in_q:queue.Queue, 
                 out_q:queue.Queue,
                 model_size: str = "base", 
                 device:str="cuda",
                 compute_type: Optional[str] = "float16",
                 language: Optional[str] = "en",
                 sample_rate: int = 16000,
                 beam_size: int=3,
                 max_q_wait: float=0.5
            ):
        self.in_q = in_q
        self.out_q = out_q
        self.language = language
        self.sample_rate = sample_rate
        self.beam_size = beam_size
        self.max_queue_wait = max_q_wait
        self.device = device.lower()
        
        self._stop = threading.Event()
        self._cancel = threading.Event()
        self._thread = None
        
        # Load Faster-Whisper model
        try:
            self.model = WhisperModel(
                model_size,
                device=device,
                compute_type=compute_type
            )
        except Exception as e:
            print("Failed to load faster-whisper model:", e)
            tb.print_exc()
            self.model = None
            
        # metrics
        self.count = 0
        self.total_audio_seconds = 0.0
        self.total_processing_seconds = 0.0
        
    # -- Control --
    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self.transcribe_loop, daemon=True)
        self._thread.start()
        
    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        print("STTService stopped")
        
    def cancel(self):
        self._cancel.set()    
        
    def clear_cancel(self):
        self._cancel.clear()
        
    
    # -- helpers --
    def process_audio(self, item):
        ts = time.time()
        sr = self.sample_rate
        audio = None
        if isinstance(item, dict):
            audio = item.get("audio")
            sr = int(item.get("sample_rate", sr))
            ts = float(item.get("ts", ts))
        else:
            audio = item
        
        if audio is None:
            return None, sr, ts
        
        if audio.dtype == np.int16:
            float_audio = audio.astype(np.float32) / 32768.0
        elif audio.dtype == np.float32 or audio.dtype == np.float64:
            float_audio = audio.astype(np.float32)
            if float_audio.max() > 1000.0:
                float_audio = float_audio / 32768.0
        else:
            float_audio = audio.astype(np.float32)
            mx = np.max(np.abs(float_audio)) if float_audio.size else 1.0
            if mx > 0:
                if mx > 2.0:  # probably int16-like
                    float_audio = float_audio / 32768.0
                else:
                    float_audio = float_audio / mx
                    
        return float_audio, sr, ts
            
        
    # -- Main loop --            
    def transcribe_loop(self): # func is used for pulls audio from queue and transcribes.
        while not self._stop.is_set():
            try:
                item = self.in_q.get(timeout=self.max_queue_wait)
            except queue.Empty:
                continue
            if item is None:
                continue
            
            audio, sr, ts = self.process_audio(item)
            if audio is None or audio.size == 0:
                continue
            
            duration = len(audio) / float(sr)
            # optional: ignore extremely short segments
            if duration < 0.05:
                continue
            
            # update metrics
            self.count += 1
            self.total_audio_seconds += duration

            # reset cancel flag before starting
            if self._cancel.is_set():
                self.clear_cancel()
            
            # reset cancel flag before starting
            if self._cancel.is_set():
                self.clear_cancel()

            # Transcribe
            start_proc = time.perf_counter()
            text = ""
            try:
                if self.model is not None:
                    # faster-whisper expects float32 numpy array for single-utt transcribe
                    segments, info = self.model.transcribe(audio, beam_size=self.beam_size, language=self.language)
                    parts = []
                    for seg in segments:
                        parts.append(seg.text)
                        # allow best-effort cancel between segments
                        if self._cancel.is_set():
                            break
                    text = " ".join(parts).strip()
                else:
                    # No model available
                    text = "<no-stt-model>"
            except Exception as e:
                print("STT error:", e)
                tb.print_exc()
                text = ""
                
            proc_time = time.perf_counter() - start_proc
            self.total_processing_seconds += proc_time
            
            # push transcript (non-blocking)
            out = {"text": text, "ts": ts, "duration": duration, "proc_time": proc_time}
            try:
                self.out_q.put_nowait(out)
            except queue.Full:
                # drop and move on
                pass   
            
        # end loop - optional cleanup/log
        # print basic metrics
        if self.count:
            print(f"STTService processed {self.count} segments, audio_seconds={self.total_audio_seconds:.2f}, total_proc={self.total_processing_seconds:.2f}")
            
    def transcribe_once(self, audio_np: np.ndarray, sr: int = 16000) -> str:
        audio = audio_np
        if audio.dtype == np.int16:
            audio_f = audio.astype(np.float32) / 32768.0
        else:
            audio_f = audio.astype(np.float32)
            if audio_f.max() > 1000.0:
                audio_f = audio_f / 32768.0
        if self.model is None:
            return "<no-stt-model>"
        segments, info = self.model.transcribe(audio_f, beam_size=self.beam_size, language=self.language)
        return " ".join([s.text for s in segments]).strip()
            
 
 
 
from vad_mic import VADMicService
from llm import LLMService
def print_transcripts_loop(out_q: queue.Queue):
    """Consume and print transcripts from STT service."""
    llm = LLMService()
    while True:
        try:
            item = out_q.get(timeout=1.0)
        except queue.Empty:
            continue
        if item is None:
            continue
        text = item.get("text", "")
        ts = item.get("ts", None)
        if text:
            reply_text = llm.stream_response(text)
        print(f"[STT @ {time.strftime('%H:%M:%S', time.localtime(ts or time.time()))}] {text}")
        print(f"[LLM ANSWER @ {time.strftime('%H:%M:%S', time.localtime(ts or time.time()))}] {reply_text}")

def main():
    q = queue.Queue(maxsize=8)
    t = queue.Queue(maxsize=8)
    vad = VADMicService(q)
    stt = STTServive(q,t)
    
    stt.start()
    tprinter = threading.Thread(target=print_transcripts_loop, args=(t,), daemon=True)
    tprinter.start()
    vad.start()
    print("Listening... (Ctrl+C to stop)")  

    try:
        while True:
            # main loop handles preemption: if new utterance comes in while stt is busy, cancel
            # We detect queue size to decide whether to cancel
            if q.qsize() > 1:
                # user spoke again quickly: cancel current transcription
                print("Interrupt detected: user spoke again — signalling cancel to STT")
                stt.cancel()
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("Shutting down")
    finally:
        vad.stop()
        stt.stop()
         
if __name__ == "__main__":
    main()
          
             