#!/usr/bin/env python3
"""
STS Voice Agent - FIXED TELEMETRY
==================================
Previous version had BROKEN timing:
- E2E TTFB was printing Unix timestamp, not duration
- STT showing 0ms (timer started after completion)
- Metrics were useless

This version has CORRECT timing:
- All durations = end_time - start_time
- Timestamps captured at the RIGHT moments
- Actually useful for optimization

Target latencies:
- End-of-turn silence: 400ms
- STT: <500ms  
- LLM TTFB: <200ms
- TTS TTFB: <200ms
- E2E TTFB: <900ms
"""

import os
import sys
import time
import threading
import wave
import re
from io import BytesIO
from typing import List, Dict, Optional, Generator, Callable
from dataclasses import dataclass
from datetime import datetime
from collections import deque
from dotenv import load_dotenv

# Audio
import numpy as np
import pyaudio
import torch

# APIs
from groq import Groq
from elevenlabs import VoiceSettings
from elevenlabs.client import ElevenLabs

# GUI
import tkinter as tk
from tkinter import scrolledtext, messagebox

# Playback
import pygame

load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))


# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    # Audio
    SAMPLE_RATE = 16000
    CHUNK_SIZE = 512
    
    # VAD
    VAD_THRESHOLD = 0.5
    
    # Turn detection - REDUCED from 1000ms
    END_OF_TURN_SILENCE_MS = 400
    MIN_SPEECH_MS = 200
    
    # STT
    STT_MODEL = "scribe_v1"
    
    # LLM
    LLM_MODEL = "llama-3.3-70b-versatile"
    LLM_TEMPERATURE = 0.7
    LLM_MAX_TOKENS = 120
    
    # TTS
    TTS_MODEL = "eleven_flash_v2_5"
    TTS_VOICE_ID = "pNInz6obpgDQGcFmaJgB"
    
    # Bot
    BOT_NAME = "Nova"
    SYSTEM_PROMPT = """You are Nova, a fast voice assistant.
Keep responses under 2 sentences. No markdown or formatting.
Use contractions. Get to the point immediately."""
    
    HISTORY_LENGTH = 8
    
    # Targets (for display)
    TARGET_STT_MS = 500
    TARGET_LLM_TTFB_MS = 200
    TARGET_TTS_TTFB_MS = 200
    TARGET_E2E_TTFB_MS = 900


# =============================================================================
# TIMING CONTEXT - THE FIX FOR BROKEN TELEMETRY
# =============================================================================

@dataclass
class TurnTiming:
    """
    Holds all timestamps for a single turn.
    
    CRITICAL: All times are captured with time.time() at the EXACT moment.
    Latencies are calculated as: end_time - start_time
    NOT by printing timestamps directly.
    """
    turn_id: int = 0
    
    # === TIMESTAMPS (captured at exact moments) ===
    turn_start: float = 0.0          # When VAD detects end of speech
    
    stt_start: float = 0.0           # When STT begins
    stt_end: float = 0.0             # When STT returns text
    
    llm_start: float = 0.0           # When LLM request sent
    llm_first_token: float = 0.0     # When first token received
    llm_end: float = 0.0             # When LLM stream completes
    
    tts_start: float = 0.0           # When TTS request sent (first sentence)
    tts_first_byte: float = 0.0      # When first audio byte received
    first_audio_played: float = 0.0  # When audio actually starts playing
    tts_end: float = 0.0             # When all TTS complete
    
    # === CALCULATED DURATIONS (computed, not stored timestamps) ===
    silence_wait_ms: float = 0.0     # How long we waited for silence
    
    # User's speech info
    user_text: str = ""
    bot_response: str = ""
    
    def calc_stt_latency(self) -> float:
        """STT duration = stt_end - stt_start"""
        if self.stt_end > 0 and self.stt_start > 0:
            return (self.stt_end - self.stt_start) * 1000
        return 0.0
    
    def calc_llm_ttfb(self) -> float:
        """LLM TTFB = first_token - llm_start"""
        if self.llm_first_token > 0 and self.llm_start > 0:
            return (self.llm_first_token - self.llm_start) * 1000
        return 0.0
    
    def calc_llm_total(self) -> float:
        """LLM total = llm_end - llm_start"""
        if self.llm_end > 0 and self.llm_start > 0:
            return (self.llm_end - self.llm_start) * 1000
        return 0.0
    
    def calc_tts_ttfb(self) -> float:
        """TTS TTFB = first_byte - tts_start"""
        if self.tts_first_byte > 0 and self.tts_start > 0:
            return (self.tts_first_byte - self.tts_start) * 1000
        return 0.0
    
    def calc_e2e_ttfb(self) -> float:
        """
        E2E TTFB = first_audio_played - turn_start
        This is THE KEY METRIC: how long user waits from stopping speech
        to hearing the first response audio.
        """
        if self.first_audio_played > 0 and self.turn_start > 0:
            return (self.first_audio_played - self.turn_start) * 1000
        return 0.0
    
    def calc_total_latency(self) -> float:
        """Total = tts_end - turn_start"""
        if self.tts_end > 0 and self.turn_start > 0:
            return (self.tts_end - self.turn_start) * 1000
        return 0.0


class MetricsLogger:
    """
    Proper metrics logging with CORRECT calculations.
    
    Key fix: We store timestamps and CALCULATE durations.
    We never print raw timestamps as if they were durations.
    """
    
    def __init__(self):
        self.turn_count = 0
        self.current: Optional[TurnTiming] = None
        self.session_start = time.time()
        
        # Rolling windows for averages
        self.stt_latencies = deque(maxlen=50)
        self.llm_ttfbs = deque(maxlen=50)
        self.tts_ttfbs = deque(maxlen=50)
        self.e2e_ttfbs = deque(maxlen=50)
        self.totals = deque(maxlen=50)
        
        # Log file
        os.makedirs("logs", exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = f"logs/sts_fixed_{ts}.log"
        
        # Callback for GUI
        self.on_log: Optional[Callable] = None
        
        self._log("=" * 60)
        self._log("STS VOICE AGENT - FIXED TELEMETRY")
        self._log("=" * 60)
        self._log(f"Targets: STT<{Config.TARGET_STT_MS}ms, LLM_TTFB<{Config.TARGET_LLM_TTFB_MS}ms, "
                  f"TTS_TTFB<{Config.TARGET_TTS_TTFB_MS}ms, E2E<{Config.TARGET_E2E_TTFB_MS}ms")
        self._log("=" * 60)
    
    def _log(self, msg: str, level: str = "INFO"):
        ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        line = f"[{ts}] {msg}"
        print(line)
        try:
            with open(self.log_file, "a") as f:
                f.write(line + "\n")
        except:
            pass
        if self.on_log:
            self.on_log(msg, level)
    
    # === Turn lifecycle ===
    
    def start_turn(self, silence_ms: float) -> TurnTiming:
        """
        Called when VAD detects end of speech.
        This is T=0 for the turn.
        """
        self.turn_count += 1
        self.current = TurnTiming(
            turn_id=self.turn_count,
            turn_start=time.time(),  # THIS IS T=0
            silence_wait_ms=silence_ms
        )
        self._log(f"━━━ TURN {self.turn_count} ━━━")
        self._log(f"🔇 Turn start (silence wait: {silence_ms:.0f}ms)")
        return self.current
    
    def log_stt_start(self):
        """Called when STT processing begins"""
        if self.current:
            self.current.stt_start = time.time()
            self._log(f"📝 STT started (stt_start={self.current.stt_start:.3f})")
        else:
            self._log("⚠️ log_stt_start called but self.current is None!")
    
    def log_stt_end(self, text: str):
        """Called when STT returns text"""
        if self.current:
            self.current.stt_end = time.time()
            self.current.user_text = text
            
            # Debug: print raw values
            self._log(f"   DEBUG: stt_start={self.current.stt_start:.3f}, stt_end={self.current.stt_end:.3f}")
            
            latency = self.current.calc_stt_latency()
            self.stt_latencies.append(latency)
            
            status = "✅" if latency <= Config.TARGET_STT_MS else "⚠️"
            preview = text[:40] + "..." if len(text) > 40 else text
            self._log(f"{status} STT: {latency:.0f}ms | \"{preview}\"")
        else:
            self._log("⚠️ log_stt_end called but self.current is None!")
    
    def log_llm_start(self):
        """Called when LLM request is sent"""
        if self.current:
            self.current.llm_start = time.time()
            self._log("🧠 LLM started")
    
    def log_llm_first_token(self):
        """Called when first token arrives from LLM stream"""
        if self.current and self.current.llm_first_token == 0:
            self.current.llm_first_token = time.time()
            
            ttfb = self.current.calc_llm_ttfb()
            self.llm_ttfbs.append(ttfb)
            
            status = "✅" if ttfb <= Config.TARGET_LLM_TTFB_MS else "⚠️"
            self._log(f"{status} LLM TTFB: {ttfb:.0f}ms")
    
    def log_llm_end(self, response: str):
        """Called when LLM stream completes"""
        if self.current:
            self.current.llm_end = time.time()
            self.current.bot_response = response
            
            total = self.current.calc_llm_total()
            self._log(f"✅ LLM complete: {total:.0f}ms")
    
    def log_tts_start(self):
        """Called when first TTS request is sent"""
        if self.current and self.current.tts_start == 0:
            self.current.tts_start = time.time()
            self._log("🗣️ TTS started")
    
    def log_tts_first_byte(self):
        """Called when first audio byte is received from TTS"""
        if self.current and self.current.tts_first_byte == 0:
            self.current.tts_first_byte = time.time()
            
            ttfb = self.current.calc_tts_ttfb()
            self.tts_ttfbs.append(ttfb)
            
            status = "✅" if ttfb <= Config.TARGET_TTS_TTFB_MS else "⚠️"
            self._log(f"{status} TTS TTFB: {ttfb:.0f}ms")
    
    def log_first_audio_played(self):
        """
        Called when audio ACTUALLY starts playing on speakers.
        This is the moment the user first hears the response.
        """
        if self.current and self.current.first_audio_played == 0:
            self.current.first_audio_played = time.time()
            
            e2e = self.current.calc_e2e_ttfb()
            self.e2e_ttfbs.append(e2e)
            
            status = "✅" if e2e <= Config.TARGET_E2E_TTFB_MS else "⚠️"
            self._log(f"{status} ★ E2E TTFB: {e2e:.0f}ms ★")
    
    def log_tts_end(self):
        """Called when all audio playback is complete"""
        if self.current:
            self.current.tts_end = time.time()
    
    def end_turn(self):
        """Finalize turn and print summary with CALCULATED values"""
        if not self.current:
            self._log("⚠️ end_turn called but self.current is None!")
            return
        
        t = self.current
        
        # Debug: print raw timestamps
        self._log(f"   DEBUG RAW: stt_start={t.stt_start}, stt_end={t.stt_end}")
        self._log(f"   DEBUG RAW: llm_start={t.llm_start}, llm_first_token={t.llm_first_token}")
        self._log(f"   DEBUG RAW: tts_start={t.tts_start}, first_audio_played={t.first_audio_played}")
        
        # Calculate all metrics
        stt = t.calc_stt_latency()
        llm_ttfb = t.calc_llm_ttfb()
        llm_total = t.calc_llm_total()
        tts_ttfb = t.calc_tts_ttfb()
        e2e = t.calc_e2e_ttfb()
        total = t.calc_total_latency()
        
        self.totals.append(total)
        
        # Print summary with CALCULATED values, not timestamps
        self._log("─" * 50)
        self._log(f"📊 TURN {t.turn_id} SUMMARY (all values are DURATIONS, not timestamps)")
        self._log(f"   Silence wait:   {t.silence_wait_ms:>6.0f}ms")
        self._log(f"   STT latency:    {stt:>6.0f}ms  (target: {Config.TARGET_STT_MS}ms)")
        self._log(f"   LLM TTFB:       {llm_ttfb:>6.0f}ms  (target: {Config.TARGET_LLM_TTFB_MS}ms)")
        self._log(f"   LLM total:      {llm_total:>6.0f}ms")
        self._log(f"   TTS TTFB:       {tts_ttfb:>6.0f}ms  (target: {Config.TARGET_TTS_TTFB_MS}ms)")
        self._log(f"   ─────────────────────────")
        self._log(f"   ★ E2E TTFB:     {e2e:>6.0f}ms  (target: {Config.TARGET_E2E_TTFB_MS}ms)")
        self._log(f"   Total:          {total:>6.0f}ms")
        self._log("─" * 50)
        
        # Sanity check - flag if something looks wrong
        if e2e > 10000:
            self._log("⚠️ WARNING: E2E TTFB > 10s - check timestamp capture points!")
        if stt == 0:
            self._log("⚠️ WARNING: STT = 0ms - timer may not be started correctly!")
            self._log(f"   stt_start={t.stt_start}, stt_end={t.stt_end}")
        
        self.current = None
    
    def get_stats(self) -> Dict:
        """Get session statistics with CALCULATED averages"""
        def avg(d): return sum(d) / len(d) if d else 0
        def p95(d):
            if not d: return 0
            s = sorted(d)
            idx = int(len(s) * 0.95)
            return s[min(idx, len(s) - 1)]
        
        return {
            "turns": self.turn_count,
            "avg_stt": avg(self.stt_latencies),
            "avg_llm_ttfb": avg(self.llm_ttfbs),
            "avg_tts_ttfb": avg(self.tts_ttfbs),
            "avg_e2e_ttfb": avg(self.e2e_ttfbs),
            "avg_total": avg(self.totals),
            "p95_e2e_ttfb": p95(list(self.e2e_ttfbs)),
            "p95_total": p95(list(self.totals)),
        }


# =============================================================================
# VAD
# =============================================================================

class VAD:
    def __init__(self):
        print("Loading VAD...")
        self.model, utils = torch.hub.load(
            'snakers4/silero-vad', 'silero_vad',
            force_reload=False, onnx=False, verbose=False
        )
        self.get_speech_ts = utils[0]
        print("✅ VAD ready")
        self.reset()
    
    def reset(self):
        self.model.reset_states()
    
    def is_speech(self, audio: bytes) -> bool:
        arr = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0
        tensor = torch.from_numpy(arr).unsqueeze(0)
        prob = self.model(tensor, Config.SAMPLE_RATE).item()
        return prob > Config.VAD_THRESHOLD


# =============================================================================
# TURN DETECTOR
# =============================================================================

class TurnDetector:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.has_speech = False
        self.last_speech_time = 0
        self.silence_start_time = 0
    
    def update(self, is_speech: bool) -> tuple[bool, float]:
        """
        Returns (is_turn_complete, silence_duration_ms)
        """
        now = time.time()
        
        if is_speech:
            self.has_speech = True
            self.last_speech_time = now
            self.silence_start_time = 0
            return False, 0
        
        if self.has_speech:
            if self.silence_start_time == 0:
                self.silence_start_time = now
            
            silence_ms = (now - self.silence_start_time) * 1000
            
            if silence_ms >= Config.END_OF_TURN_SILENCE_MS:
                return True, silence_ms
        
        return False, 0


# =============================================================================
# AUDIO BUFFER
# =============================================================================

class AudioBuffer:
    def __init__(self):
        self.chunks = []
    
    def add(self, chunk: bytes):
        self.chunks.append(chunk)
    
    def clear(self):
        self.chunks = []
    
    def to_wav(self) -> BytesIO:
        if not self.chunks:
            return BytesIO()
        data = b''.join(self.chunks)
        buf = BytesIO()
        with wave.open(buf, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(Config.SAMPLE_RATE)
            wf.writeframes(data)
        buf.seek(0)
        return buf
    
    @property
    def duration_ms(self) -> float:
        total = sum(len(c) for c in self.chunks)
        return (total / 2 / Config.SAMPLE_RATE) * 1000


# =============================================================================
# SENTENCE CHUNKER
# =============================================================================

class SentenceChunker:
    """Buffers LLM tokens and emits complete sentences"""
    
    ENDINGS = re.compile(r'[.!?]+\s*')
    
    def __init__(self):
        self.buffer = ""
    
    def add(self, token: str) -> Optional[str]:
        self.buffer += token
        match = self.ENDINGS.search(self.buffer)
        if match:
            end = match.end()
            sentence = self.buffer[:end].strip()
            self.buffer = self.buffer[end:]
            if len(sentence) > 5:  # Min length
                return sentence
        return None
    
    def flush(self) -> Optional[str]:
        if self.buffer.strip():
            result = self.buffer.strip()
            self.buffer = ""
            return result
        return None
    
    def reset(self):
        self.buffer = ""


# =============================================================================
# AI CLIENTS
# =============================================================================

class STTClient:
    def __init__(self):
        self.client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))
    
    def transcribe(self, wav_data: BytesIO) -> str:
        if wav_data.getbuffer().nbytes == 0:
            return ""
        try:
            result = self.client.speech_to_text.convert(
                file=wav_data,
                model_id=Config.STT_MODEL
            )
            return result.text if result else ""
        except Exception as e:
            print(f"STT error: {e}")
            return ""


class LLMClient:
    def __init__(self):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.history = []
    
    def stream(self, user_text: str) -> Generator[str, None, None]:
        self.history.append({"role": "user", "content": user_text})
        
        messages = [
            {"role": "system", "content": Config.SYSTEM_PROMPT}
        ] + self.history[-Config.HISTORY_LENGTH:]
        
        try:
            stream = self.client.chat.completions.create(
                model=Config.LLM_MODEL,
                messages=messages,
                temperature=Config.LLM_TEMPERATURE,
                max_tokens=Config.LLM_MAX_TOKENS,
                stream=True
            )
            
            full = ""
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    token = chunk.choices[0].delta.content
                    full += token
                    yield token
            
            self.history.append({"role": "assistant", "content": full})
            if len(self.history) > Config.HISTORY_LENGTH * 2:
                self.history = self.history[-Config.HISTORY_LENGTH * 2:]
                
        except Exception as e:
            print(f"LLM error: {e}")
            yield "Sorry, I had an error."


class TTSClient:
    def __init__(self):
        self.client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))
    
    def synthesize(self, text: str) -> Generator[bytes, None, None]:
        """Stream TTS audio chunks"""
        try:
            stream = self.client.text_to_speech.convert(
                voice_id=Config.TTS_VOICE_ID,
                model_id=Config.TTS_MODEL,
                text=text,
                output_format="mp3_22050_32",
                voice_settings=VoiceSettings(
                    stability=0.5,
                    similarity_boost=0.75,
                )
            )
            for chunk in stream:
                if chunk:
                    yield chunk
        except Exception as e:
            print(f"TTS error: {e}")


class AudioPlayer:
    def __init__(self):
        pygame.mixer.init()
        self._playing = False
        self._stop = False
    
    def play(self, audio: bytes, on_play_start: Callable = None):
        """Play audio. on_play_start called when audio actually starts."""
        self._playing = True
        self._stop = False
        
        try:
            buf = BytesIO(audio)
            pygame.mixer.music.load(buf)
            pygame.mixer.music.play()
            
            # Audio is now playing - call callback
            if on_play_start:
                on_play_start()
            
            while pygame.mixer.music.get_busy() and not self._stop:
                pygame.time.wait(10)
        except Exception as e:
            print(f"Play error: {e}")
        
        self._playing = False
    
    def stop(self):
        self._stop = True
        try:
            pygame.mixer.music.stop()
        except:
            pass
        self._playing = False
    
    @property
    def is_playing(self) -> bool:
        return pygame.mixer.music.get_busy()


# =============================================================================
# PIPELINE
# =============================================================================

class Pipeline:
    def __init__(self, metrics: MetricsLogger):
        self.metrics = metrics
        
        self.vad = VAD()
        self.turn_detector = TurnDetector()
        self.audio_buffer = AudioBuffer()
        self.stt = STTClient()
        self.llm = LLMClient()
        self.tts = TTSClient()
        self.player = AudioPlayer()
        self.chunker = SentenceChunker()
        
        self._cancelled = False
        print("✅ Pipeline ready")
    
    def process_chunk(self, chunk: bytes) -> Optional[tuple[str, float]]:
        """
        Process audio chunk.
        Returns (transcript, silence_ms) if turn complete, None otherwise.
        """
        # Barge-in check
        if self.player.is_playing:
            if self.vad.is_speech(chunk):
                self._cancel()
            return None
        
        is_speech = self.vad.is_speech(chunk)
        
        if is_speech or self.turn_detector.has_speech:
            self.audio_buffer.add(chunk)
        
        turn_done, silence_ms = self.turn_detector.update(is_speech)
        
        if turn_done:
            if self.audio_buffer.duration_ms < Config.MIN_SPEECH_MS:
                self._reset()
                return None
            
            # === START TURN TIMING HERE (before STT) ===
            self.metrics.start_turn(silence_ms)
            
            # Get audio and transcribe
            wav = self.audio_buffer.to_wav()
            self._reset()
            
            # STT with timing - now self.metrics.current exists!
            self.metrics.log_stt_start()
            text = self.stt.transcribe(wav)
            self.metrics.log_stt_end(text)
            
            if text.strip():
                return text, silence_ms
            else:
                # No valid text, cancel this turn
                self.metrics.current = None
        
        return None
    
    def respond(self, user_text: str):
        """Generate and play response with sentence streaming"""
        self._cancelled = False
        self.chunker.reset()
        
        full_response = ""
        first_sentence = True
        first_tts_byte_logged = False
        first_audio_logged = False
        
        # Start LLM
        self.metrics.log_llm_start()
        first_token_logged = False
        
        for token in self.llm.stream(user_text):
            if self._cancelled:
                break
            
            # Log first token
            if not first_token_logged:
                self.metrics.log_llm_first_token()
                first_token_logged = True
            
            full_response += token
            
            # Check for complete sentence
            sentence = self.chunker.add(token)
            if sentence and not self._cancelled:
                self._speak_sentence(
                    sentence, 
                    first_sentence,
                    first_tts_byte_logged,
                    first_audio_logged,
                    lambda: setattr(self, '_first_tts_logged', True),
                    lambda: setattr(self, '_first_audio_logged', True)
                )
                first_sentence = False
                first_tts_byte_logged = getattr(self, '_first_tts_logged', False)
                first_audio_logged = getattr(self, '_first_audio_logged', False)
        
        # Log LLM end
        self.metrics.log_llm_end(full_response)
        
        # Flush remaining
        if not self._cancelled:
            remaining = self.chunker.flush()
            if remaining:
                self._speak_sentence(
                    remaining,
                    first_sentence,
                    first_tts_byte_logged,
                    first_audio_logged,
                    lambda: None,
                    lambda: None
                )
        
        self.metrics.log_tts_end()
        return full_response
    
    def _speak_sentence(self, text: str, is_first: bool, 
                        tts_logged: bool, audio_logged: bool,
                        on_tts_first: Callable, on_audio_first: Callable):
        """Synthesize and play a single sentence"""
        if self._cancelled:
            return
        
        # Log TTS start for first sentence
        if is_first:
            self.metrics.log_tts_start()
        
        # Collect TTS audio
        audio_chunks = []
        first_byte = True
        
        for chunk in self.tts.synthesize(text):
            if self._cancelled:
                return
            
            # Log first byte
            if first_byte and not tts_logged:
                self.metrics.log_tts_first_byte()
                on_tts_first()
                first_byte = False
            
            audio_chunks.append(chunk)
        
        if self._cancelled or not audio_chunks:
            return
        
        # Play audio
        audio_data = b''.join(audio_chunks)
        
        def on_play():
            if not audio_logged:
                self.metrics.log_first_audio_played()
                on_audio_first()
        
        self.player.play(audio_data, on_play_start=on_play)
    
    def _cancel(self):
        self._cancelled = True
        self.player.stop()
        self.metrics._log("⚡ BARGE-IN")
        self._reset()
    
    def _reset(self):
        self.audio_buffer.clear()
        self.turn_detector.reset()
        self.vad.reset()
        self.chunker.reset()


# =============================================================================
# GUI
# =============================================================================

class GUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title(f"🎤 {Config.BOT_NAME} - Fixed Telemetry")
        self.root.geometry("800x650")
        self.root.configure(bg="#0d1117")
        
        self.running = True
        self.listening = True
        self.metrics: Optional[MetricsLogger] = None
        self.pipeline: Optional[Pipeline] = None
        
        self._build()
        self._start()
    
    def _build(self):
        # Title
        tk.Label(
            self.root, text=f"🎤 {Config.BOT_NAME}",
            font=("Arial", 20, "bold"), bg="#0d1117", fg="#58a6ff"
        ).pack(pady=10)
        
        # Status
        self.status = tk.StringVar(value="Starting...")
        tk.Label(
            self.root, textvariable=self.status,
            font=("Arial", 12, "bold"), bg="#0d1117", fg="#f0883e"
        ).pack()
        
        # Metrics
        mf = tk.Frame(self.root, bg="#161b22")
        mf.pack(fill=tk.X, padx=20, pady=10)
        
        self.metric_vars = {}
        for label, key in [("E2E TTFB", "e2e"), ("STT", "stt"), 
                           ("LLM TTFB", "llm"), ("TTS TTFB", "tts")]:
            f = tk.Frame(mf, bg="#161b22")
            f.pack(side=tk.LEFT, expand=True, padx=15, pady=8)
            tk.Label(f, text=label, font=("Arial", 9), bg="#161b22", fg="#8b949e").pack()
            var = tk.StringVar(value="--")
            tk.Label(f, textvariable=var, font=("Courier", 14, "bold"), 
                    bg="#161b22", fg="#7ee787").pack()
            self.metric_vars[key] = var
        
        # Chat
        cf = tk.Frame(self.root, bg="#0d1117")
        cf.pack(fill=tk.BOTH, expand=True, padx=20, pady=5)
        
        self.chat = scrolledtext.ScrolledText(
            cf, wrap=tk.WORD, bg="#0d1117", fg="#c9d1d9",
            font=("Arial", 11), borderwidth=0
        )
        self.chat.pack(fill=tk.BOTH, expand=True)
        self.chat.tag_config('user', foreground="#58a6ff")
        self.chat.tag_config('bot', foreground="#7ee787")
        
        # Log
        tk.Label(self.root, text="📋 Log", font=("Arial", 10, "bold"),
                bg="#0d1117", fg="#8b949e").pack(anchor=tk.W, padx=20)
        
        lf = tk.Frame(self.root, bg="#161b22")
        lf.pack(fill=tk.X, padx=20, pady=5)
        
        self.log = scrolledtext.ScrolledText(
            lf, wrap=tk.WORD, height=8, bg="#0d1117", fg="#7ee787",
            font=("Courier", 9), borderwidth=0
        )
        self.log.pack(fill=tk.X)
        
        # Controls
        ctrl = tk.Frame(self.root, bg="#0d1117")
        ctrl.pack(fill=tk.X, padx=20, pady=10)
        
        self.mute_btn = tk.Button(
            ctrl, text="🔇 Mute", command=self._mute,
            bg="#21262d", fg="white", relief=tk.FLAT, padx=15, pady=6
        )
        self.mute_btn.pack(side=tk.LEFT)
        
        tk.Button(
            ctrl, text="📊 Stats", command=self._stats,
            bg="#21262d", fg="white", relief=tk.FLAT, padx=15, pady=6
        ).pack(side=tk.LEFT, padx=10)
        
        tk.Button(
            ctrl, text="🛑 Quit", command=self._quit,
            bg="#da3633", fg="white", relief=tk.FLAT, padx=15, pady=6
        ).pack(side=tk.RIGHT)
    
    def _start(self):
        def init():
            self.metrics = MetricsLogger()
            self.metrics.on_log = self._log_cb
            self.pipeline = Pipeline(self.metrics)
            self.root.after(0, lambda: self.status.set("🎤 Listening..."))
            
            self.thread = threading.Thread(target=self._loop, daemon=True)
            self.thread.start()
        
        threading.Thread(target=init, daemon=True).start()
    
    def _loop(self):
        p = pyaudio.PyAudio()
        stream = p.open(
            format=pyaudio.paInt16, channels=1,
            rate=Config.SAMPLE_RATE, input=True,
            frames_per_buffer=Config.CHUNK_SIZE
        )
        
        while self.running:
            if not self.listening or not self.pipeline:
                time.sleep(0.05)
                continue
            
            try:
                chunk = stream.read(Config.CHUNK_SIZE, exception_on_overflow=False)
            except:
                continue
            
            result = self.pipeline.process_chunk(chunk)
            
            if result:
                text, silence_ms = result
                self._handle(text, silence_ms)
        
        stream.stop_stream()
        stream.close()
        p.terminate()
    
    def _handle(self, text: str, silence_ms: float):
        self.root.after(0, lambda: self._msg(text, "user"))
        self.root.after(0, lambda: self.status.set("🧠 Processing..."))
        
        # Turn timing already started in process_chunk() - STT is already recorded
        # Use self.pipeline.metrics to ensure we're using the same object
        
        def respond():
            self.root.after(0, lambda: self.status.set("🗣️ Speaking..."))
            
            response = self.pipeline.respond(text)
            
            self.root.after(0, lambda: self._msg(response, "bot"))
            
            # Use pipeline.metrics to be explicit
            self.pipeline.metrics.end_turn()
            self._update_metrics()
            
            if self.running:
                self.root.after(0, lambda: self.status.set("🎤 Listening..."))
        
        threading.Thread(target=respond, daemon=True).start()
    
    def _msg(self, text: str, who: str):
        prefix = "👤 You" if who == "user" else f"🤖 {Config.BOT_NAME}"
        self.chat.insert(tk.END, f"{prefix}: {text}\n\n", who)
        self.chat.see(tk.END)
    
    def _log_cb(self, msg: str, level: str):
        def update():
            self.log.insert(tk.END, f"{msg}\n")
            self.log.see(tk.END)
            lines = int(self.log.index('end-1c').split('.')[0])
            if lines > 100:
                self.log.delete('1.0', '50.0')
        self.root.after(0, update)
    
    def _update_metrics(self):
        def update():
            s = self.pipeline.metrics.get_stats()
            self.metric_vars["e2e"].set(f"{s['avg_e2e_ttfb']:.0f}ms")
            self.metric_vars["stt"].set(f"{s['avg_stt']:.0f}ms")
            self.metric_vars["llm"].set(f"{s['avg_llm_ttfb']:.0f}ms")
            self.metric_vars["tts"].set(f"{s['avg_tts_ttfb']:.0f}ms")
        self.root.after(0, update)
    
    def _mute(self):
        self.listening = not self.listening
        self.mute_btn.config(text="🔈 Unmute" if not self.listening else "🔇 Mute")
        self.status.set("🔇 Muted" if not self.listening else "🎤 Listening...")
    
    def _stats(self):
        if not self.pipeline:
            return
        s = self.pipeline.metrics.get_stats()
        messagebox.showinfo("Stats", f"""
Turns: {s['turns']}

AVERAGES:
  E2E TTFB: {s['avg_e2e_ttfb']:.0f}ms (target: {Config.TARGET_E2E_TTFB_MS}ms)
  STT: {s['avg_stt']:.0f}ms
  LLM TTFB: {s['avg_llm_ttfb']:.0f}ms
  TTS TTFB: {s['avg_tts_ttfb']:.0f}ms

P95:
  E2E TTFB: {s['p95_e2e_ttfb']:.0f}ms
  Total: {s['p95_total']:.0f}ms
        """)
    
    def _quit(self):
        self.running = False
        self.root.destroy()


# =============================================================================
# MAIN
# =============================================================================

def main():
    if not os.getenv("ELEVENLABS_API_KEY"):
        print("❌ ELEVENLABS_API_KEY missing")
        sys.exit(1)
    if not os.getenv("GROQ_API_KEY"):
        print("❌ GROQ_API_KEY missing")
        sys.exit(1)
    
    print("=" * 60)
    print("STS VOICE AGENT - FIXED TELEMETRY")
    print("=" * 60)
    print("All latencies are now CALCULATED correctly:")
    print("  duration = end_time - start_time")
    print("  (not raw timestamps)")
    print("=" * 60)
    
    root = tk.Tk()
    app = GUI(root)
    root.protocol("WM_DELETE_WINDOW", app._quit)
    root.mainloop()


if __name__ == "__main__":
    main()