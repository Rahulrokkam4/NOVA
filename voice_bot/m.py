from modules.stt import STTSerive
from modules.vad_mic import VADMicService
from modules.llm import LLMService
from modules.tts import TTSService
import time, queue, threading, sys


def run_conversation_pipeline():
    mic_data_queue = queue.Queue()
    # Init Services
    vad = VADMicService(out_q=mic_data_queue)
    # Note: We instantiate STT but we won't run its loop. We call it manually.
    stt = STTSerive(input_queue=None) 
    llm = LLMService()
    tts = TTSService()
    
    # Start VAD Stream in Background
    vad_thread = threading.Thread(target=vad.start_stream, daemon=True)
    vad_thread.start()
    
    print("\n Conversation Loop Started. Waiting for voice...")
    while True:
        try:
            # -------------------------------------------------------
            # 1. LISTENING PHASE
            # -------------------------------------------------------
            # Ensure VAD is listening
            vad.set_pause(False) 
            
            # Block and wait for audio (raw utterance) from VAD
            # This waits until the user finishes a sentence.
            audio_data = mic_data_queue.get() 
            
            print("🎤 Processing Audio...")

            # -------------------------------------------------------
            # 2. PROCESSING PHASE (MUTE VAD)
            # -------------------------------------------------------
            # Immediately pause VAD so we don't record the bot thinking or speaking
            vad.set_pause(True) 
            
            # --- STT Step ---
            # Manually process the audio chunk we just got
            # Note: We access the internal processing method of your STT class
            # You might need to make `_process_audio` return the text directly
            # or use the STT class as a library function here.
            
            # Let's assume a helper function based on your STT code:
            audio_float = audio_data.astype(float) / 32768.0
            user_text = stt(audio_float)
            # # Run transcription directly (not via queue) for synchronous control
            # segments, _ = stt.model.transcribe(audio_float, language="en")
            
            # # Collect text
            # user_text = " ".join([seg.text for seg in segments]).strip()
            
            # print(f"👤 User: {user_text}")
            

            if not user_text:
                continue

            # -------------------------------------------------------
            # 3. EXIT CHECK
            # -------------------------------------------------------
            triggers = ["bye", "goodbye", "exit", "quit"]
            if any(x in user_text.lower() for x in triggers):
                print("👋 User requested exit. Goodbye!")
                tts.speak("Goodbye! Have a nice day.")
                break

            # -------------------------------------------------------
            # 4. LLM GENERATION
            # -------------------------------------------------------
            bot_response = llm.stream_response(user_text)
            print(f"🤖 Bot: {bot_response}")

            # -------------------------------------------------------
            # 5. TTS SPEAKING
            # -------------------------------------------------------
            # VAD is still PAUSED here, so the bot won't hear itself.
            tts.speak(bot_response)
            
            # -------------------------------------------------------
            # 6. RESET FOR NEXT TURN
            # -------------------------------------------------------
            # Loop goes back to top -> calls vad.set_pause(False)
            # This "unmutes" the mic only after TTS is fully done.
            
        except KeyboardInterrupt:
            print("\n🛑 Stopping Bot...")
            sys.exit(0)
            break
        except Exception as e:
            print(f"Error in loop: {e}")
            # Ensure we resume listening even on error
            vad.set_pause(False)

if __name__ == "__main__":
    run_conversation_pipeline()
    
        