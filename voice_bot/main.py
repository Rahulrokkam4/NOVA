from modules.t import STTTranscriber
from modules.vad_mic import VADMicService
# from modules.llm import LLMService
# from modules.tts import TTSService
import time
import queue
import threading

def main():
    # A thread-safe queue to pass the NumPy audio array from VAD to STT
    audio_queue = queue.Queue()

    # 1. Initialize Modules
    vad_streamer = VADMicService(out_q=audio_queue)
    stt_processor = STTTranscriber(input_queue=audio_queue)
    
    # 2. Start Threads
    # The VAD stream runs its callback in a background thread provided by sounddevice.
    # We will put the STT logic into its own thread to run concurrently.
    
    stt_thread = threading.Thread(target=stt_processor.transcribe_loop, daemon=True)
    stt_thread.start()
    
    # The VAD stream needs to be run in the main thread or its own thread 
    # to keep the context manager alive and handle Ctrl+C gracefully.
    try:
        vad_streamer.start_stream()
    except KeyboardInterrupt:
        print("\nStopping VAD Stream...")
    finally:
        # Wait for any remaining items in the queue to be processed
        audio_queue.join()
        print("Application Shut Down.")

if __name__ == "__main__":
    # Ensure you have saved the two modules as vad_mic_module.py and stt_transcriber_module.py
    main()




# # Main:-


# # The Main Class of Head Of The Class:-
# class Chatbot:
#     # Defined TTS and LLM Modules:-
#     def __init__(self):
#         self.llm = LLMService()
#         self.tts = TTSService()

#     # Here Start The Conversation Among User and AI:-
#     def start(self):
#         print("Chatbot Started Of TTS. Type 'exit' to quit.")

#         # User Input Here:-
#         while True:
#             user_query = input("You: ")

#             if user_query.lower() == "exit":
#                 break

#             # Based On User Input Model Given Response Here:-
#             ai_reply = self.llm.stream_response(user_query)
#             print("AI:", ai_reply,flush=True)

#             # Here is Voice Generated Via AI Reply with Used Elevenlabs Model:-
#             self.tts.speak(ai_reply)

# # Main Object Calling Funciton of Class:-
# if __name__ == "__main__":
#     bot = Chatbot()
#     bot.start()




# class chatbot:
#     def __init__(self): # 1. Initialize Modules
#         self.audio_queue = queue.Queue()
#         self.llm = LLMService()
#         self.tts = TTSService()
#         self.stt = STTTranscriber(input_queue=self.audio_queue)
#         self.vad = VADMicService(out_q=self.audio_queue)
        
#     def start(self):
#         print("Chatbot Started. Type 'exit' to quit.")
        