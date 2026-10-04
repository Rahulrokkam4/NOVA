## CAPTURING SPEECH AND CONVERT TO TEXT PIPELINE

1. User speaks → VAD detects speech start
2. User stops → VAD detects silence (500ms default)
3. Complete audio chunk sent to STT
4. Transcription returned
5. System immediately ready for next input


# Silero_vad (state-of-the-art accuracy)
> it is one of the most accurate open-source VAD models
> Automatically detects speech start/end



1.ffmpeg
