import os
from typing import Optional, Tuple, List, Iterable
import warnings
import whisper
from dotenv import load_dotenv
from elevenlabs import ElevenLabs, VoiceSettings
from gtts import gTTS
import io

warnings.filterwarnings(action = "ignore")

load_dotenv()

# Eleven labs speech at 0.85 speed


class MindHavenVoice:
    def __init__(self):

        # ElevenLabs client
        if not os.getenv("ELEVENLABS_API_KEY"):
            raise EnvironmentError("ELEVENLABS_API_KEY is not set in environment variables.")
        self.eleven_client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))

    def speech_to_text(self, audio_file):
        # Loading the Whisper model (small = fast, large = accurate)
        model = whisper.load_model("small")
        result = model.transcribe(audio = audio_file)
        return result["text"]

    def text_to_speech_stream(
            self, text: str, voice_id = "m28sDRnudtExG3WLAufB" , model_id = "eleven_flash_v2_5"
    ):
        if not text.strip():
            raise ValueError("Cannot synthesize empty text.")

        # Call the ElevenLabs SDK
        audio = self.eleven_client.text_to_speech.convert(
            voice_id=voice_id,
            model_id=model_id,
            text=text,
            voice_settings=VoiceSettings(
                stability=0.7,
                speed=0.87,
                similarity_boost=0.6,
            )
        )

        # Handle different audio response types
        if isinstance(audio, (bytes, bytearray, memoryview)):
            # If it's a single bytes object, yield it in chunks for consistent streaming
            chunk_size = 8192  # 8KB chunks
            data = bytes(audio)
            for i in range(0, len(data), chunk_size):
                yield data[i:i + chunk_size]

        elif hasattr(audio, "read"):
            # File-like object - read in chunks
            while True:
                chunk = audio.read(8192)
                if not chunk:
                    break
                # Ensure bytes
                if isinstance(chunk, str):
                    chunk = chunk.encode()
                yield chunk

        elif isinstance(audio, Iterable):
            # Iterable/generator of chunks (most streaming clients)
            for chunk in audio:
                if chunk is None:
                    continue
                if isinstance(chunk, str):
                    chunk = chunk.encode()
                elif isinstance(chunk, memoryview):
                    chunk = bytes(chunk)
                yield chunk

        else:
            # Fallback: attempt to convert to bytes and stream in chunks
            try:
                data = bytes(audio)
                chunk_size = 8192
                for i in range(0, len(data), chunk_size):
                    yield data[i:i + chunk_size]
            except Exception as e:
                raise TypeError(f"Unsupported audio type returned from TTS client: {type(audio)}. Error: {e}")

    def gtts_stream(self, text: str, lang: str = "en", chunk_size: int = 1024):
        """
        Generate TTS audio using gTTS and stream it in chunks.

        Args:
            text (str): The text to convert to speech.
            lang (str): Language code (default "en").
            chunk_size (int): Number of bytes per chunk.

        Yields:
            bytes: Chunks of MP3 data.
        """
        if not text.strip():
            raise ValueError("Cannot synthesize empty text.")

        # Generate speech into memory buffer
        buffer = io.BytesIO()
        tts = gTTS(text=text, lang=lang)
        tts.write_to_fp(buffer)

        # Rewind buffer to start
        buffer.seek(0)

        # Yield in small chunks
        while True:
            chunk = buffer.read(chunk_size)
            if not chunk:
                break
            yield chunk

    @staticmethod
    def _auto_filename(fmt: str) -> str:
        from datetime import datetime
        timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        return f"mindhaven_tts_{timestamp}.{fmt}"


if __name__ == "__main__":

    voice = MindHavenVoice()

    # 1. Speech to Text
    text = voice.speech_to_text(audio_file = "C:/Users/mranj/PycharmProjects/Mental_Health_AI/backend/main/11Labs_testing.m4a")
    print(f"User said: {text}")

    # # 2. Text to Speech
    # audio_file = voice.text_to_speech("No, need to worry you are safe here. You can tell me everything.", output_path = "reply.mp3")
    # print(f"Saved reply audio to: {audio_file}")