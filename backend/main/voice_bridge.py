import os
from dataclasses import dataclass
from typing import Optional, Tuple, List, Iterable
import warnings
import whisper
from dotenv import load_dotenv
import playsound
from google.cloud import speech
from elevenlabs import ElevenLabs, VoiceSettings
from gtts import gTTS

warnings.filterwarnings(action = "ignore")

load_dotenv()

# Eleven labs speech at 0.85 speed

@dataclass
class VoiceConfig:
    # Google STT
    language_code: str = "en-US"
    sample_rate_hz: Optional[int] = None
    enable_automatic_punctuation: bool = True
    profanity_filter: bool = False
    alternative_language_codes: Optional[List[str]] = None

    # ElevenLabs TTS
    elevenlabs_voice_id: str = "m28sDRnudtExG3WLAufB"  # Replace with desired voice ID currently the voice id of Alekhya is being used for indian english.
    elevenlabs_model_id: str = "eleven_multilingual_v2"
    default_tts_format: str = "mp3"  # mp3 or wav


class MindHavenVoice:
    def __init__(self, config: Optional[VoiceConfig] = None):
        self.config = config or VoiceConfig()

        # ElevenLabs client
        if not os.getenv("ELEVENLABS_API_KEY"):
            raise EnvironmentError("ELEVENLABS_API_KEY is not set in environment variables.")
        self.eleven_client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))

    def speech_to_text(self, audio_file):
        # Loading the Whisper model (small = fast, large = accurate)
        model = whisper.load_model("small")
        result = model.transcribe(audio = audio_file)
        return result["text"]

    def text_to_speech(
            self, text: str, output_path: Optional[str] = None, voice_id: Optional[str] = None,
            model_id: Optional[str] = None
    ):
        if not text.strip():
            raise ValueError("Cannot synthesize empty text.")

        voice_id = voice_id or self.config.elevenlabs_voice_id
        model_id = model_id or self.config.elevenlabs_model_id
        fmt = self.config.default_tts_format.lower()

        if output_path is None:
            output_path = self._auto_filename(fmt)

        # Call the ElevenLabs SDK (may return bytes, generator, file-like, etc.)
        audio = self.eleven_client.text_to_speech.convert(
            voice_id = voice_id,
            model_id = model_id,
            text = text,
            voice_settings = VoiceSettings(
                stability = 0.7,
                speed = 0.87,
                similarity_boost = 0.6,
            )
        )

        # Defensive write: support bytes, file-like, and generators/iterables
        with open(output_path, "wb") as f:
            # If it's bytes-like, write directly
            if isinstance(audio, (bytes, bytearray, memoryview)):
                f.write(bytes(audio))

            # If it has a .read() method (file-like), read then write
            elif hasattr(audio, "read"):
                chunk = audio.read()
                # some file-like return str; ensure bytes
                if isinstance(chunk, str):
                    chunk = chunk.encode()
                f.write(chunk)

            # If it's an iterable/generator of chunks (most streaming clients)
            elif isinstance(audio, Iterable):
                for chunk in audio:
                    if chunk is None:
                        continue
                    if isinstance(chunk, str):
                        chunk = chunk.encode()
                    elif isinstance(chunk, memoryview):
                        chunk = bytes(chunk)
                    f.write(chunk)

            else:
                # Fallback: attempt to convert to bytes
                try:
                    b = bytes(audio)
                    f.write(b)
                except Exception as e:
                    raise TypeError(f"Unsupported audio type returned from TTS client: {type(audio)}. Error: {e}")

        return output_path

    @staticmethod
    def _auto_filename(fmt: str) -> str:
        from datetime import datetime
        timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        return f"mindhaven_tts_{timestamp}.{fmt}"


if __name__ == "__main__":
    # Load config
    config = VoiceConfig(
        language_code = "en-IN",
        elevenlabs_voice_id = "m28sDRnudtExG3WLAufB"  # Replace with your chosen ElevenLabs voice
    )

    voice = MindHavenVoice(config)

    # 1. Speech to Text
    text = voice.speech_to_text(audio_file = "C:/Users/mranj/PycharmProjects/Mental_Health_AI/backend/main/11Labs_testing.m4a")
    print(f"User said: {text}")

    # 2. Text to Speech
    audio_file = voice.text_to_speech("No, need to worry you are safe here. You can tell me everything.", output_path = "reply.mp3")
    print(f"Saved reply audio to: {audio_file}")