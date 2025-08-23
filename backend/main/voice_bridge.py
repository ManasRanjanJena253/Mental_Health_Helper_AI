import os
from dataclasses import dataclass
from typing import Optional, Tuple, List
import warnings
import whisper
from dotenv import load_dotenv
import playsound
from google.cloud import speech
from elevenlabs import ElevenLabs
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
    elevenlabs_voice_id: str = "EXAVITQu4vr4xnSDxMaL"  # Replace with desired voice ID
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
        self, text: str, output_path: Optional[str] = None, voice_id: Optional[str] = None, model_id: Optional[str] = None
    ) -> str:
        if not text.strip():
            raise ValueError("Cannot synthesize empty text.")

        voice_id = voice_id or self.config.elevenlabs_voice_id
        model_id = model_id or self.config.elevenlabs_model_id
        fmt = self.config.default_tts_format.lower()

        if output_path is None:
            output_path = self._auto_filename(fmt)

        audio = self.eleven_client.text_to_speech.convert(
            voice_id = voice_id,
            model_id = model_id,
            text = text
        )

        with open(output_path, "wb") as f:
            f.write(audio)

        return output_path

    def google_text_to_speech(self, file_path, output_path = "output.mp3", lang = "en"):
        tts = gTTS(text=text, lang=lang)
        tts.save(output_path)
        return output_path

    @staticmethod
    def _infer_google_encoding(path: str):
        from google.cloud.speech import RecognitionConfig
        ext = os.path.splitext(path)[1].lower()
        mapping = {
            ".wav": RecognitionConfig.AudioEncoding.LINEAR16,
            ".flac": RecognitionConfig.AudioEncoding.FLAC,
            ".mp3": RecognitionConfig.AudioEncoding.MP3,
            ".ogg": RecognitionConfig.AudioEncoding.OGG_OPUS,
            ".opus": RecognitionConfig.AudioEncoding.OGG_OPUS,
            ".webm": RecognitionConfig.AudioEncoding.WEBM_OPUS,
        }
        return mapping.get(ext, RecognitionConfig.AudioEncoding.ENCODING_UNSPECIFIED)

    @staticmethod
    def _auto_filename(fmt: str) -> str:
        from datetime import datetime
        timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        return f"mindhaven_tts_{timestamp}.{fmt}"


if __name__ == "__main__":
    # Load config
    config = VoiceConfig(
        language_code = "en-IN",
        elevenlabs_voice_id = "EXAVITQu4vr4xnSDxMaL"  # Replace with your chosen ElevenLabs voice
    )

    voice = MindHavenVoice(config)

    # 1. Speech to Text
    text = voice.speech_to_text(audio_file = "C:/Users/mranj/PycharmProjects/Mental_Health_AI/backend/main/11Labs_testing.m4a")
    print(f"User said: {text}")

    # 2. Text to Speech
    audio_file = voice.google_text_to_speech("Hello, welcome to MindHaven.", output_path = "reply.mp3")
    print(f"Saved reply audio to: {audio_file}")