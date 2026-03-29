"""
voice_bridge.py
~~~~~~~~~~~~~~~
STT (Whisper) + TTS (ElevenLabs / gTTS) bridge for MindHaven.

whisper.load_model() is called ONCE in __init__, not on every
speech_to_text() call. The old code was loading the model from disk on
every request — extremely expensive.

Usage: instantiate MindHavenVoice once (ideally as a module-level singleton
or inside the Redis-backed cache) and reuse across requests.
"""

import io
import os
import warnings
from typing import Iterable, Iterator

import whisper
from dotenv import load_dotenv
from gtts import gTTS

warnings.filterwarnings(action="ignore")
load_dotenv()


class MindHavenVoice:
    """
    Singleton-friendly voice service.

    STT:  OpenAI Whisper (local, runs on CPU)
    TTS:  ElevenLabs (premium) with gTTS as a free fallback
    """

    # Whisper model size — "small" is the best speed/accuracy trade-off for
    # a mental-health app where clarity matters more than real-time latency.
    # Options: tiny | base | small | medium | large
    WHISPER_MODEL_SIZE: str = os.getenv("WHISPER_MODEL", "small")

    def __init__(self, use_elevenlabs: bool = True):
        # Load Whisper once — this is the critical fix.
        # The old code called whisper.load_model() inside speech_to_text()
        # meaning it hit disk on EVERY request. Moving it here means the
        # model is loaded once when the class is instantiated and then
        # reused across all subsequent calls.

        print(f"[MindHavenVoice] Loading Whisper '{self.WHISPER_MODEL_SIZE}'...")
        self._whisper = whisper.load_model(self.WHISPER_MODEL_SIZE)
        print("[MindHavenVoice] Whisper ready.")

        # ElevenLabs (optional — gracefully disabled if key is absent)
        self._eleven = None
        if use_elevenlabs:
            api_key = os.getenv("ELEVENLABS_API_KEY")
            if api_key:
                from elevenlabs import ElevenLabs
                self._eleven = ElevenLabs(api_key=api_key)
            else:
                print("[MindHavenVoice] ELEVENLABS_API_KEY not set — falling back to gTTS.")


    # STT
    def speech_to_text(self, audio_file: str) -> str:
        """
        Transcribe an audio file to text using the preloaded Whisper model.

        :param audio_file: Path to the audio file (wav, mp3, m4a, etc.)
        :return: Transcribed text string.
        """
        result = self._whisper.transcribe(audio=audio_file)
        return result["text"].strip()

    # TTS — ElevenLabs (streaming)
    def text_to_speech_stream(
        self,
        text: str,
        voice_id: str = "m28sDRnudtExG3WLAufB",
        model_id: str = "eleven_flash_v2_5",
    ) -> Iterator[bytes]:
        """
        Stream TTS audio via ElevenLabs.
        Raises RuntimeError if ElevenLabs is not configured.
        Falls back gracefully — callers should prefer gtts_stream() when
        ElevenLabs is unavailable.
        """
        if not self._eleven:
            raise RuntimeError(
                "ElevenLabs is not configured. Use gtts_stream() as a fallback."
            )
        if not text.strip():
            raise ValueError("Cannot synthesise empty text.")

        from elevenlabs import VoiceSettings

        audio = self._eleven.text_to_speech.convert(
            voice_id=voice_id,
            model_id=model_id,
            text=text,
            voice_settings=VoiceSettings(
                stability=0.7,
                speed=0.87,
                similarity_boost=0.6,
            ),
        )

        yield from self._normalise_audio_iterable(audio)

    # TTS — gTTS (free fallback, streaming)

    def gtts_stream(
        self, text: str, lang: str = "en", chunk_size: int = 4096
    ) -> Iterator[bytes]:
        """
        Generate TTS audio using gTTS and stream it in chunks.
        Completely free, no API key required. Slightly robotic but reliable.

        :param text: Text to convert to speech.
        :param lang: Language code.
        :param chunk_size: Bytes per chunk yielded.
        """
        if not text.strip():
            raise ValueError("Cannot synthesise empty text.")

        buf = io.BytesIO()
        gTTS(text=text, lang=lang).write_to_fp(buf)
        buf.seek(0)

        while True:
            chunk = buf.read(chunk_size)
            if not chunk:
                break
            yield chunk


    # Smart TTS — tries ElevenLabs, falls back to gTTS

    def tts_stream(self, text: str) -> Iterator[bytes]:
        """
        Preferred entry point for TTS in the API layer.
        Uses ElevenLabs if available, otherwise gTTS.
        """
        if self._eleven:
            try:
                yield from self.text_to_speech_stream(text)
                return
            except Exception as e:
                print(f"[MindHavenVoice] ElevenLabs failed ({e}), falling back to gTTS.")
        yield from self.gtts_stream(text)


    # Internal
    @staticmethod
    def _normalise_audio_iterable(audio) -> Iterator[bytes]:
        """Normalise whatever the ElevenLabs SDK returns into a bytes iterator."""
        chunk_size = 8192

        if isinstance(audio, (bytes, bytearray, memoryview)):
            data = bytes(audio)
            for i in range(0, len(data), chunk_size):
                yield data[i: i + chunk_size]

        elif hasattr(audio, "read"):
            while True:
                chunk = audio.read(chunk_size)
                if not chunk:
                    break
                yield chunk if isinstance(chunk, bytes) else chunk.encode()

        elif isinstance(audio, Iterable):
            for chunk in audio:
                if chunk is None:
                    continue
                if isinstance(chunk, str):
                    chunk = chunk.encode()
                elif isinstance(chunk, memoryview):
                    chunk = bytes(chunk)
                yield chunk

        else:
            try:
                data = bytes(audio)
                for i in range(0, len(data), chunk_size):
                    yield data[i: i + chunk_size]
            except Exception as e:
                raise TypeError(
                    f"Unsupported audio type from ElevenLabs SDK: {type(audio)}. Error: {e}"
                )