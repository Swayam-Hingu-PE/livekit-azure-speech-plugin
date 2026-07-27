from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

from livekit.agents import llm

if TYPE_CHECKING:
    from .session import AzureSpeechTranslationSession

logger = logging.getLogger(__name__)

# Azure's TranslationRecognizer synthesizes spoken translation at 16 kHz / 16-bit
# / mono PCM and ignores set_speech_synthesis_output_format, so this is the true
# rate of the frames we emit. RealtimeTranslator reads it via model.sample_rate
# (mislabeling it as 24 kHz plays the voice ~1.5x fast / high-pitched).
OUTPUT_SAMPLE_RATE = 16_000

# Sample rate the recognizer's push stream is fed with. Incoming LiveKit frames
# (native track rate, usually 48 kHz) are resampled to this in the session.
INPUT_SAMPLE_RATE = 16_000

# Generation close: seconds of no synthesized audio before a turn is finalized
# (overridable via the `llm` config).
DEFAULT_SYNTH_FLUSH_DELAY = 0.5


# ---------------------------------------------------------------------------
# Language / voice mapping
# ---------------------------------------------------------------------------
# Participant attributes / metadata carry SHORT language codes ("en", "hi").
# Azure needs a recognition LOCALE ("en-US") for the source and a neural voice
# name for the spoken target. These maps bridge that; unknown codes fall back
# gracefully so a missing entry degrades instead of crashing.

_AZURE_SOURCE_LOCALE: dict[str, str] = {
    "en": "en-US",
    "hi": "hi-IN",
    "es": "es-ES",
    "fr": "fr-FR",
    "de": "de-DE",
    "it": "it-IT",
    "pt": "pt-BR",
    "nl": "nl-NL",
    "ru": "ru-RU",
    "ja": "ja-JP",
    "ko": "ko-KR",
    "zh": "zh-CN",
    "ar": "ar-SA",
    "tr": "tr-TR",
    "pl": "pl-PL",
    "id": "id-ID",
    "ta": "ta-IN",
    "te": "te-IN",
    "mr": "mr-IN",
    "gu": "gu-IN",
    "bn": "bn-IN",
    "kn": "kn-IN",
    "ml": "ml-IN",
}

# Default neural voice per target language. Azure only produces spoken output
# when a voice is set, so we always need one; a per-participant "voice" attr
# overrides this via update_options().
_AZURE_DEFAULT_VOICE: dict[str, str] = {
    "en": "en-US-AvaNeural",
    "hi": "hi-IN-SwaraNeural",
    "es": "es-ES-ElviraNeural",
    "fr": "fr-FR-DeniseNeural",
    "de": "de-DE-KatjaNeural",
    "it": "it-IT-ElsaNeural",
    "pt": "pt-BR-FranciscaNeural",
    "nl": "nl-NL-ColetteNeural",
    "ru": "ru-RU-SvetlanaNeural",
    "ja": "ja-JP-NanamiNeural",
    "ko": "ko-KR-SunHiNeural",
    "zh": "zh-CN-XiaoxiaoNeural",
    "ar": "ar-SA-ZariyahNeural",
    "tr": "tr-TR-EmelNeural",
    "pl": "pl-PL-ZofiaNeural",
    "id": "id-ID-GadisNeural",
    "ta": "ta-IN-PallaviNeural",
    "te": "te-IN-ShrutiNeural",
    "mr": "mr-IN-AarohiNeural",
    "gu": "gu-IN-DhwaniNeural",
    "bn": "bn-IN-TanishaaNeural",
    "kn": "kn-IN-SapnaNeural",
    "ml": "ml-IN-SobhanaNeural",
}


def to_source_locale(code: str) -> str:
    """Short language code ("en") -> Azure recognition locale ("en-US")."""
    if not code:
        return "en-US"
    if "-" in code:  # already a locale
        return code
    return _AZURE_SOURCE_LOCALE.get(code.lower(), code)


def to_target_code(code: str) -> str:
    """Short target code passed to add_target_language (Azure uses the short form)."""
    if not code:
        return "en"
    return code.split("-", 1)[0].lower()


def default_voice_for(code: str) -> str | None:
    """Best-effort default neural voice for a target language code."""
    if not code:
        return None
    return _AZURE_DEFAULT_VOICE.get(to_target_code(code))


class AzureSpeechTranslationModel(llm.RealtimeModel):
    """
    RealtimeModel backed by Azure's speech-to-speech translation cascade.

    Mirrors the OpenAI translate provider's contract (segment-based translation,
    audio + translated-text output) but runs on the Azure Cognitive Services
    Speech SDK instead of a websocket. See `AzureSpeechTranslationSession`.
    """

    def __init__(
        self,
        *,
        speech_key: str | None = None,
        region: str | None = None,
        endpoint: str | None = None,
        source_language: str = "en",
        target_language: str = "en",
        voice_name: str | None = None,
        synth_flush_delay: float | None = None,
    ) -> None:
        super().__init__(
            capabilities=llm.RealtimeCapabilities(
                message_truncation=False,
                turn_detection=True,        # Azure emits speech start/stop events
                user_transcription=True,    # final source transcript via `recognized`
                auto_tool_reply_generation=False,
                audio_output=True,
                manual_function_calls=False,
                mutable_chat_context=False,
                mutable_instructions=False,
                mutable_tools=False,
            )
        )

        self._speech_key = speech_key or os.environ.get("SPEECH_KEY")
        self._region = region or os.environ.get("SPEECH_REGION")
        self._endpoint = endpoint or os.environ.get("SPEECH_ENDPOINT")

        if not self._speech_key:
            raise ValueError(
                "AzureSpeechTranslationModel requires a speech key (arg or SPEECH_KEY env)."
            )
        if not self._region and not self._endpoint:
            raise ValueError(
                "AzureSpeechTranslationModel requires a region or endpoint "
                "(args or SPEECH_REGION / SPEECH_ENDPOINT env)."
            )

        self._source_locale = to_source_locale(source_language)
        self._target_code = to_target_code(target_language)
        self._voice_name = voice_name or default_voice_for(target_language)

        self._synth_flush_delay = (
            float(synth_flush_delay)
            if synth_flush_delay is not None
            else DEFAULT_SYNTH_FLUSH_DELAY
        )

        self._sessions: set[AzureSpeechTranslationSession] = set()

        logger.info(
            "[azure] model created  source=%s  target=%s  voice=%s  region=%s",
            self._source_locale,
            self._target_code,
            self._voice_name,
            self._region or self._endpoint,
        )

    # ------------------------------------------------------------------
    # Config accessors (read by the session)
    # ------------------------------------------------------------------

    @property
    def synth_flush_delay(self) -> float:
        return self._synth_flush_delay

    @property
    def speech_key(self) -> str:
        return self._speech_key  # type: ignore[return-value]

    @property
    def region(self) -> str | None:
        return self._region

    @property
    def endpoint(self) -> str | None:
        return self._endpoint

    @property
    def source_locale(self) -> str:
        return self._source_locale

    @property
    def target_code(self) -> str:
        return self._target_code

    @property
    def voice_name(self) -> str | None:
        return self._voice_name

    @property
    def sample_rate(self) -> int:
        """Output sample rate of the audio frames this model produces (16 kHz)."""
        return OUTPUT_SAMPLE_RATE

    @property
    def model(self) -> str:
        return "azure-speech-translation"

    @property
    def provider(self) -> str:
        return "azure"

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def session(self) -> AzureSpeechTranslationSession:
        from .session import AzureSpeechTranslationSession

        sess = AzureSpeechTranslationSession(self)
        self._sessions.add(sess)
        return sess

    async def aclose(self) -> None:
        self._sessions.clear()
