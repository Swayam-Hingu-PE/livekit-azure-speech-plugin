"""
Azure AI Speech (Live Interpreter) integration for the LiveKit RTT pipeline.

Exposes a `RealtimeModel` / `RealtimeSession` pair that wraps Azure's own
speech-to-speech translation cascade (ASR -> translate -> TTS) so it can be
used interchangeably with the OpenAI realtime / translate providers inside
`RealtimeTranslator` (see translator/realtime/realtime_translator.py).

Wire it up through `get_llm` with metadata `llm.type = "AZURE_SPEECH"`.
"""

from .model import AzureSpeechTranslationModel
from .session import AzureSpeechTranslationSession

__all__ = [
    "AzureSpeechTranslationModel",
    "AzureSpeechTranslationSession",
]
