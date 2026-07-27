from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Any

import azure.cognitiveservices.speech as speechsdk
from livekit import rtc
from livekit.agents import llm, utils, vad
from livekit.agents.types import NOT_GIVEN, NotGivenOr
from livekit.agents.utils import is_given

from .model import INPUT_SAMPLE_RATE, OUTPUT_SAMPLE_RATE

if TYPE_CHECKING:
    from .model import AzureSpeechTranslationModel

logger = logging.getLogger(__name__)


# Azure gives no reliable per-utterance VAD (speech_start/end_detected bracket the
# whole speech run), which the filler needs. So — exactly like gpt-realtime's
# server VAD — we run a real-time Silero VAD on the input audio to emit precise
# input_speech_started / input_speech_stopped. One shared model, one stream per
# session (Silero inference is cheap).
_shared_vad: vad.VAD | None = None


def _get_shared_vad() -> vad.VAD:
    global _shared_vad
    if _shared_vad is None:
        from livekit.plugins import silero

        _shared_vad = silero.VAD.load(
            min_speech_duration=0.05,
            min_silence_duration=0.55,
            activation_threshold=0.5,
            prefix_padding_duration=0.5,
        )
        logger.info("[azure] loaded shared Silero VAD for filler speech events")
    return _shared_vad


class _Generation:
    """One translated utterance: the channels RealtimeTranslator drains."""

    def __init__(self, message_id: str) -> None:
        self.message_id = message_id
        self.message_ch: utils.aio.Chan[llm.MessageGeneration] = utils.aio.Chan()
        self.function_ch: utils.aio.Chan[llm.FunctionCall] = utils.aio.Chan()
        self.audio_ch: utils.aio.Chan[rtc.AudioFrame] = utils.aio.Chan()
        self.text_ch: utils.aio.Chan[str] = utils.aio.Chan()
        self.text_delivered = False


class AzureSpeechTranslationSession(llm.RealtimeSession):
    """
    RealtimeSession that drives Azure's `TranslationRecognizer` (ASR -> translate
    -> TTS) and re-emits its callbacks as the generic realtime events consumed by
    `RealtimeTranslator`:

        - input_speech_started / input_speech_stopped   (Azure VAD)
        - input_audio_transcription_completed           (final source transcript)
        - generation_created -> MessageGeneration        (translated audio + text)
        - error                                          (Azure canceled)

    The Azure SDK invokes its callbacks on its own threads; every callback here
    extracts primitives and hops back onto the agent event loop with
    `loop.call_soon_threadsafe` before touching any channel or emitting an event.
    """

    def __init__(self, realtime_model: AzureSpeechTranslationModel) -> None:
        super().__init__(realtime_model)
        self._azure_model = realtime_model
        self._loop = asyncio.get_event_loop()

        self._chat_ctx = llm.ChatContext.empty()
        self._instructions: str = ""
        self._voice: str | None = None

        # Azure objects (created lazily on first audio so update_options(voice=...)
        # is applied before the recognizer is built).
        self._recognizer: speechsdk.translation.TranslationRecognizer | None = None
        self._push_stream: speechsdk.audio.PushAudioInputStream | None = None
        self._started = False
        self._closed = False

        # Real-time VAD that drives the filler speech events.
        self._vad_stream: vad.VADStream | None = None
        self._tasks: list[asyncio.Task] = []

        # Input resampler (native track rate -> 16 kHz mono), cached by rate/channels.
        self._resampler: rtc.AudioResampler | None = None
        self._resampler_key: tuple[int, int] | None = None

        # Current turn's generation + translated text waiting to be delivered.
        self._current_gen: _Generation | None = None
        self._pending_text: str = ""
        # Fallback timer that closes the generation after a gap in synthesized audio.
        self._synth_flush_handle: asyncio.TimerHandle | None = None

        # Per-utterance speech state. Azure's speech_start/end_detected bracket the
        # WHOLE speech run (not each sentence), so we drive input_speech_started /
        # input_speech_stopped off the ASR instead: started on the first hint of a
        # new utterance, stopped on `recognized` (the reliable per-sentence event).
        self._speech_active = False

    # ------------------------------------------------------------------
    # RealtimeSession contract (read-only / config)
    # ------------------------------------------------------------------

    @property
    def chat_ctx(self) -> llm.ChatContext:
        return self._chat_ctx

    @property
    def tools(self) -> llm.ToolContext:
        return llm.ToolContext.empty()

    async def update_instructions(self, instructions: str) -> None:
        # Azure's cascade takes no prompt; keep it for parity but don't send it.
        self._instructions = instructions

    async def update_chat_ctx(self, chat_ctx: llm.ChatContext) -> None:
        pass  # not supported

    async def update_tools(self, tools: list[llm.Tool]) -> None:
        pass  # not supported

    def update_options(
        self,
        *,
        voice: NotGivenOr[str] = NOT_GIVEN,
        tool_choice: NotGivenOr[Any] = NOT_GIVEN,
        **kwargs: Any,
    ) -> None:
        if is_given(voice):
            self._voice = voice

    # ------------------------------------------------------------------
    # Unsupported operations (translation mode)
    # ------------------------------------------------------------------

    def push_video(self, frame: rtc.VideoFrame) -> None:
        pass

    def commit_audio(self) -> None:
        pass

    def clear_audio(self) -> None:
        pass

    def interrupt(self) -> None:
        pass

    def truncate(self, **kwargs: Any) -> None:
        pass

    def generate_reply(
        self, **kwargs: Any
    ) -> asyncio.Future[llm.GenerationCreatedEvent]:
        fut: asyncio.Future[llm.GenerationCreatedEvent] = asyncio.Future()
        fut.set_exception(
            llm.RealtimeError("generate_reply is not supported in Azure translation mode")
        )
        return fut

    # ------------------------------------------------------------------
    # Audio ingestion
    # ------------------------------------------------------------------

    def push_audio(self, frame: rtc.AudioFrame) -> None:
        if self._closed:
            return
        self._ensure_started()
        # Feed Azure FIRST so the translation path is never delayed; the VAD (which
        # only drives filler speech events) runs on a separate parallel stream.
        try:
            for f in self._resample(frame):
                self._push_stream.write(f.data.tobytes())  # type: ignore[union-attr]
        except Exception:
            logger.exception("[azure] push_audio failed")
        # Raw frame to the VAD (it resamples internally); non-blocking enqueue.
        if self._vad_stream is not None:
            try:
                self._vad_stream.push_frame(frame)
            except RuntimeError:
                pass  # stream input ended during teardown

    def _resample(self, frame: rtc.AudioFrame) -> list[rtc.AudioFrame]:
        if frame.sample_rate == INPUT_SAMPLE_RATE and frame.num_channels == 1:
            return [frame]
        key = (frame.sample_rate, frame.num_channels)
        if self._resampler is None or self._resampler_key != key:
            self._resampler = rtc.AudioResampler(
                input_rate=frame.sample_rate,
                output_rate=INPUT_SAMPLE_RATE,
                num_channels=frame.num_channels,
            )
            self._resampler_key = key
        return self._resampler.push(frame)

    # ------------------------------------------------------------------
    # Azure recognizer setup (lazy)
    # ------------------------------------------------------------------

    def _ensure_started(self) -> None:
        if self._started:
            return
        self._started = True
        model = self._azure_model

        fmt = speechsdk.audio.AudioStreamFormat(
            samples_per_second=INPUT_SAMPLE_RATE, bits_per_sample=16, channels=1
        )
        self._push_stream = speechsdk.audio.PushAudioInputStream(fmt)
        audio_config = speechsdk.audio.AudioConfig(stream=self._push_stream)

        if model.endpoint:
            config = speechsdk.translation.SpeechTranslationConfig(
                subscription=model.speech_key, endpoint=model.endpoint
            )
        else:
            config = speechsdk.translation.SpeechTranslationConfig(
                subscription=model.speech_key, region=model.region
            )

        config.speech_recognition_language = model.source_locale
        config.add_target_language(model.target_code)

        voice = self._voice or model.voice_name
        if voice:
            config.voice_name = voice  # enables spoken translation output

        # NOTE: TranslationRecognizer ignores set_speech_synthesis_output_format;
        # its synthesis is always 16 kHz mono PCM (= OUTPUT_SAMPLE_RATE). Frames
        # are labelled with that true rate so downstream resampling is correct.

        self._recognizer = speechsdk.translation.TranslationRecognizer(
            translation_config=config, audio_config=audio_config
        )
        self._connect_callbacks(self._recognizer)
        self._recognizer.start_continuous_recognition_async()

        # Start the real-time VAD that drives input_speech_started/stopped.
        self._vad_stream = _get_shared_vad().stream()
        self._tasks.append(asyncio.create_task(self._process_vad_events()))

        logger.info(
            "[azure] recognition started  source=%s -> target=%s  voice=%s",
            model.source_locale,
            model.target_code,
            voice,
        )

    def _connect_callbacks(
        self, recognizer: speechsdk.translation.TranslationRecognizer
    ) -> None:
        recognizer.recognized.connect(self._cb_recognized)
        recognizer.synthesizing.connect(self._cb_synthesizing)
        recognizer.canceled.connect(self._cb_canceled)
        # NOTE: input_speech_started / input_speech_stopped are driven by the local
        # Silero VAD (see _process_vad_events), NOT Azure's speech_start/end_detected
        # — those bracket the whole speech run, not each sentence, and break filler.
        recognizer.session_started.connect(
            lambda evt: logger.debug("[azure] session started: %s", evt)
        )
        recognizer.session_stopped.connect(
            lambda evt: logger.debug("[azure] session stopped: %s", evt)
        )

    # ------------------------------------------------------------------
    # Azure callbacks (run on SDK threads) -> hop onto the event loop
    # ------------------------------------------------------------------

    def _cb_recognized(self, evt: Any) -> None:
        try:
            if evt.result.reason != speechsdk.ResultReason.TranslatedSpeech:
                return
            source = evt.result.text or ""
            translations = evt.result.translations
            translation = (
                translations.get(self._azure_model.target_code, "") if translations else ""
            )
            self._loop.call_soon_threadsafe(self._on_recognized_main, source, translation)
        except Exception:
            logger.exception("[azure] recognized callback failed")

    def _cb_synthesizing(self, evt: Any) -> None:
        try:
            audio = evt.result.audio
            audio_bytes = bytes(audio) if audio else b""
            completed = (
                evt.result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted
            )
            self._loop.call_soon_threadsafe(
                self._on_synthesizing_main, audio_bytes, completed
            )
        except Exception:
            logger.exception("[azure] synthesizing callback failed")

    def _cb_canceled(self, evt: Any) -> None:
        reason = str(getattr(evt, "reason", ""))
        details = getattr(evt, "error_details", None)
        self._loop.call_soon_threadsafe(self._on_canceled_main, reason, details)

    # ------------------------------------------------------------------
    # Real-time VAD -> filler speech events (matches gpt-realtime server VAD)
    # ------------------------------------------------------------------

    @utils.log_exceptions(logger=logger)
    async def _process_vad_events(self) -> None:
        assert self._vad_stream is not None
        async for ev in self._vad_stream:
            if ev.type == vad.VADEventType.START_OF_SPEECH:
                self._ensure_speech_started()
            elif ev.type == vad.VADEventType.END_OF_SPEECH:
                self._end_speech()

    # ------------------------------------------------------------------
    # Loop-thread handlers (own all channel / emit mutations)
    # ------------------------------------------------------------------

    def _ensure_speech_started(self) -> None:
        """Emit input_speech_started once per utterance (idempotent).

        Deliberately does NOT touch the generation (matching gpt-realtime, whose
        speech events are independent of the response lifecycle). Closing the prior
        generation here would run its filler cleanup on a later loop tick — after
        this synchronous emit already tried to arm the next filler — which is the
        race that made the filler intermittent. Generations close on their own via
        SynthesizingAudioCompleted / the flush timer.
        """
        if self._speech_active:
            return
        self._speech_active = True
        self.emit("input_speech_started", llm.InputSpeechStartedEvent())

    def _end_speech(self) -> None:
        """Emit input_speech_stopped once per utterance (arms the filler wait)."""
        if not self._speech_active:
            return
        self._speech_active = False
        self.emit(
            "input_speech_stopped",
            llm.InputSpeechStoppedEvent(user_transcription_enabled=True),
        )

    def _on_recognized_main(self, source: str, translation: str) -> None:
        # Speech start/stop events are owned by the VAD; here we only surface the
        # final source transcript and hand the translated text to the generation.
        if source:
            self.emit(
                "input_audio_transcription_completed",
                llm.InputTranscriptionCompleted(
                    item_id=utils.shortuuid("azure_src_"),
                    transcript=source,
                    is_final=True,
                    confidence=None,
                ),
            )

        self._pending_text = translation or ""
        # If synthesis already opened this turn's generation, deliver text now;
        # otherwise it is flushed when the generation opens.
        if self._current_gen is not None and self._pending_text:
            self._deliver_text(self._current_gen, self._pending_text)
            self._pending_text = ""

    def _on_synthesizing_main(self, audio_bytes: bytes, completed: bool) -> None:
        if audio_bytes:
            if self._current_gen is None:
                self._open_generation()
            gen = self._current_gen
            if gen is not None and not gen.audio_ch.closed:
                gen.audio_ch.send_nowait(
                    rtc.AudioFrame(
                        data=audio_bytes,
                        sample_rate=OUTPUT_SAMPLE_RATE,
                        num_channels=1,
                        samples_per_channel=len(audio_bytes) // 2,
                    )
                )
            # (Re)arm the fallback close in case SynthesizingAudioCompleted never
            # fires for this utterance — keeps filler cleanup prompt (see above).
            self._arm_synth_flush()
        if completed:
            self._finish_generation()

    def _arm_synth_flush(self) -> None:
        if self._synth_flush_handle is not None:
            self._synth_flush_handle.cancel()
        self._synth_flush_handle = self._loop.call_later(
            self._azure_model.synth_flush_delay, self._finish_generation
        )

    def _on_canceled_main(self, reason: str, details: str | None) -> None:
        error = Exception(f"Azure recognition canceled: {reason} — {details}")
        logger.error("[azure] canceled: %s — %s", reason, details)
        self.emit(
            "error",
            llm.RealtimeModelError(
                timestamp=time.time(),
                label=self._realtime_model.label,
                error=error,
                recoverable=False,
            ),
        )

    # ------------------------------------------------------------------
    # Generation lifecycle helpers
    # ------------------------------------------------------------------

    def _open_generation(self) -> _Generation:
        gen = _Generation(utils.shortuuid("azure_tr_"))
        modalities: asyncio.Future[list[str]] = asyncio.Future()
        modalities.set_result(["audio", "text"])
        gen.message_ch.send_nowait(
            llm.MessageGeneration(
                message_id=gen.message_id,
                text_stream=gen.text_ch,
                audio_stream=gen.audio_ch,
                modalities=modalities,
            )
        )
        self._current_gen = gen
        self.emit(
            "generation_created",
            llm.GenerationCreatedEvent(
                message_stream=gen.message_ch,
                function_stream=gen.function_ch,
                user_initiated=False,
                response_id=gen.message_id,
            ),
        )
        if self._pending_text:
            self._deliver_text(gen, self._pending_text)
            self._pending_text = ""
        return gen

    def _deliver_text(self, gen: _Generation, text: str) -> None:
        if not gen.text_ch.closed:
            gen.text_ch.send_nowait(text)
            gen.text_delivered = True

    def _finish_generation(self) -> None:
        if self._synth_flush_handle is not None:
            self._synth_flush_handle.cancel()
            self._synth_flush_handle = None
        gen = self._current_gen
        if gen is None:
            return
        if self._pending_text and not gen.text_delivered:
            self._deliver_text(gen, self._pending_text)
        self._pending_text = ""
        for ch in (gen.audio_ch, gen.text_ch, gen.function_ch, gen.message_ch):
            if not ch.closed:
                ch.close()
        self._current_gen = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True

        recognizer = self._recognizer
        if recognizer is not None:
            try:
                await self._loop.run_in_executor(
                    None, lambda: recognizer.stop_continuous_recognition_async().get()
                )
            except Exception:
                logger.exception("[azure] error stopping recognizer")

        if self._vad_stream is not None:
            try:
                self._vad_stream.end_input()
            except Exception:
                pass
            try:
                await self._vad_stream.aclose()
            except Exception:
                pass

        await utils.aio.cancel_and_wait(*self._tasks)
        self._tasks.clear()

        if self._push_stream is not None:
            try:
                self._push_stream.close()
            except Exception:
                pass

        self._finish_generation()
        self._azure_model._sessions.discard(self)
