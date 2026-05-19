#!/usr/bin/env python3
"""Localhost push-to-talk voice client for tray_finder.

Pipeline (one turn):

    mic (sounddevice)
      -> Deepgram nova-3 STT
        -> OpenAI chat completions with tools=[find_tray, send_to_sleep]
          -> ElevenLabs TTS (eleven_turbo_v2_5, voice Rachel, pcm_24000)
            -> sounddevice speaker playback

Required env vars:
    OPENAI_API_KEY
    DEEPGRAM_API_KEY
    ELEVENLABS_API_KEY

Optional env vars:
    ROBOT_HOST          default http://localhost:8080
    OPENAI_MODEL        default gpt-5.5
    ELEVENLABS_VOICE_ID default 21m00Tcm4TlvDq8ikWAM (Rachel)
    ELEVENLABS_MODEL_ID default eleven_turbo_v2_5

Usage:
    python voice_client.py                  # hold SPACE to talk (default)
    python voice_client.py --mode=toggle    # press ENTER to start/stop
"""

import argparse
import io
import json
import os
import queue
import sys
import threading
import time
import wave
from typing import Any, Callable, Dict, List, Optional

import httpx
import numpy as np
import sounddevice as sd
from openai import OpenAI

from llm_tools import TOOL_IMPLS, TOOL_SCHEMAS


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
DEEPGRAM_API_KEY = os.environ.get("DEEPGRAM_API_KEY", "")
ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY", "")

OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-5.5")
ELEVENLABS_VOICE_ID = os.environ.get(
    "ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM"
)  # Rachel
ELEVENLABS_MODEL_ID = os.environ.get("ELEVENLABS_MODEL_ID", "eleven_turbo_v2_5")

# Deepgram nova-3 happily accepts a wide range of sample rates; 16 kHz
# mono int16 is a good cheap default for speech.
SAMPLE_RATE_IN = 16000

# Keyterm Prompting (nova-3, English only). Each term gets a small recognition
# boost, which helps with the robot's command vocabulary and the digit words
# that show up in tray ids.
DEEPGRAM_KEYTERMS: List[str] = [
    "tray",
    "find",
    "check",
    "sleep"
    "one",
    "two",
    "four",
    "five",
    "six",
]

# ElevenLabs ``pcm_24000`` returns raw little-endian int16 mono at 24 kHz.
SAMPLE_RATE_OUT = 24000

# Conversation memory cap, counted in completed user-anchored turns. The
# system prompt is always kept on top of this.
MAX_TURNS = 10

# Minimum recording length we bother sending to STT (filters accidental
# space taps).
MIN_RECORDING_S = 0.3

# Heartbeat tick for the long find_tray wait.
HEARTBEAT_INTERVAL_S = 2.0

# Cap on tool-call rounds per LLM turn (defensive; prevents loops).
MAX_TOOL_ROUNDS = 4


SYSTEM_PROMPT = (
    "You are a voice control agent for a tabletop robot arm that can locate "
    "trays by their AprilTag id and return to a rest pose.\n\n"
    "Your replies are spoken aloud, so keep every reply to ONE short "
    "sentence. Do not use lists, code, or markdown.\n\n"
    "If the user asks to find a tray but does not give an id, ask for the "
    "integer id and do nothing else. After find_tray returns found=true, "
    "confirm briefly (\"Found tray 3.\"). If found=false, say so and "
    "suggest trying another id or going to sleep. After send_to_sleep, "
    "confirm briefly (\"Going to sleep.\"). Do not narrate what you are "
    "about to do; just call the tool, then speak the result. Do not "
    "invent tools or capabilities."
)


# ---------------------------------------------------------------------------
# Audio capture
# ---------------------------------------------------------------------------


class Recorder:
    """Captures mic audio into an in-memory buffer between start() and stop().

    Thread-safe: start() / stop() are guarded by a lock so the pynput
    listener thread can drive them while the main thread queries state.
    """

    def __init__(self, sample_rate: int = SAMPLE_RATE_IN):
        self.sample_rate = sample_rate
        self._chunks: List[np.ndarray] = []
        self._stream: Optional[sd.InputStream] = None
        self._lock = threading.Lock()
        self.active = False

    def _callback(self, indata, frames, time_info, status):  # noqa: ARG002
        # Don't print xruns - they spam stdout and we tolerate a few dropped
        # frames at the start/end of a push-to-talk press anyway.
        self._chunks.append(indata.copy())

    def start(self) -> None:
        with self._lock:
            if self.active:
                return
            self._chunks = []
            self._stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="int16",
                callback=self._callback,
            )
            self._stream.start()
            self.active = True

    def stop(self) -> np.ndarray:
        with self._lock:
            if not self.active:
                return np.zeros((0, 1), dtype=np.int16)
            assert self._stream is not None
            self._stream.stop()
            self._stream.close()
            self._stream = None
            self.active = False
        if not self._chunks:
            return np.zeros((0, 1), dtype=np.int16)
        return np.concatenate(self._chunks, axis=0)


def pcm_to_wav_bytes(pcm: np.ndarray, sample_rate: int) -> bytes:
    """Wrap mono int16 PCM in an in-memory WAV container."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)  # int16
        w.setframerate(sample_rate)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Deepgram STT
# ---------------------------------------------------------------------------


def transcribe(wav_bytes: bytes) -> str:
    """POST WAV audio to Deepgram nova-3 and return the recognized text."""
    url = "https://api.deepgram.com/v1/listen"
    # httpx serializes a list value as repeated query params, which is what
    # Deepgram's keyterm endpoint expects: ?keyterm=tray&keyterm=find&...
    params: Dict[str, Any] = {
        "model": "nova-3",
        "smart_format": "true",
        "language": "en",
        "punctuate": "true",
    }
    if DEEPGRAM_KEYTERMS:
        params["keyterm"] = DEEPGRAM_KEYTERMS
    headers = {
        "Authorization": f"Token {DEEPGRAM_API_KEY}",
        "Content-Type": "audio/wav",
    }
    resp = httpx.post(
        url, params=params, headers=headers, content=wav_bytes, timeout=30.0
    )
    resp.raise_for_status()
    data = resp.json()
    try:
        return (
            data["results"]["channels"][0]["alternatives"][0]["transcript"].strip()
        )
    except (KeyError, IndexError, TypeError):
        return ""


# ---------------------------------------------------------------------------
# OpenAI LLM with tool calling
# ---------------------------------------------------------------------------


_openai_client: Optional[OpenAI] = None


def _openai() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI(api_key=OPENAI_API_KEY)
    return _openai_client


ToolRunner = Callable[
    [str, Dict[str, Any], Callable[[Dict[str, Any]], Dict[str, Any]]],
    Dict[str, Any],
]


def run_llm_turn(
    messages: List[Dict[str, Any]],
    run_tool: Optional[ToolRunner] = None,
) -> str:
    """Drive one user turn through the tool-calling loop.

    ``messages`` must already contain the new user message. Mutated in
    place: assistant and tool responses are appended. Returns the final
    assistant text.

    ``run_tool(name, args, impl) -> dict`` lets the caller wrap each
    individual tool invocation (e.g. with a heartbeat). If omitted, the
    impl is called directly.
    """
    for _ in range(MAX_TOOL_ROUNDS):
        resp = _openai().chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            tools=TOOL_SCHEMAS,
        )
        msg = resp.choices[0].message

        assistant_msg: Dict[str, Any] = {
            "role": "assistant",
            "content": msg.content,
        }
        if msg.tool_calls:
            assistant_msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in msg.tool_calls
            ]
        messages.append(assistant_msg)

        if not msg.tool_calls:
            return (msg.content or "").strip()

        for tc in msg.tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}

            impl = TOOL_IMPLS.get(name)
            if impl is None:
                result: Dict[str, Any] = {
                    "ok": False,
                    "error": f"unknown tool {name}",
                }
            else:
                try:
                    if run_tool is not None:
                        result = run_tool(name, args, impl)
                    else:
                        result = impl(args)
                except Exception as e:  # noqa: BLE001 - surface any tool failure
                    result = {"ok": False, "error": str(e)}

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result),
                }
            )

    # If we hit the round cap, force one final no-tools completion so we
    # always end with a spoken assistant reply.
    resp = _openai().chat.completions.create(
        model=OPENAI_MODEL,
        messages=messages,
    )
    text = (resp.choices[0].message.content or "").strip()
    messages.append({"role": "assistant", "content": text})
    return text


def _trim_history(history: List[Dict[str, Any]], max_turns: int = MAX_TURNS) -> None:
    """Keep system prompt + last ``max_turns`` user-anchored turns.

    Cutting at user-message boundaries keeps assistant/tool message pairs
    intact, which OpenAI requires (a tool message must follow the
    assistant message that contained its tool_call_id).
    """
    if not history or history[0].get("role") != "system":
        return
    user_indices = [i for i, m in enumerate(history) if m.get("role") == "user"]
    if len(user_indices) <= max_turns:
        return
    cut_at = user_indices[-max_turns]
    history[:] = [history[0]] + history[cut_at:]


# ---------------------------------------------------------------------------
# ElevenLabs TTS
# ---------------------------------------------------------------------------


def speak(text: str) -> None:
    """Synthesize ``text`` with ElevenLabs and play through the speaker.

    Uses ``output_format=pcm_24000`` which returns raw little-endian int16
    PCM at 24 kHz mono - no mp3 decoder needed, just hand the buffer to
    sounddevice. Blocks until playback finishes.
    """
    if not text:
        return
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}"
    params = {"output_format": "pcm_24000"}
    headers = {
        "xi-api-key": ELEVENLABS_API_KEY,
        "Content-Type": "application/json",
        "Accept": "audio/pcm",
    }
    body = {
        "text": text,
        "model_id": ELEVENLABS_MODEL_ID,
        "voice_settings": {"stability": 0.5, "similarity_boost": 0.75},
    }
    resp = httpx.post(url, params=params, headers=headers, json=body, timeout=60.0)
    resp.raise_for_status()
    pcm = np.frombuffer(resp.content, dtype=np.int16)
    if pcm.size == 0:
        return
    sd.play(pcm, samplerate=SAMPLE_RATE_OUT)
    sd.wait()


# ---------------------------------------------------------------------------
# Heartbeat for long tool waits
# ---------------------------------------------------------------------------


class Heartbeat:
    """Updates a single status line with elapsed seconds until stopped.

    Use as a context manager around a blocking call you want the user to
    see progress on (e.g. ``with Heartbeat("[searching for tray 3]"):
    impl(args)``).
    """

    def __init__(self, label: str, interval: float = HEARTBEAT_INTERVAL_S):
        self.label = label
        self.interval = interval
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._start = 0.0

    def __enter__(self) -> "Heartbeat":
        self._start = time.time()
        sys.stdout.write(f"{self.label} (0s)")
        sys.stdout.flush()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            elapsed = int(time.time() - self._start)
            sys.stdout.write(f"\r\033[K{self.label} ({elapsed}s)")
            sys.stdout.flush()

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.0)
        elapsed = int(time.time() - self._start)
        sys.stdout.write(f"\r\033[K{self.label} done in {elapsed}s\n")
        sys.stdout.flush()


# ---------------------------------------------------------------------------
# Per-turn pipeline orchestration
# ---------------------------------------------------------------------------


def _format_args(name: str, args: Dict[str, Any]) -> str:
    if name == "find_tray":
        return f"find_tray({args.get('tray_id')})"
    return f"{name}()"


def _wrap_tool_call(
    name: str,
    args: Dict[str, Any],
    impl: Callable[[Dict[str, Any]], Dict[str, Any]],
) -> Dict[str, Any]:
    """Pretty-print tool invocations; heartbeat the slow ones."""
    if name == "find_tray":
        tray_id = args.get("tray_id")
        with Heartbeat(f"[searching for tray {tray_id}]"):
            return impl(args)
    print(f"[calling {_format_args(name, args)}]")
    return impl(args)


def process_audio(pcm: np.ndarray, history: List[Dict[str, Any]]) -> None:
    """Run one full conversational turn: STT -> LLM (+tools) -> TTS."""
    duration_s = len(pcm) / SAMPLE_RATE_IN
    if duration_s < MIN_RECORDING_S:
        print("[recording too short, ignoring]")
        return

    print("[transcribing]")
    try:
        wav_bytes = pcm_to_wav_bytes(pcm, SAMPLE_RATE_IN)
        user_text = transcribe(wav_bytes)
    except Exception as e:  # noqa: BLE001
        print(f"[stt error: {e}]")
        return

    if not user_text:
        print("[no speech detected]")
        return

    print(f"you: {user_text}")
    history.append({"role": "user", "content": user_text})
    _trim_history(history)

    print("[thinking]")
    try:
        assistant_text = run_llm_turn(history, run_tool=_wrap_tool_call)
    except Exception as e:  # noqa: BLE001
        print(f"[llm error: {e}]")
        # Roll the user message back so the history doesn't drift out of
        # sync with the model's view.
        if history and history[-1].get("role") == "user":
            history.pop()
        return

    if not assistant_text:
        print("[empty assistant reply]")
        return

    print(f"robot: {assistant_text}")
    print("[speaking]")
    try:
        speak(assistant_text)
    except Exception as e:  # noqa: BLE001
        print(f"[tts error: {e}]")


# ---------------------------------------------------------------------------
# Interaction modes
# ---------------------------------------------------------------------------


_READY_LINE = "[ready]  hold SPACE to talk - q or Ctrl-C to quit"


def run_ptt(history: List[Dict[str, Any]]) -> None:
    """Hold-SPACE push-to-talk via pynput global key capture."""
    try:
        from pynput import keyboard
    except ImportError:
        print(
            "pynput is not installed. Install it (pip install pynput) or run "
            "with --mode=toggle.",
            file=sys.stderr,
        )
        sys.exit(1)

    recorder = Recorder()
    work_q: "queue.Queue[np.ndarray]" = queue.Queue()
    quit_event = threading.Event()
    processing = threading.Event()

    print(_READY_LINE)

    def on_press(key):
        # Ignore key events while a turn is in flight - the recorder must
        # not start mid-pipeline.
        if processing.is_set():
            return
        if key == keyboard.Key.space and not recorder.active:
            recorder.start()
            print("[recording...]")
        elif key == keyboard.KeyCode.from_char("q"):
            quit_event.set()
            return False  # stop listener

    def on_release(key):
        if key == keyboard.Key.space and recorder.active:
            pcm = recorder.stop()
            work_q.put(pcm)

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    try:
        while not quit_event.is_set():
            try:
                pcm = work_q.get(timeout=0.1)
            except queue.Empty:
                continue
            processing.set()
            try:
                process_audio(pcm, history)
            finally:
                processing.clear()
            print(_READY_LINE)
    except KeyboardInterrupt:
        pass
    finally:
        listener.stop()
        print("\n[bye]")


def run_toggle(history: List[Dict[str, Any]]) -> None:
    """Press-ENTER-to-toggle mode for terminals without global key capture."""
    recorder = Recorder()
    print("[ready]  press ENTER to start recording, ENTER again to stop "
          "- Ctrl-C to quit")
    try:
        while True:
            input("")
            recorder.start()
            print("[recording...]  press ENTER to stop")
            input("")
            pcm = recorder.stop()
            process_audio(pcm, history)
            print("[ready]  press ENTER to record")
    except (KeyboardInterrupt, EOFError):
        print("\n[bye]")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def check_env() -> None:
    missing = [
        name
        for name, val in [
            ("OPENAI_API_KEY", OPENAI_API_KEY),
            ("DEEPGRAM_API_KEY", DEEPGRAM_API_KEY),
            ("ELEVENLABS_API_KEY", ELEVENLABS_API_KEY),
        ]
        if not val
    ]
    if missing:
        print(
            "Missing required environment variables: " + ", ".join(missing),
            file=sys.stderr,
        )
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Localhost push-to-talk voice client for tray_finder.",
    )
    parser.add_argument(
        "--mode",
        choices=["ptt", "toggle"],
        default="ptt",
        help=(
            "ptt: hold SPACE (default; needs pynput + a desktop session). "
            "toggle: press ENTER to start/stop (works over SSH/Wayland)."
        ),
    )
    args = parser.parse_args()

    check_env()

    robot_host = os.environ.get("ROBOT_HOST", "http://localhost:8080")
    print(f"openai model:      {OPENAI_MODEL}")
    print(f"robot host:        {robot_host}")
    print(f"elevenlabs voice:  {ELEVENLABS_VOICE_ID} ({ELEVENLABS_MODEL_ID})")
    print()

    history: List[Dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]

    if args.mode == "ptt":
        run_ptt(history)
    else:
        run_toggle(history)


if __name__ == "__main__":
    main()
