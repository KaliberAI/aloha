# Voice client for `tray_finder`

A lightweight localhost CLI that lets you talk to the robot:

```
mic -> Deepgram nova-3 -> OpenAI (tools: find_tray, send_to_sleep)
    -> ElevenLabs eleven_turbo_v2_5 (Rachel) -> speaker
```

The LLM's tools are thin HTTP wrappers around the existing `tray_finder.py`
endpoints (`POST /find_tray`, `POST /sleep`), so `tray_finder.py` itself
does not need to change.

## Prereqs

`tray_finder.py` is running and reachable at `ROBOT_HOST`
(default `http://localhost:8080`).

System dependency for the mic/speaker (PortAudio):

```bash
sudo apt install -y libportaudio2
```

Python deps - install into a venv so they don't collide with the aloha env:

```bash
python3 -m venv ~/voice_client_venv
source ~/voice_client_venv/bin/activate
pip install -r requirements.txt
```

## Configure

```bash
export OPENAI_API_KEY=sk-...
export DEEPGRAM_API_KEY=...
export ELEVENLABS_API_KEY=...

# Optional overrides
export ROBOT_HOST=http://localhost:8080
export OPENAI_MODEL=gpt-5.5
export ELEVENLABS_VOICE_ID=21m00Tcm4TlvDq8ikWAM   # Rachel
export ELEVENLABS_MODEL_ID=eleven_turbo_v2_5
```

## Run

Hold-to-talk (default; requires an active desktop session):

```bash
cd src/aloha/scripts/voice_client
python voice_client.py
```

Press-ENTER-to-toggle (works over SSH / Wayland / headless):

```bash
python voice_client.py --mode=toggle
```

## What it looks like

```
openai model:      gpt-5.5
robot host:        http://localhost:8080
elevenlabs voice:  21m00Tcm4TlvDq8ikWAM (eleven_turbo_v2_5)

[ready]  hold SPACE to talk - q or Ctrl-C to quit
[recording...]
[transcribing]
you: find tray three
[thinking]
[searching for tray 3] (12s)
[searching for tray 3] done in 14s
robot: Found tray three.
[speaking]
[ready]  hold SPACE to talk - q or Ctrl-C to quit
```

Then:

```
[recording...]
[transcribing]
you: go to sleep
[thinking]
[calling send_to_sleep()]
robot: Going to sleep.
[speaking]
```

## Files

- `voice_client.py` - main CLI: PTT, STT, LLM loop, TTS, playback.
- `llm_tools.py` - `find_tray` / `send_to_sleep` HTTP wrappers and the
  OpenAI tool-call schemas.
- `requirements.txt` - Python deps (separate from aloha's).

## Notes & gotchas

- **PTT key capture is global.** `pynput` grabs SPACE across the whole
  desktop, so don't hold SPACE in another app while the client is
  running. Use `--mode=toggle` if that's a problem (or you're over SSH /
  Wayland).
- **Audio devices** are sounddevice's defaults. Pick a mic/speaker via
  PulseAudio / pavucontrol (or `pactl set-default-source` /
  `set-default-sink`) before launching.
- **`find_tray` blocks server-side** for up to 600 s. The client shows a
  `[searching for tray N] (Ns)` heartbeat while it waits. The OpenAI
  call won't return until the tool returns, so the spoken confirmation
  arrives right after the arm freezes.
- **Conversation memory** is kept in-process and trimmed to the last
  ~10 turns. Restart the script to reset.
- The `OPENAI_MODEL` default is `gpt-5.5`. If your OpenAI account uses a
  different model id, set the env var.
