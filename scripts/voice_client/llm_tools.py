"""HTTP wrappers around tray_finder.py's API plus OpenAI tool schemas.

The voice client exposes two tools to the LLM:

  - ``find_tray(tray_id: int)`` -> POST /find_tray on the robot. Blocks
    server-side until the tag is detected (or the robot's safety timeout
    fires, default 600 s).
  - ``send_to_sleep()`` -> POST /sleep on the robot. Returns immediately.

The robot host is read from the ``ROBOT_HOST`` env var
(default ``http://localhost:8080``).
"""

import os
from typing import Any, Callable, Dict

import httpx


DEFAULT_ROBOT_HOST = "http://localhost:8080"

# tray_finder's safety timeout is 600 s; give the HTTP client a slightly
# larger ceiling so the server-side timeout fires first and we get a
# structured ``{"found": false}`` response rather than a client-side
# ReadTimeout exception.
FIND_TRAY_HTTP_TIMEOUT_S = 900.0
SLEEP_HTTP_TIMEOUT_S = 5.0


TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "find_tray",
            "description": (
                "Sweep the robot arm to look for the tray with the given "
                "AprilTag id. The arm freezes the moment it sees the tag. "
                "The first call also wakes the arm from its sleep pose. "
                "Blocks until the tray is found or a safety timeout fires."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "tray_id": {
                        "type": "integer",
                        "description": "AprilTag id of the target tray.",
                    },
                },
                "required": ["tray_id"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_to_sleep",
            "description": (
                "Send the robot arm back to its rest/sleep pose and clear "
                "any active tray search. Returns immediately."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
    },
]


def _robot_host() -> str:
    return os.environ.get("ROBOT_HOST", DEFAULT_ROBOT_HOST).rstrip("/")


def find_tray(tray_id: Any) -> Dict[str, Any]:
    """Block until tray_finder reports the tag is detected (or it gives up).

    Returns the JSON dict from the server, or an error dict on failure.
    """
    try:
        tray_id_int = int(tray_id)
    except (TypeError, ValueError):
        return {"ok": False, "error": "tray_id must be an integer"}

    url = f"{_robot_host()}/find_tray"
    try:
        resp = httpx.post(
            url,
            json={"tray_id": tray_id_int},
            timeout=FIND_TRAY_HTTP_TIMEOUT_S,
        )
        resp.raise_for_status()
        return resp.json()
    except httpx.HTTPError as e:
        return {"ok": False, "error": f"robot HTTP error: {e}"}


def send_to_sleep() -> Dict[str, Any]:
    url = f"{_robot_host()}/sleep"
    try:
        resp = httpx.post(url, timeout=SLEEP_HTTP_TIMEOUT_S)
        resp.raise_for_status()
        return resp.json()
    except httpx.HTTPError as e:
        return {"ok": False, "error": f"robot HTTP error: {e}"}


# Dispatch table the LLM loop uses to actually invoke tools by name. Each
# impl accepts the parsed-JSON args dict and returns a JSON-serializable
# result dict.
TOOL_IMPLS: Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]] = {
    "find_tray": lambda args: find_tray(args.get("tray_id")),
    "send_to_sleep": lambda args: send_to_sleep(),
}
