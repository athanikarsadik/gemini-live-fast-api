import contextlib
import os
import uuid
import json
import base64
import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, status
from fastapi.responses import HTMLResponse
from starlette.websockets import WebSocketState
from dotenv import load_dotenv

from google import genai
from google.genai import types

# -----------------------------
# Config
# -----------------------------
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise ValueError("Missing GOOGLE_API_KEY environment variable.")

MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-exp")
HTTP_OPTIONS = {"api_version": "v1alpha"}

# -----------------------------
# App + shared client
# -----------------------------
app = FastAPI(title="Gemini Live over FastAPI")

genai_client: Optional[genai.Client] = None

@app.on_event("startup")
async def startup() -> None:
    global genai_client
    # Create once per process; reuse
    genai_client = genai.Client(api_key=API_KEY, http_options=HTTP_OPTIONS)

@app.on_event("shutdown")
async def shutdown() -> None:
    # google-genai client currently doesn't need explicit close
    pass

# -----------------------------
# Session manager (in-memory)
# -----------------------------
@dataclass
class LiveSessionState:
    session_id: str
    user_id: Optional[str]
    ws: WebSocket
    model: str
    gemini_context: Any = None             # async context manager (the "connect(...)" object)
    gemini_session: Any = None             # the live session object inside the context
    tasks: set[asyncio.Task] = field(default_factory=set)

class SessionManager:
    def __init__(self) -> None:
        self._sessions: Dict[str, LiveSessionState] = {}
        self._lock = asyncio.Lock()

    async def register(self, state: LiveSessionState) -> None:
        async with self._lock:
            self._sessions[state.session_id] = state

    async def unregister(self, session_id: str) -> None:
        async with self._lock:
            self._sessions.pop(session_id, None)

    def get(self, session_id: str) -> Optional[LiveSessionState]:
        return self._sessions.get(session_id)

    def count_for_user(self, user_id: Optional[str]) -> int:
        if not user_id:
            return 0
        return sum(1 for s in self._sessions.values() if s.user_id == user_id)

sessions = SessionManager()

# -----------------------------
# Helpers
# -----------------------------
def flatten_generation_config(config: dict) -> dict:
    """Bring generation_config.* to the top level to match your sample."""
    cfg = dict(config or {})
    if "generation_config" in cfg and isinstance(cfg["generation_config"], dict):
        gen_cfg = cfg.pop("generation_config")
        cfg.update(gen_cfg)
    return cfg

async def forward_client_to_gemini(ws: WebSocket, gemini_session: Any) -> None:
    """
    Receive JSON frames from the client and forward media chunks to Gemini.
    Expected JSON:
    {
      "realtime_input": {
        "media_chunks": [
          {"mime_type":"audio/pcm","sample_rate":16000,"data":"<base64>"},
          {"mime_type":"image/jpeg","data":"<base64>"}
        ]
      }
    }
    """
    while True:
        msg = await ws.receive_text()
        try:
            payload = json.loads(msg)
        except Exception:
            # Ignore malformed frames; you can also close with 1003
            continue

        realtime = payload.get("realtime_input")
        if not realtime:
            continue

        for chunk in realtime.get("media_chunks", []):
            mime = chunk.get("mime_type")
            if not mime or "data" not in chunk:
                continue

            raw = base64.b64decode(chunk["data"])

            # PCM: include sample rate in mime type per your original pattern
            if mime == "audio/pcm":
                rate = chunk.get("sample_rate", 16000)
                mime = f"audio/pcm;rate={rate}"

            blob = types.Blob(mime_type=mime, data=raw)

            # Map mime to the correct parameter
            if mime.startswith("audio/"):
                await gemini_session.send_realtime_input(audio=blob)
            elif mime.startswith("image/") or mime.startswith("video/"):
                # For images/video, Google's Python SDK uses "video" param; it handles images too.
                await gemini_session.send_realtime_input(video=blob)
            else:
                # Unsupported mime: ignore or log
                pass

async def stream_gemini_to_client(ws: WebSocket, gemini_session: Any) -> None:
    """
    Read streaming responses from Gemini and forward to the client.
    Emits:
      {"text": "..."} for text tokens
      {"audio": "<base64>", "sample_rate": 24000} for audio chunks
    """
    while True:
        async for response in gemini_session.receive():
            if response.server_content is None:
                continue

            model_turn = response.server_content.model_turn
            if model_turn:
                for part in model_turn.parts:
                    # Text
                    text = getattr(part, "text", None)
                    if text is not None:
                        print(f"text: {text}")
                        await ws.send_text(json.dumps({"text": text}))
                        continue

                    # Audio (inline_data)
                    inline = getattr(part, "inline_data", None)
                    if inline is not None:
                        # Google returns 24kHz PCM by default for audio-out
                        base64_audio = base64.b64encode(inline.data).decode("utf-8")
                        await ws.send_text(json.dumps({"audio": base64_audio, "sample_rate": 24000}))
                        continue

            if getattr(response.server_content, "turn_complete", False):
                # Optional: notify client the model finished a turn
                await ws.send_text(json.dumps({"turn_complete": True}))

# -----------------------------
# WebSocket endpoint
# -----------------------------
@app.websocket("/ws/realtime")
async def websocket_realtime(ws: WebSocket):
    """
    Connect like: ws://<host>/ws/realtime?user_id=alice&model=gemini-2.0-flash-exp
    Protocol:
      1) Client sends a first JSON frame { "setup": { ...config... } } immediately after connect.
         We flatten .generation_config into the top-level.
         We'll default response_modalities=["AUDIO"] unless provided.
      2) Then client may send realtime_input frames with base64 media as shown above.
    """
    # Accept with limits (tune as needed)
    
    await ws.accept(subprotocol=None)

    # Basic per-connection metadata
    user_id = ws.query_params.get("user_id")
    model = ws.query_params.get("model", MODEL)

    # 1) Receive setup/config frame
    try:
        raw = await ws.receive_text()
        setup_msg = json.loads(raw)
        config = flatten_generation_config(setup_msg.get("setup", {}))
    except Exception:
        
        await ws.close(code=status.WS_1003_UNSUPPORTED_DATA)
        return

    # Ensure audio responses by default (you can override from client)
    config.setdefault("response_modalities", ["AUDIO"])

    # Optional: limit how many concurrent live sessions a single user can hold
    if sessions.count_for_user(user_id) >= 1:
        await ws.close(code=status.WS_1013_TRY_AGAIN_LATER)
        return

    # 2) Create Gemini live session bound to this WebSocket
    session_id = str(uuid.uuid4())
    state = LiveSessionState(session_id=session_id, user_id=user_id, ws=ws, model=model)

    await sessions.register(state)

    # Build the async context manager for live session
    # NOTE: google-genai uses "async with client.aio.live.connect(...)" normally.
    # Inside a long-lived WS handler we manually __aenter__/__aexit__ so we can
    # run tasks and always cleanup, even on abrupt disconnects.
    if genai_client is None:
        await ws.close(code=status.WS_1011_INTERNAL_ERROR)
        return

    state.gemini_context = genai_client.aio.live.connect(model=model, config=config)

    # Enter live session context
    try:
        state.gemini_session = await state.gemini_context.__aenter__()
    except Exception as e:
        await sessions.unregister(session_id)
        await ws.close(code=status.WS_1011_INTERNAL_ERROR)
        return

    # 3) Pump data both ways
    try:
        to_gemini = asyncio.create_task(forward_client_to_gemini(ws, state.gemini_session))
        from_gemini = asyncio.create_task(stream_gemini_to_client(ws, state.gemini_session))
        state.tasks.update({to_gemini, from_gemini})

        # If either task ends (disconnect or error), cancel the other
        done, pending = await asyncio.wait(
            {to_gemini, from_gemini},
            return_when=asyncio.FIRST_EXCEPTION
        )

        for task in pending:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    except WebSocketDisconnect:
        # Client closed connection
        pass
    except Exception:
        # Log in real app
        pass
    finally:
        # Cleanup
        try:
            # Exit gemini live context
            if state.gemini_context is not None:
                await state.gemini_context.__aexit__(None, None, None)
        finally:
            await sessions.unregister(session_id)
            if ws.application_state is WebSocketState.CONNECTED:
                with contextlib.suppress(Exception):
                    await ws.close()