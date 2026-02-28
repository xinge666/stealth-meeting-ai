import json
import base64
from enum import Enum
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict

class ClientMessageType(str, Enum):
    AUDIO_CHUNK = "audio_chunk"
    AUDIO_SPEECH_END = "audio_speech_end"
    SCREEN_TEXT = "screen_text"

@dataclass
class ClientMessage:
    type: ClientMessageType
    payload: Dict[str, Any]

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, data: str) -> "ClientMessage":
        parsed = json.loads(data)
        return cls(type=ClientMessageType(parsed["type"]), payload=parsed.get("payload", {}))

    @classmethod
    def audio_chunk(cls, pcm_data: bytes) -> "ClientMessage":
        """Creates an audio chunk message. pcm_data is base64 encoded."""
        b64_data = base64.b64encode(pcm_data).decode('utf-8')
        return cls(type=ClientMessageType.AUDIO_CHUNK, payload={"audio": b64_data})
        
    @classmethod
    def audio_speech_end(cls) -> "ClientMessage":
        """Signals that the current speech segment has ended."""
        return cls(type=ClientMessageType.AUDIO_SPEECH_END, payload={})

    @classmethod
    def screen_text(cls, text: str) -> "ClientMessage":
        """Sends extracted OCR text from the screen."""
        return cls(type=ClientMessageType.SCREEN_TEXT, payload={"text": text})

# Server to Client messages (for future status updates, etc.)
class ServerMessageType(str, Enum):
    STATUS_UPDATE = "status_update"
    ERROR = "error"

@dataclass
class ServerMessage:
    type: ServerMessageType
    payload: Dict[str, Any]

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, data: str) -> "ServerMessage":
        parsed = json.loads(data)
        return cls(type=ServerMessageType(parsed["type"]), payload=parsed.get("payload", {}))
