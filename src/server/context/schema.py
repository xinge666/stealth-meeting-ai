from dataclasses import dataclass, field
from datetime import datetime

@dataclass
class ConversationTurn:
    """A single turn in the conversation history."""
    speaker: str  # "other", "self", or "ai"
    text: str
    timestamp: datetime = field(default_factory=datetime.now)
