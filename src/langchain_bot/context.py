"""Context schema for the e-commerce support agent.

Uses ToolRuntime to pass user_email and conversation_id to tools
without requiring the LLM to provide them explicitly.
"""

from dataclasses import dataclass


@dataclass
class SessionContext:
    """Context passed to tools via ToolRuntime.

    Contains session information that tools need but shouldn't
    be part of the LLM's tool call parameters.
    """
    user_email: str
    conversation_id: str
    role: str = "customer"  # "customer" or "admin"
