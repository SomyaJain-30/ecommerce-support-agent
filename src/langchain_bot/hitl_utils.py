"""Utility functions for handling Human-in-the-Loop (HITL) interruptions."""

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

from langchain_bot.agent import is_interrupted, get_pending_action


def get_db_path() -> Path:
    """Return the path to the ecommerce database."""
    return Path(__file__).resolve().parents[2] / "ecommerce.db"



def save_pending_action(
    thread_id: str,
    user_email: str,
    action_type: str,
    order_id: int,
    product_name: Optional[str] = None,
    reason: Optional[str] = None,
) -> None:
    """Save a pending action to the database when agent is interrupted by HITL."""
    db_path = get_db_path()
    now = datetime.now().isoformat()

    with sqlite3.connect(db_path) as conn:
        existing = conn.execute(
            """
            SELECT id FROM pending_actions
            WHERE thread_id = ? AND order_id = ? AND action_type = ? AND status = 'PENDING'
            """,
            (thread_id, order_id, action_type),
        ).fetchone()
        if existing:
            return
        conn.execute(
            """
            INSERT INTO pending_actions
            (thread_id, user_email, action_type, order_id, product_name, reason, status, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, 'PENDING', ?, ?)
            """,
            (thread_id, user_email, action_type, order_id, product_name, reason, now, now),
        )
        conn.commit()


def has_pending_action_for_thread(thread_id: str) -> bool:
    """Return True if this conversation already has a pending HITL action."""
    with sqlite3.connect(get_db_path()) as conn:
        row = conn.execute(
            "SELECT 1 FROM pending_actions WHERE thread_id = ? AND status = 'PENDING' LIMIT 1",
            (thread_id,),
        ).fetchone()
    return row is not None


def _order_allows_return(order_id: int, user_email: str) -> tuple[bool, str | None]:
    """Return (True, None) if return is allowed, else (False, error_message)."""
    with sqlite3.connect(get_db_path()) as conn:
        row = conn.execute(
            """
            SELECT o.status FROM orders o
            JOIN users u ON o.user_id = u.id
            WHERE o.id = ? AND u.email = ?
            """,
            (order_id, user_email),
        ).fetchone()
    if not row:
        return False, f"Order #{order_id} was not found for your account."
    status = row[0]
    if status == "PLACED":
        return False, (
            f"Order #{order_id} has not shipped yet (status: PLACED). "
            "You cannot return it — please **cancel the order** instead if you no longer want it."
        )
    if status == "CANCELLED":
        return False, f"Order #{order_id} is already cancelled."
    if status not in ("SHIPPED", "DELIVERED"):
        return False, f"Returns are not allowed for order #{order_id} with status '{status}'."
    return True, None


def handle_interrupt(
    result: dict,
    thread_id: str,
    user_email: str,
) -> Tuple[bool, Optional[str]]:
    """
    Check if agent result is interrupted and save pending action if needed.

    This function:
    1. Checks if the agent was interrupted by HITL
    2. Extracts the pending action details
    3. Saves the pending action to the database
    4. Returns the appropriate response message for the user

    Args:
        result: The agent's result dictionary
        thread_id: The conversation thread ID (user_email:conversation_id)
        user_email: Customer email

    Returns:
        Tuple of (is_interrupted: bool, response_message: str | None)
        - If interrupted: (True, user-friendly message about pending approval)
        - If not interrupted: (False, None) - caller should use normal response
    """
    if not is_interrupted(result):
        return False, None

    pending_action = get_pending_action(result)
    if not pending_action:
        return False, None

    action_name = pending_action.get("name", "action")
    action_args = pending_action.get("args", {})

    order_id = action_args.get("order_id")
    product_name = action_args.get("product_name")
    reason = action_args.get("reason")

    if "cancel" in action_name.lower():
        action_type = "CANCEL_ORDER"
        agent_response = (
            "**Your cancellation request has been submitted!**\n\n"
            "Your request is now **pending admin review**.\n\n"
            "You will receive an email once an admin approves or rejects your cancellation."
        )
    else:
        action_type = "CREATE_RETURN"
        agent_response = (
            "**Your return request has been submitted!**\n\n"
            "Your request is now **pending admin review**.\n\n"
            "You will receive an email once an admin approves or rejects your return."
        )

    if order_id is not None:
        if action_type == "CREATE_RETURN":
            allowed, block_msg = _order_allows_return(order_id, user_email)
            if not allowed:
                return True, block_msg
        save_pending_action(
            thread_id=thread_id,
            user_email=user_email,
            action_type=action_type,
            order_id=order_id,
            product_name=product_name,
            reason=reason,
        )

    return True, agent_response

__all__ = ["handle_interrupt", "save_pending_action", "get_db_path", "has_pending_action_for_thread"]