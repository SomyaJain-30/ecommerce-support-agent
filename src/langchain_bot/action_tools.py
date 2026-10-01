"""Action tools for cancel/return that trigger HITL interruption.

These tools are the interrupt points - admin must approve before they execute.
The actual database updates happen INSIDE these tools after HITL approval.

Uses ToolRuntime[SessionContext] to access user_email and conversation_id
without requiring the LLM to provide them as parameters.
"""

import sqlite3
from datetime import datetime
from pathlib import Path

from langchain.tools import tool, ToolRuntime

from langchain_bot.context import SessionContext


def get_db_path() -> Path:
    """Return path to ecommerce database."""
    return Path(__file__).resolve().parents[2] / "ecommerce.db"


@tool
def cancel_order_action(
    order_id: int,
    runtime: ToolRuntime[SessionContext],
) -> str:
    """
    Cancel an order for a customer. REQUIRES ADMIN APPROVAL (HITL).

    This tool will be interrupted by HITL middleware. When admin approves,
    it will execute and update the database.

    Args:
        order_id: The order ID to cancel
        runtime: ToolRuntime providing access to SessionContext

    Returns:
        Confirmation message after cancellation is complete
    """
    context = runtime.context
    user_email = context.user_email
    conversation_id = context.conversation_id
    thread_id = f"{user_email}:{conversation_id}"

    conn = sqlite3.connect(get_db_path())
    cursor = conn.cursor()

    try:
        cursor.execute("SELECT id FROM users WHERE email = ?", (user_email,))
        user = cursor.fetchone()
        if not user:
            return f"Error: User {user_email} not found"
        user_id = user[0]

        cursor.execute(
            """
            SELECT id, status, total_amount FROM orders
            WHERE id = ? AND user_id = ?
        """,
            (order_id, user_id),
        )
        order = cursor.fetchone()

        if not order:
            return f"Error: Order #{order_id} not found for user"
        if order[1] == "CANCELLED":
            return f"Order #{order_id} is already cancelled"
        if order[1] in ("SHIPPED", "DELIVERED"):
            return f"Cannot cancel - order has been {order[1].lower()}. Please create a return request instead."

        now = datetime.now().isoformat()
        total_amount = order[2]

        cursor.execute("UPDATE orders SET status = 'CANCELLED' WHERE id = ?", (order_id,))

        cursor.execute("UPDATE payments SET status = 'REFUNDED' WHERE order_id = ?", (order_id,))

        cursor.execute(
            """
            INSERT INTO tickets (user_id, subject, status, thread_id, user_email, created_at, updated_at)
            VALUES (?, ?, 'RESOLVED', ?, ?, ?, ?)
        """,
            (user_id, f"Order #{order_id} Cancellation", thread_id, user_email, now, now),
        )

        conn.commit()

        return f"""Order #{order_id} has been successfully cancelled!

**Refund Details:**
- Amount: ${total_amount:.2f}
- Status: REFUNDED
- The refund will be processed to your original payment method within 3-5 business days.

A confirmation email has been sent to {user_email}."""

    except Exception as e:
        return f"Error cancelling order: {e}"
    finally:
        conn.close()


@tool
def create_return_action(
    order_id: int,
    product_name: str,
    reason: str,
    runtime: ToolRuntime[SessionContext],
) -> str:
    """
    Create a return request for a customer. REQUIRES ADMIN APPROVAL (HITL).

    Only allowed when the order status is SHIPPED or DELIVERED.
    PLACED orders must be cancelled instead — do not use this tool for them.

    Args:
        order_id: The order ID containing the item
        product_name: Name of the product to return
        reason: Reason for the return
        runtime: ToolRuntime providing access to SessionContext

    Returns:
        Confirmation message after return is processed
    """
    context = runtime.context
    user_email = context.user_email
    conversation_id = context.conversation_id
    thread_id = f"{user_email}:{conversation_id}"

    conn = sqlite3.connect(get_db_path())
    cursor = conn.cursor()

    try:
        cursor.execute("SELECT id FROM users WHERE email = ?", (user_email,))
        user = cursor.fetchone()
        if not user:
            return f"Error: User {user_email} not found"
        user_id = user[0]

        cursor.execute(
            """
            SELECT id, status FROM orders WHERE id = ? AND user_id = ?
        """,
            (order_id, user_id),
        )
        order = cursor.fetchone()

        if not order:
            return f"Error: Order #{order_id} not found for user"
        if order[1] == "CANCELLED":
            return "Cannot return items from a cancelled order"
        if order[1] == "PLACED":
            return (
                f"Cannot create a return for order #{order_id} — it has not shipped yet "
                f"(status: PLACED). If you no longer want the order, please request a **cancellation** instead."
            )
        if order[1] not in ("SHIPPED", "DELIVERED"):
            return (
                f"Cannot return items from order #{order_id} with status '{order[1]}'. "
                "Returns are only allowed for SHIPPED or DELIVERED orders."
            )

        cursor.execute(
            """
            SELECT oi.id, p.name, oi.unit_price, oi.quantity
            FROM order_items oi
            JOIN products p ON oi.product_id = p.id
            WHERE oi.order_id = ? AND p.name LIKE ?
        """,
            (order_id, f"%{product_name}%"),
        )
        item = cursor.fetchone()

        if not item:
            return f"Error: Product '{product_name}' not found in order #{order_id}"

        order_item_id, actual_product_name, unit_price, quantity = item
        refund_amount = unit_price * quantity
        now = datetime.now().isoformat()

        cursor.execute(
            """
            INSERT INTO returns (order_id, order_item_id, user_id, reason, status, requested_at, resolved_at)
            VALUES (?, ?, ?, ?, 'APPROVED', ?, ?)
        """,
            (order_id, order_item_id, user_id, reason, now, now),
        )
        return_id = cursor.lastrowid

        cursor.execute(
            """
            INSERT INTO tickets (user_id, return_id, subject, status, thread_id, user_email, created_at, updated_at)
            VALUES (?, ?, ?, 'RESOLVED', ?, ?, ?, ?)
        """,
            (user_id, return_id, f"Return: {actual_product_name}", thread_id, user_email, now, now),
        )

        cursor.execute(
            """
            INSERT INTO payments (order_id, amount, status, payment_method, transaction_reference, paid_at)
            VALUES (?, ?, 'REFUNDED', 'REFUND', ?, ?)
        """,
            (order_id, refund_amount, f"RETURN-{return_id}", now),
        )

        conn.commit()

        return f"""Return request #{return_id} has been APPROVED!

**Return Details:**
- Product: {actual_product_name}
- Order: #{order_id}
- Reason: {reason}

**Refund:**
- Amount: ${refund_amount:.2f}
- Status: REFUNDED

**Next Steps:**
1. Please ship the item to our returns center within 14 days
2. Use the prepaid shipping label sent to {user_email}
3. Refund will be processed within 3-5 business days after we receive the item

A confirmation email has been sent to {user_email}."""

    except Exception as e:
        return f"Error processing return: {e}"
    finally:
        conn.close()


def get_action_tools() -> list:
    """Return the action tools that trigger HITL."""
    return [cancel_order_action, create_return_action]


__all__ = ["get_action_tools", "cancel_order_action", "create_return_action"]