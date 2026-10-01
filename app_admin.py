import sqlite3
from datetime import datetime
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from langchain_bot.auth import authenticate_user
from langchain_bot.agent import resume_with_decision
from langchain_bot.gmail_tools import initialize_gmail
from langchain_bot.rag_tool import initialize_vector_store
from langchain_bot.hitl_utils import handle_interrupt


def get_db_path() -> Path:
    """Return the path to the ecommerce database."""
    return Path(__file__).resolve().parent / "ecommerce.db"


def init_session() -> None:
    """Initialize session defaults."""
    st.session_state.setdefault("admin_email", None)
    st.session_state.setdefault("admin_name", None)


def get_pending_actions():
    """
    Get pending actions from the database.

    These are actions that were interrupted by HITL and are waiting for admin approval.
    """
    conn = sqlite3.connect(get_db_path())
    cursor = conn.cursor()

    try:
        cursor.execute("""
            SELECT
                id,
                thread_id,
                user_email,
                action_type,
                order_id,
                product_name,
                reason,
                created_at
            FROM pending_actions
            WHERE status = 'PENDING'
            ORDER BY created_at DESC
        """)

        rows = cursor.fetchall()
        pending = []

        for row in rows:
            (action_id, thread_id, user_email, action_type, order_id,
             product_name, reason, created_at) = row

            if action_type == "CANCEL_ORDER":
                tool_name = "cancel_order_action"
            else:
                tool_name = "create_return_action"

            pending.append({
                "id": action_id,
                "thread_id": thread_id,
                "action": tool_name,
                "action_type": action_type,
                "user_email": user_email,
                "order_id": order_id,
                "product_name": product_name or "",
                "reason": reason or "",
                "created_at": created_at,
            })

        return pending
    finally:
        conn.close()


def approve_action(thread_id: str, admin_email: str) -> tuple[bool, str]:
    """
    Approve a pending action and resume the agent.

    Returns:
        Tuple of (success: bool, result_message: str)
    """
    db_path = get_db_path()

    try:
        result = resume_with_decision(thread_id, "approve")

        with sqlite3.connect(db_path) as conn:
            conn.execute(
                """
                UPDATE pending_actions
                SET status = 'APPROVED', updated_at = ?
                WHERE thread_id = ? AND status = 'PENDING'
                """,
                (datetime.now().isoformat(), thread_id),
            )
            conn.commit()

        user_email = thread_id.split(":", 1)[0] if ":" in thread_id else ""

        is_interrupted_flag, interrupt_response = handle_interrupt(result, thread_id, user_email)

        if is_interrupted_flag:
            return True, f"Action approved. {interrupt_response}"
        elif result and result.get("messages"):
            agent_response = result["messages"][-1].content
            return True, agent_response
        else:
            return True, "Action approved and processed."

    except Exception as e:
        return False, f"Error: {e}"


def reject_action(thread_id: str, admin_email: str) -> tuple[bool, str]:
    """
    Reject a pending action and resume the agent.

    Returns:
        Tuple of (success: bool, result_message: str)
    """
    db_path = get_db_path()

    try:
        result = resume_with_decision(thread_id, "reject")

        with sqlite3.connect(db_path) as conn:
            conn.execute(
                """
                UPDATE pending_actions
                SET status = 'REJECTED', updated_at = ?
                WHERE thread_id = ? AND status = 'PENDING'
                """,
                (datetime.now().isoformat(), thread_id),
            )
            conn.commit()

        user_email = thread_id.split(":", 1)[0] if ":" in thread_id else ""

        is_interrupted_flag, interrupt_response = handle_interrupt(result, thread_id, user_email)

        if is_interrupted_flag:
            return True, f"Action rejected. {interrupt_response}"
        elif result and result.get("messages"):
            agent_response = result["messages"][-1].content
            return True, agent_response
        else:
            return True, "Action rejected."

    except Exception as e:
        return False, f"Error: {e}"


def main() -> None:
    st.set_page_config(
        page_title="Admin Dashboard - HITL Actions",
        page_icon="⚙️",
        layout="wide",
    )

    load_dotenv()
    init_session()

    if "services_initialized" not in st.session_state:
        with st.spinner("Initializing services..."):
            initialize_vector_store()
            initialize_gmail()
            st.session_state.services_initialized = True

    if not st.session_state.admin_email:
        st.title("🔐 Admin Login")

        with st.form("admin_login"):
            email = st.text_input("Email", value="admin@example.com")
            password = st.text_input("Password", type="password", value="admin123")
            submitted = st.form_submit_button("Login")

            if submitted:
                user = authenticate_user(email.strip(), password.strip(), "admin")
                if user and user["role"] == "admin":
                    st.session_state.admin_email = user["email"]
                    st.session_state.admin_name = user["full_name"]
                    st.success(f"Welcome, {user['full_name']}!")
                    st.rerun()
                else:
                    st.error("Invalid admin credentials.")
        return

    st.title("⚙️ Admin Dashboard - Human-in-the-Loop Actions")
    st.caption(f"Logged in as: {st.session_state.admin_name} ({st.session_state.admin_email})")

    st.subheader("📋 Pending Cancel/Return Requests")
    st.caption("These requests are waiting for your approval. The agent is paused until you decide.")

    pending_actions = get_pending_actions()

    if not pending_actions:
        st.success("No pending actions! 🎉")
        st.info("When customers request cancellations or returns, they will appear here for approval.")
    else:
        for action in pending_actions:
            thread_id = action["thread_id"]
            action_type = action["action"]
            user_email = action["user_email"]
            order_id = action["order_id"]

            if "cancel" in action_type.lower():
                icon = "❌"
                title = f"Cancel Order #{order_id}"
                action_desc = "Cancellation Request"
            else:
                icon = "↩️"
                product = action.get("product_name", "Item")
                title = f"Return: {product} (Order #{order_id})"
                action_desc = "Return Request"

            with st.expander(f"{icon} {title} - {user_email}", expanded=True):
                col1, col2 = st.columns([2, 1])

                with col1:
                    st.write(f"**Type:** {action_desc}")
                    st.write(f"**Customer:** {user_email}")
                    st.write(f"**Order ID:** {order_id}")

                    if action.get("product_name"):
                        st.write(f"**Product:** {action['product_name']}")
                    if action.get("reason"):
                        st.write(f"**Reason:** {action['reason']}")

                    st.caption(f"🔗 Thread: `{thread_id}`")

                with col2:
                    st.write("**Decision:**")

                    col_approve, col_reject = st.columns(2)

                    with col_approve:
                        if st.button("✅ Approve", key=f"approve_{thread_id}"):
                            with st.spinner("Approving & resuming agent..."):
                                success, result = approve_action(
                                    thread_id,
                                    st.session_state.admin_email,
                                )
                                if success:
                                    st.success("Approved!")
                                    st.info(f"📧 Agent: {result[:500]}")
                                    st.rerun()
                                else:
                                    st.error(result)

                    with col_reject:
                        if st.button("❌ Reject", key=f"reject_{thread_id}"):
                            with st.spinner("Rejecting..."):
                                success, result = reject_action(
                                    thread_id,
                                    st.session_state.admin_email,
                                )
                                if success:
                                    st.warning("Rejected.")
                                    st.info(f"📧 Agent: {result[:500]}")
                                    st.rerun()
                                else:
                                    st.error(result)


if __name__ == "__main__":
    main()