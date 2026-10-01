"""LangChain agent with RAG, SQL toolkit, Gmail, Action tools, and HITL middleware.

HITL Flow:
1. Customer requests cancel/return
2. Agent sends "request received" email
3. Agent calls cancel_order_action or create_return_action -> HITL INTERRUPTS
4. Admin approves/rejects
5. Agent resumes: Tool executes (updates DB), agent sends confirmation email
"""

import os
import sqlite3

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from dotenv import load_dotenv

from langchain_bot.rag_tool import search_policies
from langchain_bot.middleware import get_logging_middleware
from langchain_bot.sql_tools import get_sql_tools
from langchain_bot.gmail_tools import get_gmail_tools, is_gmail_available
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.types import Command
from langchain_bot.action_tools import get_action_tools
from langchain_bot.context import SessionContext
load_dotenv()


def get_llm() -> ChatOpenAI:
    """Return a configured OpenAI chat model."""
    return ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

def get_tools() -> list:
    """Return the list of tools available to the agent."""
    tools = [search_policies]
    tools.extend(get_sql_tools())
    tools.extend(get_gmail_tools())
    tools.extend(get_action_tools())
    return tools


def get_system_prompt() -> str:
    """Return the system prompt for the e-commerce support agent."""
    gmail_note = (
        "Gmail is available — send a brief 'request received' email before calling action tools."
        if is_gmail_available()
        else "Gmail is NOT available — skip all email steps and go directly to action tools after SQL verification."
    )
    return f"""You are a helpful e-commerce customer support assistant.

## Your Capabilities:

### 1. Policy Questions
Use `search_policies` for: returns policy, refunds, shipping, cancellations, FAQs

### 2. Database Queries (SQL Toolkit)
Use for viewing data ONLY (orders, products, etc.):
- `sql_db_list_tables`: List tables
- `sql_db_schema`: Get table schemas
- `sql_db_query_checker`: Validate queries
- `sql_db_query`: Execute SELECT queries

### 3. Email Notifications
{gmail_note}
Use `GmailSendMessage` only when it is in your tool list.

### 4. Action Tools (Require Admin Approval)
These tools trigger Human-in-the-Loop approval:
- `cancel_order_action(order_id)`: Cancel an order (admin must approve)
- `create_return_action(order_id, product_name, reason)`: Create a return request (admin must approve)

**Note:** When calling action tools, you only need to provide `order_id` (and `product_name`/`reason` for returns).
The `user_email` and `thread_id` are automatically available to the tools via context - you don't need to provide them.

## IMPORTANT: User & Thread Identification

**Message format: `[User: email@example.com] [Thread: thread_id]`**
- Use user email to filter data in ALL queries
- For action tools: Only provide order_id and other required params - user_email and thread_id are available via context

## Database Schema:

**users**: id, email, password, full_name, role
**products**: id, name, description, price, stock_qty
**orders**: id, user_id, status (PLACED/SHIPPED/DELIVERED/CANCELLED), order_date, total_amount
**order_items**: id, order_id, product_id, quantity, unit_price
**payments**: id, order_id, amount, status (PAID/REFUNDED)
**returns**: id, order_id, order_item_id, user_id, reason, status (PENDING/APPROVED/REJECTED)
**tickets**: id, user_id, return_id, subject, status, thread_id, user_email

## Workflow for Cancel/Return Requests:

### When customer requests ORDER CANCELLATION:
1. Run at most **2 SQL queries** to verify order exists, belongs to user, and status is 'PLACED'.
2. If status is SHIPPED/DELIVERED, explain cancellation is not possible and suggest a return instead.
3. Call `cancel_order_action(order_id)` — **do not run more SQL queries first**. THIS WILL PAUSE FOR ADMIN APPROVAL.
4. Only call ONE tool per turn.
5. After admin approves, the tool updates the DB automatically.

### When customer requests RETURN:
1. Run at most **3 SQL queries** to verify: (a) order belongs to user, (b) order status is **SHIPPED or DELIVERED**, (c) exact product name in order.
2. **Returns are NOT allowed for PLACED orders** — if status is PLACED, tell the customer to **cancel** the order instead (do not call create_return_action).
3. Use: `SELECT o.id, o.status, p.name FROM order_items oi JOIN products p ON oi.product_id = p.id JOIN orders o ON oi.order_id = o.id JOIN users u ON o.user_id = u.id WHERE oi.order_id = ? AND u.email = 'user@example.com'`
4. **NEVER guess product names** — use exact names from query results.
5. If a **PENDING return already exists** for that order/item, tell the customer and **stop** (do not re-query or call create_return_action).
6. If reason is missing, ask the customer in plain text (no tools).
7. Only call `create_return_action` when status is **SHIPPED** or **DELIVERED**.
8. THIS WILL PAUSE FOR ADMIN APPROVAL. Only call ONE tool per turn.
9. For multiple items, handle one item per user turn.

## Important Notes:
- **NEVER repeat the same SQL query.** If you already have the data, act or reply in text.
- **Hard limit: max 4 tool calls per user message.** Then you MUST reply in text or call the action tool.
- **ALWAYS call ONE tool at a time.**
- Action tools pause for admin approval (you won't get immediate response)
- For queries, use SQL toolkit with user email filtering
- NEVER expose other users' data"""


_checkpointer = None
_conn = None


def get_checkpointer() -> SqliteSaver:
    """Get or create the shared SqliteSaver checkpointer instance.

    Uses CHECKPOINTS_DB_PATH env var if set, otherwise defaults to
    'checkpoints.sqlite' in the project root. Tables are created on first use"""
    global _checkpointer, _conn
    if _checkpointer is None:
        db_path = os.getenv("CHECKPOINTS_DB_PATH", "checkpoints.sqlite")
        _conn = sqlite3.connect(db_path, check_same_thread=False)
        _checkpointer = SqliteSaver(conn=_conn)
        _checkpointer.setup()
    return _checkpointer


def get_hitl_middleware() -> HumanInTheLoopMiddleware:
    """
    Create Human-in-the-Loop middleware.
    Interrupts on ACTION tools (cancel/return) for admin approval.
    """
    return HumanInTheLoopMiddleware(
        interrupt_on={
            "cancel_order_action": {"allowed_decisions": ["approve", "reject"]},
            "create_return_action": {"allowed_decisions": ["approve", "reject"]},
        }
    )


def create_support_agent():
    """Create and return the e-commerce support agent with HITL enabled."""
    llm = get_llm()
    tools = get_tools()
    system_prompt = get_system_prompt()
    checkpointer = get_checkpointer()

    middleware = [get_hitl_middleware()] + get_logging_middleware()

    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=system_prompt,
        checkpointer=checkpointer,
        context_schema=SessionContext,
        middleware=middleware,
    )

    return agent


_agent = None

def get_agent():
    global _agent 
    if _agent is None:
        _agent = create_support_agent()
    return _agent


def reset_agent():
    """Reset the agent instance."""
    global _agent
    _agent = None


def build_thread_id(user_email: str, conversation_id: str) -> str:
    """Build a thread ID from user_email and conversation_id."""
    return f"{user_email}:{conversation_id}"


def get_thread_config(user_email: str, conversation_id: str) -> dict:
    thread_id = f"{user_email}:{conversation_id}"
    return {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": 20
    }


def is_interrupted(result: dict) -> bool:
    """
    Check if the agent result indicates an HITL interruption.

    When HITL middleware interrupts, the last message will be an AIMessage
    with tool_calls for cancel_order_action or create_return_action.
    """
    if not result or "messages" not in result:
        return False

    messages = result["messages"]
    if not messages:
        return False

    last_msg = messages[-1]
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        for tc in last_msg.tool_calls:
            tool_name = tc.get("name", "")
            if tool_name in ("cancel_order_action", "create_return_action"):
                return True
    return False


def get_pending_action(result: dict) -> dict | None:
    """Get the pending action tool call from an interrupted result."""
    if not result or "messages" not in result:
        return None

    messages = result["messages"]
    if not messages:
        return None

    last_msg = messages[-1]
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        for tc in last_msg.tool_calls:
            tool_name = tc.get("name", "")
            if tool_name in ("cancel_order_action", "create_return_action"):
                return tc
    return None


def resume_with_decision(thread_id: str, decision: str, role: str = "admin") -> dict:
    """
    Resume an interrupted agent conversation with admin decision.

    Args:
        thread_id: The thread_id (user_email:conversation_id)
        decision: "approve" or "reject"
        role: The role of the person resuming (default: "admin")

    Returns:
        The agent's response after resumption
    """
    agent = get_agent()

    user_email = None
    conversation_id = None
    if ":" in thread_id:
        parts = thread_id.split(":", 1)
        user_email = parts[0]
        conversation_id = parts[1] if len(parts) > 1 else ""

    config = {
        "configurable": {
            "thread_id": thread_id,
        }
    }

    context = SessionContext(
        user_email=user_email or "",
        conversation_id=conversation_id or "",
        role=role,
    )

    if decision == "approve":
        resume_data = {"decisions": [{"type": "approve"}]}
    else:
        resume_data = {"decisions": [{"type": "reject"}]}

    result = agent.invoke(Command(resume=resume_data), config=config, context=context)
    return result


__all__ = [
    "get_agent",
    "create_support_agent",
    "get_thread_config",
    "build_thread_id",
    "get_checkpointer",
    "reset_agent",
    "is_interrupted",
    "get_pending_action",
    "resume_with_decision",
]