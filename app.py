import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from uuid import uuid4
from langchain_bot.auth import authenticate_user
import json
import sqlite3
from pathlib import Path
from langchain_bot.rag_tool import initialize_vector_store
from langchain_bot.agent import get_agent, get_thread_config, build_thread_id, reset_agent
from langchain_bot.gmail_tools import initialize_gmail, is_gmail_available
from langchain_bot.context import SessionContext
from langchain_bot.hitl_utils import handle_interrupt, has_pending_action_for_thread

def get_llm():
    return ChatOpenAI(model="gpt-4.1-mini", temperature=0.3)
 
def build_chain(llm):
    prompt = ChatPromptTemplate.from_messages([
        ("system", "you are a concise, helpful assistant. Use prior chat history to stay on context."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ])

    return prompt | llm

def init_session():
    st.session_state.setdefault("conversations", {})
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("user_email", None)
    st.session_state.setdefault("user_role", None)
    st.session_state.setdefault("conversation_id", None)
    st.session_state.setdefault("vector_store_ready", False)
    st.session_state.setdefault("gmail_ready", False)

def ensure_conv_store():
    path = Path(__file__).resolve().parent / "conversations.db"
    with sqlite3.connect(path) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                user_email TEXT NOT NULL,
                messages TEXT NOT NULL
            )
        """)
        conn.commit()
    return path

def serialize_messages(messages):
    return json.dumps([
        {"role": "human" if isinstance(m, HumanMessage) else "ai", "content": m.content}
        for m in messages
    ])

def deserialize_messages(payload):
    raw = json.loads(payload)
    out = []
    for item in raw:
        if item.get("role") == "human":
            out.append(HumanMessage(content=item.get("content", "")))
        else:
            out.append(AIMessage(content=item.get("content", "")))
    return out


def save_conversation(conv_id, user_email, messages):
    store = ensure_conv_store()
    with sqlite3.connect(store) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO conversations (id, user_email, messages) VALUES (?, ?, ?)",
            (conv_id, user_email, serialize_messages(messages)),
        )
        conn.commit()

def start_new_conversation():
    st.session_state.conversation_id = str(uuid4())
    st.session_state.messages = [AIMessage(content="Hi! I am LangChain bot. Ask me anything.")]
    st.session_state.conversations[st.session_state.conversation_id] = list(st.session_state.messages)
    save_conversation(st.session_state.conversation_id, st.session_state.user_email, st.session_state.messages)

def load_conversation(user_email):
    store = ensure_conv_store()
    with sqlite3.connect(store) as conn:
        rows = conn.execute(
            "SELECT id, messages FROM conversations WHERE user_email = ?", (user_email,)
        ).fetchall()

    conversations = {}
    for conv_id, payload in rows:
        conversations[conv_id] = deserialize_messages(payload)
    return conversations


def render_history():
    for msg in st.session_state.messages:
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        with st.chat_message(role):
            st.markdown(msg.content)

def chat_round(user_input: str):
    print(user_input)
    """Send the user prompt through the agent and save reply."""
    st.session_state.messages.append(HumanMessage(content=user_input))

    thread_id = build_thread_id(st.session_state.user_email, st.session_state.conversation_id)
    if has_pending_action_for_thread(thread_id):
        st.session_state.messages.append(AIMessage(
            content="You already have a **pending return or cancellation** awaiting admin review. "
            "Please wait for approval before submitting another request."
        ))
        if st.session_state.conversation_id:
            st.session_state.conversations[st.session_state.conversation_id] = list(st.session_state.messages)
            save_conversation(st.session_state.conversation_id, st.session_state.user_email, st.session_state.messages)
        return
    
    agent = get_agent()
    config = get_thread_config(st.session_state.user_email, st.session_state.conversation_id)

    user_context = f"[User: {st.session_state.user_email}] [Thread: {thread_id}] "
    new_message = {"role": "user", "content": user_context + user_input}

    context = SessionContext(
        user_email=st.session_state.user_email,
        conversation_id=st.session_state.conversation_id,
        role=st.session_state.user_role or "customer",
    )
    
    try:
        result = agent.invoke({"messages": [new_message]}, config=config, context=context)
    except Exception as exc:
        if exc.__class__.__name__ == "GraphRecursionError":
            agent_response = (
                "I ran into a problem completing that request — I was checking too many things. "
                "Please try again with the **order number** and **exact product name** from your order."
            )
            st.session_state.messages.append(AIMessage(content=agent_response))
            if st.session_state.conversation_id:
                st.session_state.conversations[st.session_state.conversation_id] = list(st.session_state.messages)
                save_conversation(st.session_state.conversation_id, st.session_state.user_email, st.session_state.messages)
            return
        raise

    is_interrupted_flag, interrupt_response = handle_interrupt(result, thread_id, st.session_state.user_email)
    if is_interrupted_flag:
        agent_response = interrupt_response
    else:
        agent_response = None
        for msg in reversed(result.get("messages", [])):
            if hasattr(msg, "content") and msg.content and not (hasattr(msg, "tool_calls") and msg.tool_calls):
                agent_response = msg.content
                break

    if not agent_response:
        agent_response = "I'm sorry, I couldn't process that request."

    st.session_state.messages.append(AIMessage(content=agent_response))

    if st.session_state.conversation_id:
        st.session_state.conversations[st.session_state.conversation_id] = list(st.session_state.messages)
        if st.session_state.user_email:
            save_conversation(st.session_state.conversation_id, st.session_state.user_email, st.session_state.messages)


def main():
    st.set_page_config(page_title="LangChain Bot", page_icon="🤖")
    st.title("LangChain Bot")
    st.caption("Streamlit UI with persistent conversations.")

    load_dotenv()
    init_session()
    user_email = st.session_state.user_email
    user_role = st.session_state.user_role

    if not user_email:
        with st.form("login_form"):
            email = st.text_input("Email", value="", placeholder="Enter your email")
            password = st.text_input("Password", type="password", placeholder="Enter your password")
            role = st.selectbox("Role", ["customer", "admin"], index=0)
            submitted = st.form_submit_button("Login")
            print(f"isSubmitted: {submitted}")
            if submitted:
                print("Submitted")
                user = authenticate_user(email.strip(), password.strip(), role.strip())
                print(user)
                if user:
                    st.session_state.user_email = user["email"]
                    st.session_state.user_role = user["role"]
                    st.session_state.conversations = load_conversation(user["email"])
                    if st.session_state.conversations:
                        first = next(iter(st.session_state.conversations))
                        # load_conversation(first)
                        st.session_state.conversation_id = first
                        st.session_state.messages = list(st.session_state.conversations[first])
                    else:
                        start_new_conversation()
                    st.success(f"Logged in as {user['email']} ({user['role']})")
                    st.rerun()
                    print("rerunning")
                else:
                    st.error("Invalid credentials.")

        return

    if not st.session_state.vector_store_ready:
        with st.spinner("🔄 Initializing knowledge base..."):
            try:
                initialize_vector_store()

                st.session_state.vector_store_ready = True

            except Exception as e:
                st.error(
                    f"Failed to initialize knowledge base: {e}"
                )
                st.stop()

    if not st.session_state.gmail_ready:
        st.session_state.gmail_ready = initialize_gmail()
        reset_agent()  # Reset the agent to ensure Gmail tools are loaded

    with st.sidebar:
        st.header("Conversations")

        if is_gmail_available():
            st.success("📧 Gmail: Enabled")
        else:
            st.warning("📧 Gmail: Not configured")

        if st.button("Start new conversation"):
            start_new_conversation()
            st.rerun()
        conv_ids = list(st.session_state.conversations.keys())
        if conv_ids:
            current = conv_ids.index(st.session_state.conversation_id) if st.session_state.conversation_id in conv_ids else 0
            selected = st.selectbox("Select conversation", conv_ids, index=current)
            if selected != st.session_state.conversation_id:
                # load_conversation(selected)
                st.session_state.conversation_id = selected
                st.session_state.messages = list(st.session_state.conversations[selected])
                st.rerun()
        else:
            st.caption("No thread yet")
       

    cid = st.session_state.conversation_id
    cid_display = cid if cid else "- (create a new thread)"
    st.info(f"Logged in as {user_email} ({user_role}). Conversation ID: {cid_display}")

    render_history()

    prompt = st.chat_input("Ask a question")

    if prompt:
        llm = get_llm()
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                chat_round(prompt)
        st.rerun()


if __name__ == "__main__":
    main()