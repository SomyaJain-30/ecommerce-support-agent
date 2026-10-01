# E-commerce Support Agent

A Streamlit e-commerce support chatbot built with LangChain and LangGraph. It can answer policy questions, look up customer order information, and guide customers through cancellation and return requests. Requests that change order or return data pause for admin approval in a separate dashboard.

## Features

- Customer login and persistent conversation history.
- Policy question answering with retrieval-augmented generation (RAG), OpenAI embeddings, and Chroma.
- Read-only lookups of the demo SQLite e-commerce database.
- Cancellation and return workflows with human-in-the-loop admin approval.
- Optional Gmail integration for customer email notifications.
- SQLite-backed LangGraph checkpoints for pausing and resuming agent conversations.

## Requirements

- Python 3.12 or newer.
- [uv](https://docs.astral.sh/uv/) for dependency and environment management.
- An OpenAI API key.
- Optional: Google OAuth desktop-app credentials to enable Gmail sending.

## Setup

1. Clone the repository and enter its directory:

   ```powershell
   git clone https://github.com/<your-username>/<your-repository>.git
   cd <your-repository>
   ```

2. Install the project dependencies:

   ```powershell
   uv sync
   ```

3. Create a `.env` file in the project root and add your OpenAI API key:

   ```dotenv
   OPENAI_API_KEY=your-openai-api-key
   ```

   Keep `.env` private; it is excluded from Git.

4. Create and seed the local demo database:

   ```powershell
   uv run python -m langchain_bot.db_init
   ```

   **Warning:** This command recreates the database from `ecommerce_setup.sql`, dropping and replacing existing tables and data. Run it only when you are prepared to reset the local demo database.

## Run the apps

Start the customer chat in one terminal:

```powershell
uv run streamlit run app.py
```

By default, Streamlit serves it at <http://localhost:8501>.

Start the admin approval dashboard in a second terminal:

```powershell
uv run streamlit run app_admin.py --server.port 8502
```

Open <http://localhost:8502> to review pending return and cancellation requests. The demo database includes seeded customer and admin users; see the `users` seed records in `ecommerce_setup.sql` for the local demo login details.

## Optional Gmail setup

To enable email notifications:

1. Enable the Gmail API in a Google Cloud project and create OAuth credentials for a desktop application.
2. Save the downloaded OAuth client file as `credentials.json` in the project root.
3. Run the app and complete the Google authorization flow in the browser.

The app stores the resulting token in `token.pickle`. Both credential files are excluded from Git. Without Gmail credentials, the chatbot can still run, but email sending is unavailable.

## Project structure

```text
.
├── app.py                       # Customer chat UI
├── app_admin.py                 # Admin approval dashboard
├── ecommerce_setup.sql          # Demo SQLite schema and seed data
├── policies/                    # Policy documents indexed for RAG
├── src/langchain_bot/
│   ├── agent.py                 # LangGraph agent and checkpointing
│   ├── auth.py                  # Demo authentication
│   ├── db_init.py               # Demo database initialization
│   ├── gmail_tools.py           # Optional Gmail integration
│   ├── rag_tool.py              # Policy retrieval and Chroma store
│   ├── action_tools.py          # Cancellation and return actions
│   └── sql_tools.py             # Database lookup tools
├── pyproject.toml               # Project metadata and dependencies
└── uv.lock                      # Locked dependency versions
```

## Security note

This is a learning/demo project, not a production-ready support system. The seeded database uses plain-text demo passwords, and authentication and authorization should be replaced with production-grade implementations before deployment. Do not commit API keys, Google credentials, OAuth tokens, or real customer data. The initialization script is destructive to the local database.
