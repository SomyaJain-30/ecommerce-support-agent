from pathlib import Path
from typing import List

from langchain_community.tools.gmail import GmailSendMessage


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def get_gmail_service():
    import pickle

    from google.auth.transport.requests import Request
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build

    scopes = ["https://www.googleapis.com/auth/gmail.send"]
    creds = None

    root = get_project_root()
    token_path = root / "token.pickle"
    credentials_path = root / "credentials.json"

    if token_path.exists():
        with open(token_path, "rb") as token:
            creds = pickle.load(token)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not credentials_path.exists():
                raise FileNotFoundError(f"credentials.json not found at {credentials_path}")
            flow = InstalledAppFlow.from_client_secrets_file(str(credentials_path), scopes)
            creds = flow.run_local_server(port=0)

        with open(token_path, "wb") as token:
            pickle.dump(creds, token)

    return build("gmail", "v1", credentials=creds)


_gmail_service = None
_gmail_initialized = False


def initialize_gmail() -> bool:
    global _gmail_service, _gmail_initialized
    try:
        _gmail_service = get_gmail_service()
        _gmail_initialized = True
        print("Gmail service initialized successfully!")
        return True
    except Exception as e:
        print(f"Gmail initialization failed: {e}")
        _gmail_service = None
        _gmail_initialized = False
        return False


def get_gmail_tools() -> List:
    global _gmail_service, _gmail_initialized
    if not _gmail_initialized:
        initialize_gmail()
    if _gmail_service:
        return [GmailSendMessage(api_resource=_gmail_service)]
    return []


def is_gmail_available() -> bool:
    return _gmail_initialized and _gmail_service is not None


__all__ = ["get_gmail_tools", "initialize_gmail", "is_gmail_available"]