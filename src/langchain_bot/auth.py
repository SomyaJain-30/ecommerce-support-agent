import sqlite3
from pathlib import Path
from typing import Optional, TypedDict

class UserRecord(TypedDict):
    email: str
    full_name: str
    role: str

def authenticate_user(email: str, password: str, role: Optional[str] = None) -> Optional[UserRecord]:
    db_path = Path(__file__).resolve().parents[2] / "ecommerce.db"
    query = "SELECT email, full_name, role FROM users WHERE email = ? AND password = ?"
    params = [email, password]

    if role:
        query += " AND role = ?"
        params.append(role)

    with sqlite3.connect(db_path) as conn:
        row = conn.execute(query, params).fetchone()

    if not row:
        return None
    return {"email": row[0], "full_name": row[1], "role": row[2]}

