import sqlite3
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
DB_PATH = BASE_DIR / "ecommerce.db"
SQL_SEED_PATH = BASE_DIR / "ecommerce_setup.sql"

def init_database():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    sql_text = SQL_SEED_PATH.read_text(encoding="utf-8")
    with sqlite3.connect(DB_PATH) as conn:
        conn.executescript(sql_text)
        conn.commit

    return DB_PATH


def main():
    db_path = init_database()
    print(f"Database created at: {db_path}")


if __name__ == "__main__":
    main()