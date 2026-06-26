import os
import psycopg2
from psycopg2.extras import RealDictCursor

DSN = {
    "host": os.getenv("POSTGRES_HOST", "localhost"),
    "port": int(os.getenv("POSTGRES_PORT", "5432")),
    "dbname": os.getenv("POSTGRES_DB", "bd2_proyecto2"),
    "user": os.getenv("POSTGRES_USER", "bd2"),
    "password": os.getenv("POSTGRES_PASSWORD", "bd2"),
}


def get_conn():
    return psycopg2.connect(**DSN, cursor_factory=RealDictCursor)


def vec_to_pg(arr) -> str:
    return "[" + ",".join(f"{float(x):.6f}" for x in arr) + "]"
