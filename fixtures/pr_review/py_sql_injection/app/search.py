from flask import request


def search_users(conn):
    query = request.args.get("q", "")
    sql = f"SELECT * FROM users WHERE name LIKE '%{query}%'"
    return conn.execute(sql).fetchall()
