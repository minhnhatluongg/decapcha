"""
keystore.py - Quản lý API key + log gọi API bằng SQLite (stdlib, không cần dependency).

Bảng:
  api_keys : key, name, quota (NULL = unlimited), used, active, note, created_at
  call_logs: key_id, key_name, mst, status, count_try, ip, created_at

Dùng cho endpoint /tcnnt/lookup (auth + đếm lượt) và admin dashboard.
Log tự dọn sau LOG_RETENTION_DAYS ngày.
"""
import sqlite3
import secrets
import threading
from datetime import datetime, timedelta

import config

_lock = threading.Lock()


def _conn() -> sqlite3.Connection:
    c = sqlite3.connect(config.KEYS_DB_PATH, timeout=10)
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA journal_mode=WAL")
    c.execute("PRAGMA busy_timeout=5000")
    return c


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def init_db() -> None:
    with _lock, _conn() as c:
        c.execute("""
            CREATE TABLE IF NOT EXISTS api_keys (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                key        TEXT UNIQUE NOT NULL,
                name       TEXT NOT NULL DEFAULT '',
                quota      INTEGER,                 -- NULL = unlimited
                used       INTEGER NOT NULL DEFAULT 0,
                active     INTEGER NOT NULL DEFAULT 1,
                note       TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS call_logs (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                key_id     INTEGER,
                key_name   TEXT,
                mst        TEXT,
                status     TEXT,
                count_try  INTEGER,
                ip         TEXT,
                created_at TEXT NOT NULL
            )
        """)
        c.execute("CREATE INDEX IF NOT EXISTS idx_logs_created ON call_logs(created_at)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_logs_key ON call_logs(key_id)")
        c.execute("""
            CREATE TABLE IF NOT EXISTS mst_cache (
                mst        TEXT PRIMARY KEY,
                result     TEXT NOT NULL,       -- JSON kết quả lookup (found/no_result)
                status     TEXT,
                updated_at TEXT NOT NULL         -- ISO datetime lúc cache
            )
        """)


# ===== API key CRUD =====

def create_key(name: str, quota, note: str = "") -> dict:
    """quota = số lượt cho phép, None/<=0 -> unlimited (lưu NULL)."""
    q = None if (quota is None or int(quota) <= 0) else int(quota)
    key = "dk_" + secrets.token_urlsafe(24)
    with _lock, _conn() as c:
        c.execute(
            "INSERT INTO api_keys(key,name,quota,used,active,note,created_at) VALUES(?,?,?,0,1,?,?)",
            (key, name.strip(), q, note.strip(), _now()),
        )
    return get_key(key)


def get_key(key: str):
    with _conn() as c:
        r = c.execute("SELECT * FROM api_keys WHERE key=?", (key,)).fetchone()
    return dict(r) if r else None


def get_key_by_id(key_id: int):
    with _conn() as c:
        r = c.execute("SELECT * FROM api_keys WHERE id=?", (key_id,)).fetchone()
    return dict(r) if r else None


def list_keys() -> list[dict]:
    with _conn() as c:
        rows = c.execute("SELECT * FROM api_keys ORDER BY id DESC").fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["used_today"] = c.execute(
                "SELECT COUNT(*) FROM call_logs WHERE key_id=? AND date(created_at)=date('now','localtime')",
                (d["id"],),
            ).fetchone()[0]
            out.append(d)
    return out


def update_key(key_id: int, name=None, quota="__keep__", active=None, note=None) -> None:
    sets, vals = [], []
    if name is not None:
        sets.append("name=?"); vals.append(name.strip())
    if quota != "__keep__":
        q = None if (quota is None or int(quota) <= 0) else int(quota)
        sets.append("quota=?"); vals.append(q)
    if active is not None:
        sets.append("active=?"); vals.append(1 if active else 0)
    if note is not None:
        sets.append("note=?"); vals.append(note.strip())
    if not sets:
        return
    vals.append(key_id)
    with _lock, _conn() as c:
        c.execute(f"UPDATE api_keys SET {', '.join(sets)} WHERE id=?", vals)


def reset_used(key_id: int) -> None:
    with _lock, _conn() as c:
        c.execute("UPDATE api_keys SET used=0 WHERE id=?", (key_id,))


def delete_key(key_id: int) -> None:
    with _lock, _conn() as c:
        c.execute("DELETE FROM api_keys WHERE id=?", (key_id,))
        c.execute("DELETE FROM call_logs WHERE key_id=?", (key_id,))


# ===== Quota consume (atomic) =====

def consume(key: str):
    """
    Trừ 1 lượt nếu key active và còn quota (hoặc unlimited).
    Returns: (status, record)
      status: 'ok' | 'not_found' | 'inactive' | 'quota_exceeded'
    """
    with _lock, _conn() as c:
        r = c.execute("SELECT * FROM api_keys WHERE key=?", (key,)).fetchone()
        if not r:
            return "not_found", None
        if not r["active"]:
            return "inactive", dict(r)
        cur = c.execute(
            "UPDATE api_keys SET used=used+1 WHERE key=? AND active=1 AND (quota IS NULL OR used<quota)",
            (key,),
        )
        if cur.rowcount == 0:
            return "quota_exceeded", dict(r)
        r2 = c.execute("SELECT * FROM api_keys WHERE key=?", (key,)).fetchone()
        return "ok", dict(r2)


# ===== Logs =====

def log_call(key_id, key_name, mst, status, count_try, ip) -> None:
    with _lock, _conn() as c:
        c.execute(
            "INSERT INTO call_logs(key_id,key_name,mst,status,count_try,ip,created_at) VALUES(?,?,?,?,?,?,?)",
            (key_id, key_name, mst, status, count_try, ip, _now()),
        )


def recent_logs(limit: int = 200, key_id=None) -> list[dict]:
    with _conn() as c:
        if key_id is not None:
            rows = c.execute(
                "SELECT * FROM call_logs WHERE key_id=? ORDER BY id DESC LIMIT ?",
                (key_id, limit),
            ).fetchall()
        else:
            rows = c.execute(
                "SELECT * FROM call_logs ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
    return [dict(r) for r in rows]


def cleanup_logs(days: int) -> int:
    """Xoá log cũ hơn `days` ngày. Trả về số dòng đã xoá."""
    cutoff = (datetime.now() - timedelta(days=days)).isoformat(timespec="seconds")
    with _lock, _conn() as c:
        cur = c.execute("DELETE FROM call_logs WHERE created_at < ?", (cutoff,))
        return cur.rowcount


# ===== Cache MST (giảm gọi TCT) =====

def cache_get(mst: str, ttl_seconds: int):
    """Trả JSON kết quả đã cache nếu còn hạn (trong ttl_seconds). None nếu không có/hết hạn."""
    if not mst or ttl_seconds <= 0:
        return None
    cutoff = (datetime.now() - timedelta(seconds=ttl_seconds)).isoformat(timespec="seconds")
    with _conn() as c:
        r = c.execute(
            "SELECT result FROM mst_cache WHERE mst=? AND updated_at>=?", (mst, cutoff)
        ).fetchone()
    return r["result"] if r else None


def cache_put(mst: str, result_json: str, status: str) -> None:
    """Lưu/ghi đè kết quả cache cho 1 MST."""
    if not mst:
        return
    with _lock, _conn() as c:
        c.execute(
            "INSERT INTO mst_cache(mst,result,status,updated_at) VALUES(?,?,?,?) "
            "ON CONFLICT(mst) DO UPDATE SET result=excluded.result, "
            "status=excluded.status, updated_at=excluded.updated_at",
            (mst, result_json, status, _now()),
        )


def cache_stats() -> dict:
    with _conn() as c:
        total = c.execute("SELECT COUNT(*) FROM mst_cache").fetchone()[0]
    return {"cached_mst": total}
