import uuid
from redis import Redis

_SESSIONS_KEY = "sessions:index"
_TITLE_KEY_PREFIX = "sessions:title:"

def _title_key(session_id):
    return f"{_TITLE_KEY_PREFIX}{session_id}"

def create_session(r: Redis, title="New chat"):
    sid = str(uuid.uuid4())
    r.set(_title_key(sid), title)
    r.zadd(_SESSIONS_KEY, {sid: float(r.time()[0])})
    return {"session_id": sid, "title": title}

def list_sessions(r: Redis, limit=50):
    sids = r.zrevrange(_SESSIONS_KEY, 0, limit - 1)
    out = []
    for b in sids:
        sid = b.decode("utf-8")
        title = (r.get(_title_key(sid)) or b"New chat").decode("utf-8")
        out.append({"session_id": sid, "title": title})
    return out

def get_title(r: Redis, session_id):
    val = r.get(_title_key(session_id))
    return val.decode("utf-8") if val else "New chat"

def set_title(r: Redis, session_id, title):
    r.set(_title_key(session_id), title)
    r.zadd(_SESSIONS_KEY, {session_id: float(r.time()[0])})

def delete_session(r: Redis, session_id):
    r.zrem(_SESSIONS_KEY, session_id)
    r.delete(_title_key(session_id))

