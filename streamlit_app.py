import os
import mimetypes
import requests
import streamlit as st

API_BASE = "http://127.0.0.1:9000"

st.set_page_config(page_title="Chatbot", layout="wide")

def api_get(path):
    r = requests.get(f"{API_BASE}{path}", timeout=60)
    if not r.ok:
        raise RuntimeError(f"{r.status_code} {r.reason}\n{r.text}")
    return r.json()

def api_post(path, payload=None, files=None, timeout=180):
    r = requests.post(f"{API_BASE}{path}", json=payload, files=files, timeout=timeout)
    if not r.ok:
        raise RuntimeError(f"{r.status_code} {r.reason}\n{r.text}")
    return r.json()

def list_sessions():
    return api_get("/sessions")

def delete_session_api(session_id):
    r = requests.delete(f"{API_BASE}/sessions/{session_id}", timeout=60)
    if not r.ok:
        raise RuntimeError(f"{r.status_code} {r.reason}\n{r.text}")
    return r.json()

def get_messages(session_id, limit=200):
    return api_get(f"/sessions/{session_id}/messages?limit={limit}")

def new_session():
    return api_post("/sessions")

def ask(session_id, question, k=5):
    return api_post(f"/sessions/{session_id}/ask", {"question": question, "k": k}, timeout=180)

def reindex():
    return api_post("/reindex", timeout=300)

def upload_to_kb(uploaded_files):
    files = []
    for uf in uploaded_files:
        content = uf.getvalue()
        mime = mimetypes.guess_type(uf.name)[0] or "application/octet-stream"
        files.append(("files", (uf.name, content, mime)))
    return api_post("/kb/upload", files=files, timeout=300)

st.sidebar.title("Chatbot")

st.sidebar.subheader("Knowledge Base")
uploaded = st.sidebar.file_uploader(
    "Upload docs (.md/.txt/.pdf/.docx/.csv)",
    type=["md", "txt", "pdf", "docx", "csv"],
    accept_multiple_files=True,
)

if st.sidebar.button("Upload & Reindex", disabled=not uploaded):
    with st.spinner("Uploading and reindexing..."):
        out = upload_to_kb(uploaded)
        st.sidebar.success(f"Uploaded {len(out.get('saved', []))} file(s).")
        st.sidebar.caption(f"Reindex: docs={out['reindex']['docs']} chunks={out['reindex']['chunks']}")

if st.sidebar.button("Reindex KB"):
    with st.spinner("Reindexing..."):
        out = reindex()
        st.sidebar.success(f"docs={out['docs']} chunks={out['chunks']}")

st.sidebar.subheader("Sessions")
if st.sidebar.button("New chat"):
    s = new_session()
    st.session_state["session_id"] = s["session_id"]

sessions = list_sessions()
session_labels = {s["session_id"]: s["title"] for s in sessions}

default_sid = st.session_state.get("session_id")
if not default_sid and sessions:
    default_sid = sessions[0]["session_id"]
    st.session_state["session_id"] = default_sid

sid = st.sidebar.selectbox(
    "Choose session",
    options=[s["session_id"] for s in sessions] if sessions else [],
    format_func=lambda x: f"{session_labels.get(x, x)} ({x[:8]})",
    index=0 if sessions else None,
)

if sid:
    st.session_state["session_id"] = sid

if sid:
    prev = st.session_state.get("last_session_id")
    if prev != sid:
        try:
            data = get_messages(sid, limit=200)
            msgs = data.get("messages", [])
            st.session_state["chat"] = [(m["role"], m["content"]) for m in msgs]
        except Exception as e:
            st.sidebar.error(f"Failed to load history: {e}")
            st.session_state["chat"] = []
        st.session_state["last_session_id"] = sid

if st.sidebar.button("Delete this chat", disabled=not sid):
    with st.spinner("Deleting..."):
        try:
            delete_session_api(sid)
            st.sidebar.success("Deleted.")

            st.session_state["chat"] = []

            sessions2 = list_sessions()
            if sessions2:
                st.session_state["session_id"] = sessions2[0]["session_id"]
            else:
                snew = new_session()
                st.session_state["session_id"] = snew["session_id"]
            st.session_state["last_session_id"] = None
            st.rerun()
        except Exception as e:
            st.sidebar.error(str(e))

st.title("Chat")

if "chat" not in st.session_state:
    st.session_state["chat"] = []

for role, content in st.session_state["chat"]:
    with st.chat_message(role):
        st.markdown(content)

q = st.chat_input("Ask something...")
if q and st.session_state.get("session_id"):
    session_id = st.session_state["session_id"]

    st.session_state["chat"].append(("user", q))
    with st.chat_message("user"):
        st.markdown(q)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                out = ask(session_id, q, k=5)
            except Exception as e:
                st.error(str(e))
                st.stop()

            ans = (out.get("answer") or "").strip()
            srcs = out.get("sources") or []

            st.markdown(ans if ans else "_(empty answer)_")

            if srcs:
                st.markdown("**Sources (KB):**")
                for s in srcs:
                    st.caption(f"- {s.get('source')}")


    st.session_state["chat"].append(("assistant", ans))
    try:
        data = get_messages(session_id, limit=200)
        msgs = data.get("messages", [])
        st.session_state["chat"] = [(m["role"], m["content"]) for m in msgs]
    except Exception:
        pass

