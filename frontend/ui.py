import streamlit as st
import requests
import json
import time
import uuid
from datetime import datetime
from pathlib import Path

# --- Configuration ---
API_URL = "http://127.0.0.1:8000"
HISTORY_FILE = Path("frontend/chat_history.json")

st.set_page_config(
    page_title="AI Studio RAG",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS (Google AI Studio Vibe) ---
st.markdown("""
<style>
    /* General Reset */
    .stApp {
        background-color: #0e1117; /* Dark theme base */
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #30363d;
    }
    
    /* Chat Message Styling */
    .stChatMessage {
        background-color: transparent;
        border: none;
    }
    .stChatMessage[data-testid="stChatMessageUser"] {
        background-color: #1e232a;
        border-radius: 10px;
    }
    
    /* Input Area Styling */
    .stChatInputContainer {
        padding-bottom: 20px;
    }
    
    /* Button Styling */
    .stButton button {
        border-radius: 8px;
        border: 1px solid #30363d;
        background-color: #21262d;
        color: #c9d1d9;
        transition: all 0.2s;
    }
    .stButton button:hover {
        border-color: #8b949e;
        background-color: #30363d;
    }
    
    /* Hide Streamlit Elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    /* header {visibility: hidden;}  <-- Commented out to allow sidebar toggle */
</style>
""", unsafe_allow_html=True)

# --- Persistence Functions ---

def load_history():
    """Load chat history from JSON file."""
    if HISTORY_FILE.exists():
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            return {}
    return {}

def save_history():
    """Save current session state history to JSON file."""
    # Convert session state to serializable format
    # We only save the 'sessions' dictionary
    if "sessions" in st.session_state:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(st.session_state.sessions, f, indent=2, ensure_ascii=False)

# --- State Management ---

if "sessions" not in st.session_state:
    st.session_state.sessions = load_history()

if "current_session_id" not in st.session_state:
    # If history exists, pick the most recent one, else create new
    if st.session_state.sessions:
        # Sort by timestamp descending
        sorted_sessions = sorted(
            st.session_state.sessions.items(), 
            key=lambda x: x[1].get("timestamp", 0), 
            reverse=True
        )
        st.session_state.current_session_id = sorted_sessions[0][0]
    else:
        st.session_state.current_session_id = None

def create_new_session():
    """Create a new chat session."""
    session_id = str(uuid.uuid4())
    st.session_state.sessions[session_id] = {
        "title": "New Conversation",
        "messages": [],
        "doc_id": None,
        "timestamp": time.time()
    }
    st.session_state.current_session_id = session_id
    save_history()
    return session_id

def switch_session(session_id):
    """Switch to an existing session."""
    st.session_state.current_session_id = session_id

def delete_session(session_id):
    """Delete a session."""
    if session_id in st.session_state.sessions:
        del st.session_state.sessions[session_id]
        save_history()
        # If we deleted the current session, switch to another or create new
        if st.session_state.current_session_id == session_id:
            if st.session_state.sessions:
                st.session_state.current_session_id = list(st.session_state.sessions.keys())[0]
            else:
                create_new_session()
        st.rerun()

# Ensure we have at least one session
if st.session_state.current_session_id is None or st.session_state.current_session_id not in st.session_state.sessions:
    create_new_session()

# Get current session data
current_session = st.session_state.sessions[st.session_state.current_session_id]

# --- Helper Functions ---

def stream_text(text):
    """Generator for typewriter effect."""
    for word in text.split(" "):
        yield word + " "
        time.sleep(0.02)

# --- Sidebar (Navigation) ---

with st.sidebar:
    # 1. New Chat Button
    if st.button("➕ New Chat", use_container_width=True):
        create_new_session()
        st.rerun()
    
    st.divider()
    
    # 2. History List
    st.markdown("### 🕒 Recent")
    
    # Sort sessions by timestamp (newest first)
    sorted_sessions = sorted(
        st.session_state.sessions.items(), 
        key=lambda x: x[1].get("timestamp", 0), 
        reverse=True
    )
    
    for s_id, s_data in sorted_sessions:
        col1, col2 = st.columns([0.85, 0.15])
        with col1:
            # Truncate title if too long
            title = s_data.get("title", "Untitled")
            if len(title) > 22:
                title = title[:20] + "..."
                
            # Highlight active session
            type_ = "primary" if s_id == st.session_state.current_session_id else "secondary"
            if st.button(f"💬 {title}", key=f"btn_{s_id}", use_container_width=True, type=type_):
                switch_session(s_id)
                st.rerun()
        with col2:
            if st.button("🗑️", key=f"del_{s_id}", help="Delete Chat"):
                delete_session(s_id)

    st.divider()
    
    # 3. File Uploader (Context Aware)
    st.markdown("### 📁 Data Context")
    
    if current_session["doc_id"]:
        st.success(f"Linked Doc: `{current_session['doc_id'][:8]}...`")
        if st.button("Unlink Document", use_container_width=True):
            current_session["doc_id"] = None
            save_history()
            st.rerun()
    else:
        uploaded_file = st.file_uploader("Attach PDF", type="pdf", label_visibility="collapsed")
        if uploaded_file:
            with st.status("Processing...", expanded=True) as status:
                try:
                    files = {"file": (uploaded_file.name, uploaded_file, "application/pdf")}
                    response = requests.post(f"{API_URL}/upload-pdf", files=files)
                    
                    if response.status_code == 200:
                        data = response.json()
                        doc_id = data["doc_id"]
                        
                        # Update Session
                        current_session["doc_id"] = doc_id
                        current_session["title"] = uploaded_file.name # Rename chat to filename
                        save_history()
                        
                        status.update(label="Ready!", state="complete", expanded=False)
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error(f"Upload failed: {response.text}")
                except Exception as e:
                    st.error(f"Connection error: {e}")

# --- Main Chat Area ---

# Header
st.markdown(f"## {current_session.get('title', 'New Conversation')}")
st.caption(f"Session ID: {st.session_state.current_session_id}")

# Display Messages
for msg in current_session["messages"]:
    avatar = "✨" if msg["role"] == "assistant" else None
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])
        # Show sources if available (stored in message metadata if we had it, 
        # but for now we just render content. You could enhance this to store sources in history)

# Chat Input
if prompt := st.chat_input("Type your message..."):
    # 1. Check if we have a document
    if not current_session["doc_id"]:
        st.warning("⚠️ Please upload a document in the sidebar first!")
        st.stop()

    # 2. Append User Message
    current_session["messages"].append({"role": "user", "content": prompt})
    current_session["timestamp"] = time.time() # Update timestamp
    
    # If it's the first message and title is generic, update title
    if current_session["title"] == "New Conversation":
        current_session["title"] = prompt[:30] + "..."
        
    save_history()
    
    with st.chat_message("user"):
        st.markdown(prompt)

    # 3. Generate Response
    with st.chat_message("assistant", avatar="✨"):
        with st.spinner("Generating response..."):
            try:
                # Prepare payload
                history_payload = [
                    {"role": m["role"], "content": m["content"]}
                    for m in current_session["messages"][:-1]
                ]
                
                payload = {
                    "doc_id": current_session["doc_id"],
                    "question": prompt,
                    "history": history_payload
                }
                
                response = requests.post(f"{API_URL}/chat", json=payload)
                
                if response.status_code == 200:
                    data = response.json()
                    answer = data["answer"]
                    
                    # Stream output
                    st.write_stream(stream_text(answer))
                    
                    # Show Sources
                    if data.get("sources"):
                        with st.expander("📚 Sources"):
                            for source in data["sources"]:
                                page = source.get('page_number', 'N/A')
                                snippet = source.get('snippet', '')
                                st.markdown(f"**Page {page}**")
                                st.caption(f"> {snippet}")
                    
                    # Append Assistant Message
                    current_session["messages"].append({"role": "assistant", "content": answer})
                    save_history()
                    
                else:
                    st.error(f"API Error: {response.text}")
                    
            except Exception as e:
                st.error(f"Connection failed: {e}")
