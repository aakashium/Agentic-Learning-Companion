import streamlit as st
import requests

API = "http://127.0.0.1:8000/ask"

st.set_page_config(page_title="📘 Chat with Documents", page_icon="💬")

st.title("💬 Chat with your Documents")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display existing messages
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.chat_message("user").write(msg["content"])
    else:
        st.chat_message("assistant").write(msg["content"])


# User input box
user_q = st.chat_input("Ask a question from your documents...")

if user_q:
    # Save user message
    st.session_state.messages.append({"role": "user", "content": user_q})
    st.chat_message("user").write(user_q)

    # Backend call
    with st.spinner("Thinking..."):
        try:
            res = requests.get(API, params={"q": user_q}, timeout=20).json()
            answer = res.get("answer", "No answer returned.")
        except Exception as e:
            answer = f"Error: {e}"

    # Save assistant answer
    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.chat_message("assistant").write(answer)
