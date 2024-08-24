import streamlit as st
import time
from ollama import Client

client = Client()


# get response from llama3.1:8b model
def invoke(query: str, history: list[dict]) -> str:
    response = client.chat(model="llama3.1:8b", messages=history)
    return response["message"]["content"]


# streaming response
def streaming_resopnse(resopnse: str):
    for word in resopnse:
        yield word
        time.sleep(0.05)


# Initialize the chat history
if "history" not in st.session_state:
    st.session_state.history = [
        {"role": "system", "content": "回答時請使用#zh-TW"},
    ]
if "query" not in st.session_state:
    st.session_state.user_input = ""
if "processing" not in st.session_state:
    st.session_state.processing = False

st.title("llama3.1:8b chatbot🦙")
# Display the chat history
for message in st.session_state.history[1:]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if not st.session_state["processing"]:
    input = st.chat_input("傳訊息給 llama🦙")
    if input:
        st.session_state["query"] = input
        st.session_state["processing"] = True
        st.rerun()
# Process the user input
if st.session_state["processing"]:
    query = st.session_state["query"]
    # Display the chat input not editable
    st.chat_input("傳訊息給 llama🦙", disabled=True)
    # Display the user input
    with st.chat_message("user"):
        st.markdown(query)
        st.session_state["history"].append({"role": "user", "content": query})
    # Get the response from the chatbot
    with st.chat_message("assistant"):
        with st.spinner("Processing..."):
            response = invoke(query, st.session_state["history"])
            st.markdown(st.write_stream(streaming_resopnse(response)))
            st.session_state["history"].append(
                {"role": "assistant", "content": response}
            )
    st.session_state["query"] = ""
    st.session_state["processing"] = False
    st.rerun()
