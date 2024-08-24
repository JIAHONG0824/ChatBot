import streamlit as st
from ollama import Client
import time

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
main = st.container()
# Display the chat history
with main:
    st.title("llama3.1:8b chatbot🦙")
    # st.session_state.history[1:] prevents the first message from being displayed
    for message in st.session_state.history[1:]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
query = st.chat_input("傳訊息給 llama🦙")
# Process the user input
with main:
    if query:
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
        st.rerun()
