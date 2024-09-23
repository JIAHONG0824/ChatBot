import streamlit as st
from openai import OpenAI
import os
from dotenv import load_dotenv
import base64
from document_relevant import relevant

load_dotenv()

# Initialize the session state
if "model" not in st.session_state:
    st.session_state["model"] = "gpt-4o-mini-2024-07-18"
if "client" not in st.session_state:
    st.session_state["client"] = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY")
    )
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "images" not in st.session_state:
    st.session_state["images"] = None
if "context" not in st.session_state:
    st.session_state["context"] = []
# get response from the model
def invoke(query:str) -> str:
    history = st.session_state["messages"][:-1]
    query = st.session_state["messages"][-1]["content"]
    if st.session_state["images"] is not None:
        query = query
        base64_string=base64.b64encode(st.session_state["images"].getvalue()).decode('utf-8')
        history.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"{query}"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_string}"
                        }
                    }
                ]
            }
        )
        stream = st.session_state["client"].chat.completions.create(
            model=st.session_state["model"],
            messages=history,
            stream=True
        )
        return st.write_stream(stream)
    if st.session_state["context"]:
        history.append(
            {
                "role": "system","content":"""
                請根據以下的 context 內容回答使用者的問題。
                如果 context 中包含與問題相關的資訊，請根據這些資訊提供答案；
                如果 context 中沒有相關內容，請正常回答問題，不要依賴 context 內容。
                請嚴格遵守這個指示。
                """
            }
        )
        history.append(
            {
                "role": "user",
                "content": f"context: {"\n".join(st.session_state['context'])}\n\n 使用者問題: {query}"
            }
        )
        stream = st.session_state["client"].chat.completions.create(
        model=st.session_state["model"],
        messages=history,
        stream=True
        )
        return st.write_stream(stream)
    stream = st.session_state["client"].chat.completions.create(
        model=st.session_state["model"],
        messages=st.session_state["messages"],
        stream=True
    )
    return st.write_stream(stream)

st.title("Chatbot")

MODEL_OPTIONS = ["gpt-4o-mini-2024-07-18"]
with st.sidebar:
    st.session_state["model"] = st.selectbox("Select Model", MODEL_OPTIONS)
    st.subheader(f"Model: {st.session_state['model']} is selected")
    st.session_state["images"]=st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
    
# Display the chat history
for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
# Process the user input
if query:=st.chat_input():
    with st.chat_message("user"):
        st.markdown(query)
    st.session_state["messages"].append({"role": "user", "content":query})
    st.session_state["context"]=relevant(query)
    # Get the response from the chatbot
    with st.chat_message("assistant"):
        with st.status("Thinking...",expanded=True):
            print(query)
            response = invoke(query)
            st.session_state["messages"].append(
                {"role": "assistant", "content":response}
                )
