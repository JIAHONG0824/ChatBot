import streamlit as st
from openai import OpenAI
import os
from dotenv import load_dotenv
import base64
from document_relevant import relevant
from retrieval import is_retrieval_needed

load_dotenv()

# Initialize the session state
if "model" not in st.session_state:
    st.session_state["model"] = "gpt-4o-mini-2024-07-18"
if "client" not in st.session_state:
    st.session_state["client"]= OpenAI(
        api_key=os.getenv("OPENAI_API_KEY")
    )
if "images" not in st.session_state:
    st.session_state["images"] = None
if "context" not in st.session_state:
    st.session_state["context"] = []
if "retrieval" not in st.session_state:
    st.session_state["retrieval"] = False
# get response from the model
def invoke(query) -> str:
    template = [
        {
            "role":"user",
            "content": query
        }
    ]
    if st.session_state["retrieval"]:
        print("Retrieval is needed")
        print(st.session_state["context"])
        template=[
            {
                "role": "system",
                "content": """
                請根據以下的 context 內容回答使用者的問題。
                如果 context 中包含與問題相關的資訊，請根據這些資訊提供答案；
                如果 context 中沒有相關內容，請回答「沒有足夠的資訊，無法回答」。
                請嚴格遵守這個指示。
                """ 
            },
            {
                "role": "user",
                "content": f"context: {"\n".join(st.session_state['context'])}\n\n 使用者問題: {query}"
            }
        ]
    stream = st.session_state["client"].chat.completions.create(
        model=st.session_state["model"],
        messages=template,
        stream=True
    )
    return st.write_stream(stream)

st.title("Chatbot")

MODEL_OPTIONS = ["gpt-4o-mini-2024-07-18"]
with st.sidebar:
    st.session_state["model"] = st.selectbox("Select Model", MODEL_OPTIONS)
    st.subheader(f"Model: {st.session_state['model']} is selected")
    st.session_state["images"]=st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
    
# Process the user input
if query:=st.chat_input():
    with st.chat_message("user"):
        st.markdown(query)
    with st.chat_message("assistant"):
        with st.status("Thinking...",expanded=True):
            if is_retrieval_needed(query)== "需要":
                st.session_state["retrieval"]=True
                st.session_state["context"]=relevant(query)
            else:
                st.session_state["retrieval"]=False
            response = invoke(query)
