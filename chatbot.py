import streamlit as st
from openai import OpenAI
import os
from dotenv import load_dotenv
import base64
from relevant_document import relevant
from evaluator import is_retrieval_needed

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
if "documens" not in st.session_state:
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
        template=[
            {
                "role": "system",
                "content": """
                根據使用者的問題，參考 context 中的內容回答。如果 context 中的資訊與問題不匹配或提供的資訊不同，請明確指出不一致的地方。如果 context 中沒有資料，回答「我需要求助外部資訊，才有辦法回答您」。請嚴格遵守指示，避免引入錯誤資訊。
                """ 
            },
            {
                "role": "user",
                "content": f"context:\n{"\n\n".join(st.session_state['context'])}\n\n 使用者問題:\n{query}"
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
                print("Retrieval is needed")
                st.session_state["retrieval"]=True
                st.session_state["context"]=relevant(query)
            else:
                print("Retrieval is not needed")
                st.session_state["retrieval"]=False
            response = invoke(query)
