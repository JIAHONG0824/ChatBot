import streamlit as st
from openai import OpenAI
import os
from dotenv import load_dotenv
import base64

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
# get response from the model
def invoke() -> str:
    messages = st.session_state["messages"]
    if st.session_state["images"] is not None:
        query = messages[-1]["content"]
        base64_string=base64.b64encode(st.session_state["images"].getvalue()).decode('utf-8')
        messages.append(
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
        messages=messages,
        stream=True
    )
    if st.session_state["images"] is not None:
        st.session_state["messages"].pop()
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
    st.session_state["messages"].append({"role": "user", "content":query})
    with st.chat_message("user"):
        st.markdown(query)
    # Get the response from the chatbot
    with st.chat_message("assistant"):
        response = invoke()
        st.session_state["messages"].append(
            {"role": "assistant", "content":response}
            )
        print(st.session_state["messages"])
