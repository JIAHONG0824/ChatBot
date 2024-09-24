from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI
import os
from dotenv import load_dotenv
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

vectorstore=Chroma(
    persist_directory="C:/Users/NB/Desktop/venv",
    embedding_function=OpenAIEmbeddings(),
    collection_name="rag-chroma"
)
retriever = vectorstore.as_retriever(search_kwargs={"k":5})

def relevant(query:str):
    relevant_documents = []
    documents = retriever.invoke(query)
    for document in documents:
        response = client.chat.completions.create(
            model="gpt-4o-mini-2024-07-18",
            messages=[
                {
                    "role": "system", "content": """
                    You are a grader assessing relevance of a retrieved document to a user question. \n 
    It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
    If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.
                    """
                },
                {
                    "role":"user","content":f"Retrieved document: \n\n {document.page_content} \n\n User question: {query}"
                }
            ],
            temperature=0.7
        )
        if response.choices[0].message.content == "yes":
            relevant_documents.append(document.page_content)
    return relevant_documents