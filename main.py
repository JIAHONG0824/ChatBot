from langchain.text_splitter import RecursiveCharacterTextSplitter
#from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


documents=TextLoader(file_path="文本.txt",encoding="utf-8")
splitter=RecursiveCharacterTextSplitter(chunk_size=200,chunk_overlap=0)
documents=documents.load_and_split(splitter)
embeddings=OpenAIEmbeddings()
vectorstore=Chroma().from_documents(documents,embeddings,persist_directory="C:/Users/user no1/Desktop/ChatBot/",collection_name="rag-chroma")
retriever=vectorstore.as_retriever(search_kwargs={"k":2})
print(retriever.invoke("艾爾文的父親是探險家嗎?"))