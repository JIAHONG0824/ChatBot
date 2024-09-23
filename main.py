from langchain.text_splitter import RecursiveCharacterTextSplitter
#from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

vectorstore=Chroma(persist_directory="C:/Users/NB/Desktop/venv",embedding_function=OpenAIEmbeddings(),collection_name="rag-chroma")
retriever = vectorstore.as_retriever(
    search_kwargs={"k":3}
)
### Retrieval Grader


from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
# NOTE: you must use langchain-core >= 0.3 with Pydantic v2
from pydantic import BaseModel, Field


# Data model
class GradeDocuments(BaseModel):
    """檢索文件相關性檢查的二元分數。"""

    binary_score: str = Field(
        description="文件與問題相關：'是' 或 '否'"
    )


# LLM with function call
llm = ChatOpenAI(model="gpt-4o-mini-2024-07-18", temperature=0)
structured_llm_grader = llm.with_structured_output(GradeDocuments)

# Prompt
system = """你是一名評估文件與用戶問題相關性的評分員。
不需要進行嚴格的測試，目的是過濾掉錯誤的檢索結果。
如果文件包含與用戶問題相關的關鍵字或語義，請將其評為相關。
使用二元分數「是」或「否」來表示文件是否與問題相關。"""
grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "檢索到的文件：\n\n {document} \n\n 用戶問題：{question}"),
    ]
)
retrieval_grader = grade_prompt | structured_llm_grader
question = "今天午餐吃什麼?"
docs = retriever.invoke(question)
### Generate

from langchain import hub
from langchain_core.output_parsers import StrOutputParser

# Prompt
prompt = hub.pull("rlm/rag-prompt")
print(type(prompt))
llm = ChatOpenAI(model_name="gpt-4o-mini-2024-07-18", temperature=0)


# # Post-processing
# def format_docs(docs):
#     return "\n\n".join(doc.page_content for doc in docs)


# # Chain
# rag_chain = prompt | llm | StrOutputParser()

# # Run
# generation = rag_chain.invoke({"context": docs, "question": question})
# print(generation)