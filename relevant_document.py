from pinecone import Pinecone
import os
from dotenv import load_dotenv
from openai import OpenAI
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY")) 
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index=pc.Index("rag")
# 回傳有相關的文件
def relevant(query:str):
    query="艾爾文的父親是探險家嗎?"
    embedding = pc.inference.embed(
        model="multilingual-e5-large",
        inputs=[query],
        parameters={
            "input_type": "query"
        }
    )
    results = index.query(
        namespace="ns1",
        vector=embedding[0].values,
        top_k=3,
        include_values=False,
        include_metadata=True
    )
    documents = []
    for i in range(len(results["matches"])):
        documents.append(results["matches"][i]["metadata"]["text"])
    relevant_documents = []
    for document in documents:
        response = client.chat.completions.create(
            model="gpt-4o-mini-2024-07-18",
            messages=[
                {
                    "role": "system", "content": """
                    Given a question, does the following document have exact information to answer the question? Answer "yes" or "no" only.\n
                    """
                },
                {
                    "role":"user","content":f"Retrieved document:\n{document}\n\nUser question:\n{query}\n\n"
                }
            ],
            temperature=0
        )
        if response.choices[0].message.content == "yes":
            relevant_documents.append(document)
    print(relevant_documents)
    for i in range(len(relevant_documents)):
        response = client.chat.completions.create(
            model="gpt-4o-mini-2024-07-18",
            messages=[
                {
                    "role": "system",
                    "content": """
                    You will given a documents and a question. You need to
                    preserve the relevant information in the document based on the user's question.
                    and remove the irrelevant information.
                    """
                },
                {
                    "role":"user",
                    "content": f"Document: \n\n {relevant_documents[i]} \n\n User question: {query}"
                }
            ],
            temperature=0
        )
        relevant_documents[i] = response.choices[0].message.content
    print(relevant_documents)
    return relevant_documents