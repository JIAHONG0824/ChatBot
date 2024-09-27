from openai import OpenAI
import os
from dotenv import load_dotenv
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def is_retrieval_needed(query:str):
    completion = client.chat.completions.create(
        model="gpt-4o-mini-2024-07-18",
        messages=[
            {
                "role": "system",
                "content":"""
                請你對於使用者的問題判斷是否需要進行檢索(透過外部資源)，如果檢索可以讓答案更好的話，請給出"需要"或"不需要"。
                """
            },
            {
                "role": "user","content":query
            }
        ],
        temperature=0
    )
    return completion.choices[0].message.content

# completion = client.chat.completions.create(
#     model="gpt-4o-mini",
#     messages=[
#         {
#             "role": "system","content":"""
#             請你對於使用者的問題判斷是否需要進行檢索(透過外部資源)，如果檢索可以讓答案更好的話，請給出"需要"或"不需要"並給出你的解釋。
#             請一步一步思考，
#             解釋: [你的解釋]\n
#             需要檢索?: [需要/不需要]\n
#             最後回答: [需要/不需要]即可
#             """
#         },
#         {
#             "role": "user","content":"台灣2016~2020總統是?"
#         }
#     ]
# )

# print(completion.choices[0].message.content)