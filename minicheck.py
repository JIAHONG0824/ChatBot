from openai import OpenAI

client = OpenAI(
    base_url = 'http://localhost:11434/v1',
    api_key='ollama', # required, but unused
)

response = client.chat.completions.create(
    model="bespoke-minicheck:latest",
    messages=[
        {
            "role":"user",
            "content":"""
            Document:賴清德於2023年4月12日，獲得民進黨提名參選2024年中華民國總統選舉；11月20日，與副手蕭美琴前往中央選舉委員會登記。2024年1月13日，以40.05%的得票率當選總統，是中華民國首位具有醫師背景及以副總統身分競選成功的總統。與此同時，民進黨以連續三屆執政創下自1996年中華民國總統選舉以來政黨連屆執政的最長紀錄。
            Claim:賴清德是中華民國首位具有老師背景及以副總統身分競選成功的總統
            """
        }
    ]
    )
print(response.choices[0].message.content)