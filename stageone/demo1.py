from openai import OpenAI

client = OpenAI(
    api_key="sk-5368191a39734dbba163f7516d8bcd7c",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

# 测试基础对话
completion = client.chat.completions.create(
    model="qwen-plus",  # 或 qwen-max-latest
    messages=[
        {"role": "system", "content": "你是一个有用的助手"},
        {"role": "user", "content": "你好，请介绍你自己"}
    ]
)
print(completion.choices[0].message.content)