import os
from openai import OpenAI

file_path = "./SelectConfiguration.txt"  # Replace with your file path

# Open the file and read its contents
with open(file_path, "r", encoding='utf-8', errors='ignore') as file:
    instruction = file.read()

client = OpenAI(
    api_key="sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
    base_url="https://xxxxxxx.opapi.win/v1",
)

chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": instruction,
        }
    ],
    model="gpt-4",
)

print(chat_completion.choices[0].message.content)
