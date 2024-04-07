import anthropic

client = anthropic.Anthropic()
message = client.messages.create(
    model="claude-3-haiku-20240307",
    max_tokens=1000,
    temperature=0,
    messages=[
        {
            "role": "user",
            "content": """
                あなたは役立つアシスタントです。
                create a haiku about the ocean
         """,
        },
    ],
)
if len(message.content) > 0:
    print(f"Success: content is {message.content}")
else:
    print("Failed: content is empty")
    exit(1)
