import anthropic
import base64

client = anthropic.Anthropic()

screenshot_path = "assets/claude-docs-vision.png"
screenshot_media_type = "image/png"
with open(screenshot_path, "rb") as f:
    screenshot_data = base64.b64encode(f.read()).decode("utf-8")

client = anthropic.Anthropic()
message = client.messages.create(
    model="claude-3-opus-20240229",
    max_tokens=1024,
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": screenshot_media_type,
                        "data": screenshot_data,
                    },
                },
                {
                    "type": "text",
                    "text": "Describe this image in Japanese."
                }
            ],
        }
    ],
)
print(message)
