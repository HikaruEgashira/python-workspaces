#%%
import os
from multion.client import MultiOn

# Configuration
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
MULTION_API_KEY = os.getenv('MULTION_API_KEY')

client = MultiOn(api_key=MULTION_API_KEY)
create_response = client.sessions.create(
    url="https://google.com",
    # local=True
)

command = """今日の日本のニュースをダイジェスト形式で教えて"""
browse_result = client.browse(
    cmd=command,
    session_id=create_response.session_id,
)
print(browse_result.status)
print(browse_result.url)
print(browse_result.message)
print(browse_result.screenshot)

while browse_result.status != "DONE":
    revised_command = input("Please provide a revised command: ")
    if revised_command == "exit":
        break
    if revised_command:
        command = revised_command

    browse_result = client.browse(
        cmd=command,
        session_id=browse_result.session_id,
    )
    print(browse_result.status)
    print(browse_result.url)
    print(browse_result.message)
    print(browse_result.screenshot)

# %%
