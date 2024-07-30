#%%
import os
from multion.client import MultiOn

# Configuration
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
MULTION_API_KEY = os.getenv('MULTION_API_KEY')

multion = MultiOn(api_key=MULTION_API_KEY)

command = "LLMに関する今日最新の論文を見つけてください。"
browse_result = multion.browse(cmd=command, url="https://arxiv.org/")
print(browse_result.status)
print(browse_result.message)
print(browse_result.screenshot)

while browse_result.status == "NOT_SURE":
    revised_command = input("Please provide a revised command: ")
    if revised_command == "exit":
        break
    if revised_command:
        command += revised_command

    browse_result = multion.browse(
        cmd=command,
        session_id=browse_result.session_id,
        url=browse_result.url
    )
    print(browse_result.status)
    print(browse_result.message)
    print(browse_result.screenshot)

# %%
