from openai import OpenAI
import os
import httpx

class TeacherModel:
    def __init__(self):
        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            http_client=httpx.Client(timeout=30.0)
        )
        
    def generate(self, prompt: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model="o3-mini",
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            print(f"Teacher model error: {e}")
            return ""

class StudentModel:
    def __init__(self):
        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            http_client=httpx.Client(timeout=30.0)
        )
        
    def generate(self, prompt: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            print(f"Student model error: {e}")
            return ""
