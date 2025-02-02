from openai import OpenAI
import os

class TeacherModel:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
    def generate(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model="o3-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content

class StudentModel:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
    def generate(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
