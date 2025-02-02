import os
from dotenv import load_dotenv
from knowledge_distillation import TeacherModel

def test_teacher_model():
    load_dotenv()
    
    teacher = TeacherModel()
    prompt = "以下の形式で、人間が関与するイベントを1個生成してください。\n\n### ルール\n- 短い日本語の文にすること (7～15文字程度)\n- 主語は \"X\" にする\n- 日常的にありふれた出来事を考えること\n\n### 出力\nMarkdownの-を使った順序なしリスト形式で出力してください。"
    
    print("Sending request to teacher model...")
    response = teacher.generate(prompt)
    print(f"Teacher model response:\n{response}")

if __name__ == "__main__":
    test_teacher_model()
