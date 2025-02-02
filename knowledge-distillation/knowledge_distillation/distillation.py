from typing import List
from .models import TeacherModel, StudentModel

class KnowledgeDistillation:
    def __init__(self):
        self.teacher = TeacherModel()
        self.student = StudentModel()
        
    def generate_events(self, num_events: int = 10) -> List[str]:
        prompt = f"""以下の形式で、人間が関与するイベントを{num_events}個生成してください。

### ルール
- 各イベントは短い日本語の文にすること (7～15文字程度)
- 主語は "X" にする (例: X が本を読む)
- 日常的にありふれた出来事を考えること
- 可能な限りバリエーションを持たせること

### 出力
Markdownの-を使った順序なしリスト形式で出力してください。
"""
        return self.teacher.generate(prompt).split("\n")
        
    def generate_inference(self, event: str) -> str:
        prompt = f"""以下のイベントについて、因果関係を持つ推論を生成してください。

### ルール
- 各関係に対応する推論を作成してください
- 出力はシンプルで自然な日本語の文章にしてください (10～20文字程度)
- 可能な限り現実的な推論を生成してください

### 関係の説明
- xEffect: X の行動の結果
- xWant: X の行動後に X が望むこと
- xNeed: X の行動をするために必要なこと
- xIntent: X の行動の意図
- xReact: X の行動の後の感情反応
- HinderedBy: X の行動が妨げられる要因

イベント: {event}
"""
        return self.teacher.generate(prompt)

    def filter_inference(self, event: str, relation: str, inference: str) -> bool:
        prompt = f"""以下のイベントと推論が適切かどうかを評価してください。

### チェック基準:
1. 論理的一貫性 - イベントと関係に対して適切な推論か
2. 常識的な妥当性 - 現実世界の知識と矛盾しないか
3. 情報の具体性 - 推論が単純すぎず、具体的な知識を含んでいるか
4. 不要な曖昧さがないか
5. 誤解を招く表現がないか

イベント: {event}
関係: {relation}
推論: {inference}

回答は "True" または "False" のみで答えてください。
"""
        return self.student.generate(prompt).strip().lower() == "true"
