from typing import List, Dict, Tuple
from collections import Counter
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
        
    def generate_inference(self, event: str) -> dict[str, str]:
        print(f"\nGenerating inference for event: {event}")
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

### 出力フォーマット
- xEvent: {event}
- xEffect: xxx
- xWant: xxx
- xNeed: xxx
- xIntent: xxx
- xReact: xxx
- HinderedBy: xxx

出力フォーマットには厳密に従ってください。出力フォーマットにない余計な出力は絶対に含めないようにしてください。
"""
        print("Sending request to teacher model...")
        response = self.teacher.generate(prompt)
        print(f"Teacher model response:\n{response}")
        relations = {}
        for line in response.split("\n"):
            if ":" in line:
                key, value = line.split(":", 1)
                key = key.strip("- ")
                relations[key] = value.strip()
        return relations

    def evaluate_inferences(self, events: List[str], num_samples: int = 5) -> Dict[str, Dict[str, float]]:
        metrics = {
            "generation_rate": {},  # 各関係タイプの生成成功率
            "validity_rate": {},    # 各関係タイプの妥当性評価率
            "diversity": {},        # 各関係タイプの推論の多様性
        }
        
        all_inferences = {relation: [] for relation in ["xEffect", "xWant", "xNeed", "xIntent", "xReact", "HinderedBy"]}
        valid_counts = {relation: 0 for relation in all_inferences.keys()}
        total_counts = {relation: 0 for relation in all_inferences.keys()}
        
        for event in events[:num_samples]:
            inferences = self.generate_inference(event)
            for relation in all_inferences.keys():
                if relation in inferences:
                    total_counts[relation] += 1
                    inference = inferences[relation]
                    all_inferences[relation].append(inference)
                    
                    if self.filter_inference(event, relation, inference):
                        valid_counts[relation] += 1
        
        for relation in all_inferences.keys():
            # 生成率の計算
            metrics["generation_rate"][relation] = total_counts[relation] / num_samples
            
            # 妥当性の計算
            if total_counts[relation] > 0:
                metrics["validity_rate"][relation] = valid_counts[relation] / total_counts[relation]
            else:
                metrics["validity_rate"][relation] = 0.0
            
            # 多様性の計算（重複度の逆数）
            if all_inferences[relation]:
                counter = Counter(all_inferences[relation])
                metrics["diversity"][relation] = len(all_inferences[relation]) / len(counter)
            else:
                metrics["diversity"][relation] = 0.0
        
        return metrics

    def filter_inference(self, event: str, relation: str, inference: str) -> bool:
        print(f"\nEvaluating inference:")
        print(f"Event: {event}")
        print(f"Relation: {relation}")
        print(f"Inference: {inference}")
        
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
        print("Sending request to student model...")
        response = self.student.generate(prompt)
        print(f"Student model response: {response}")
        return response.strip().lower() == "true"
