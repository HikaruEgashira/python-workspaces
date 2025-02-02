import os
from dotenv import load_dotenv
from knowledge_distillation import KnowledgeDistillation

def test_inference_generation():
    load_dotenv()
    
    distillation = KnowledgeDistillation()
    
    # テストケース: 日常的なイベントと非日常的なイベント
    test_events = [
        "X が公園で散歩する",  # 日常的
        "X が宇宙船で火星に着陸する",  # 非日常的（SF）
        "X が魔法で敵を倒す",  # 非日常的（ファンタジー）
        "X が100メートル走で世界記録を更新する",  # 非日常的（特別な達成）
        "X が時間を巻き戻して過去を変える"  # 非日常的（超現実的）
    ]
    
    for event in test_events:
        print(f"\n=== Testing event: {event} ===")
        
        # 推論の生成
        print("\nGenerating inferences...")
        inferences = distillation.generate_inference(event)
        print("\nGenerated inferences:")
        for relation, inference in inferences.items():
            print(f"{relation}: {inference}")
        
        # 推論の評価
        print("\nEvaluating inferences...")
        for relation in ["xEffect", "xIntent"]:
            if relation in inferences:
                is_valid = distillation.filter_inference(event, relation, inferences[relation])
                print(f"Evaluation result for {relation}: {is_valid}")

if __name__ == "__main__":
    test_inference_generation()
