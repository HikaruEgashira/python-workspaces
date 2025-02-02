import os
from dotenv import load_dotenv
from knowledge_distillation import KnowledgeDistillation

def main():
    load_dotenv()
    
    distillation = KnowledgeDistillation()
    
    # イベントの生成
    events = distillation.generate_events(num_events=5)
    print("生成されたイベント:")
    for event in events:
        print(f"- {event}")
        
    # 評価メトリクスの計算
    print("\n評価メトリクス:")
    metrics = distillation.evaluate_inferences(events)
    
    for metric_name, metric_values in metrics.items():
        print(f"\n{metric_name}:")
        for relation, value in metric_values.items():
            print(f"- {relation}: {value:.2f}")

if __name__ == "__main__":
    main()
