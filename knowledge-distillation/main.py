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
        
    # 最初のイベントに対する推論の生成と評価
    if events:
        print("\n最初のイベントに対する推論:")
        inferences = distillation.generate_inference(events[0])
        for relation, inference in inferences.items():
            print(f"- {relation}: {inference}")
            
            if relation != "xEvent":
                print(f"\n{relation}の評価:")
                is_valid = distillation.filter_inference(
                    event=events[0],
                    relation=relation,
                    inference=inference
                )
                print(f"評価結果: {'適切' if is_valid else '不適切'}")

if __name__ == "__main__":
    main()
