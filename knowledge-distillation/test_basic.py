import os
from dotenv import load_dotenv
from knowledge_distillation import KnowledgeDistillation

def test_basic_functionality():
    load_dotenv()
    
    distillation = KnowledgeDistillation()
    events = distillation.generate_events(num_events=1)
    print("Generated event:", events)

if __name__ == "__main__":
    test_basic_functionality()
