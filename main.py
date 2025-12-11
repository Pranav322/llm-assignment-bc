import json
from src.pipeline import EvaluationPipeline

def load(path):
    with open(path) as f:
        return json.load(f)

if __name__ == "__main__":
    chat = load("data/sample-chat-conversation-01.json")
    vectors = load("data/sample_context_vectors-01.json")

    pipeline = EvaluationPipeline()
    report = pipeline.run(chat, vectors)

    print("\n===== FINAL REPORT =====")
    print(json.dumps(report, indent=4))

    with open("evaluation_report.json", "w") as f:
        json.dump(report, f, indent=4)
