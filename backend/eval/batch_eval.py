import json
from eval.compare_retrievers import evaluate_all_retrievers

def evaluate_dataset(dataset_file, k=5):
    with open(dataset_file, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    results = []
    for sample in dataset:
        question = sample["question"]
        gold = sample["gold"]
        metrics = evaluate_all_retrievers(question, gold, k)
        metrics["question"] = question
        results.append(metrics)

    return results
