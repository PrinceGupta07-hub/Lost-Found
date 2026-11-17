import pandas as pd
from pipeline import LostFoundMatcher


def evaluate(matcher, queries_path="data/test_queries.csv"):
    tests = pd.read_csv(queries_path)
    top1_hits = 0
    top3_hits = 0

    for _, row in tests.iterrows():
        results = matcher.match(row["query"], top_n=3)
        top_ids = results["id"].tolist()

        if top_ids[0] == row["correct_id"]:
            top1_hits += 1
        if row["correct_id"] in top_ids:
            top3_hits += 1
        else:
            print("MISSED:", row["query"], "| expected id", row["correct_id"], "| got", top_ids)

    total = len(tests)
    return top1_hits / total, top3_hits / total


if __name__ == "__main__":
    matcher = LostFoundMatcher()
    top1, top3 = evaluate(matcher)
    print(f"Top-1 accuracy: {top1:.2%}")
    print(f"Top-3 accuracy: {top3:.2%}")