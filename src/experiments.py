import mlflow
from pipeline import LostFoundMatcher
from evaluate import evaluate

experiments = [
    {"name": "baseline",      "ngram_range": (1, 1), "use_stopwords": True,  "include_location": False},
    {"name": "no_stopwords",  "ngram_range": (1, 1), "use_stopwords": False, "include_location": False},
    {"name": "bigrams",       "ngram_range": (1, 2), "use_stopwords": True,  "include_location": False},
    {"name": "with_location", "ngram_range": (1, 1), "use_stopwords": True,  "include_location": True},
]

mlflow.set_experiment("lost-found-matching")

for exp in experiments:
    with mlflow.start_run(run_name=exp["name"]):
        matcher = LostFoundMatcher(
            ngram_range=exp["ngram_range"],
            use_stopwords=exp["use_stopwords"],
            include_location=exp["include_location"],
        )
        top1, top3 = evaluate(matcher)

        mlflow.log_param("ngram_range", str(exp["ngram_range"]))
        mlflow.log_param("use_stopwords", exp["use_stopwords"])
        mlflow.log_param("include_location", exp["include_location"])
        mlflow.log_metric("top1_accuracy", top1)
        mlflow.log_metric("top3_accuracy", top3)

        print(exp["name"], "-> top1:", round(top1, 3), "top3:", round(top3, 3))