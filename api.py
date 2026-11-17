import sys
import os
import csv
import datetime

sys.path.append("src")

from fastapi import FastAPI
from pydantic import BaseModel
from pipeline import LostFoundMatcher

app = FastAPI(title="Smart Lost and Found")

# Change these settings to match your best MLflow run
matcher = LostFoundMatcher()

LOG_FILE = "logs/requests.csv"


class LostItem(BaseModel):
    description: str


def log_request(query, top_id, top_score):
    os.makedirs("logs", exist_ok=True)
    new_file = not os.path.exists(LOG_FILE)
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(["time", "query", "top_id", "top_score"])
        writer.writerow([datetime.datetime.now().isoformat(), query, top_id, top_score])


@app.get("/")
def home():
    return {"message": "Lost and Found API is running"}


@app.post("/match")
def match_item(item: LostItem):
    results = matcher.match(item.description, top_n=3)
    matches = [
        {"id": int(r.id), "description": r.description,
         "location": r.location, "score": round(float(r.score), 3)}
        for r in results.itertuples()
    ]
    log_request(item.description, matches[0]["id"], matches[0]["score"])
    return {"matches": matches}