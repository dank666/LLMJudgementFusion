"""Prepare the set-valued tasks: JEC-QA, TweetEmotion and TweetTopic.

JEC-QA: the 2,000 questions of the Chinese National Judicial Examination in AGIEval v1, split 500 / 1,500 with a
fixed seed (1,000 test questions are used). TweetEmotion (SemEval-2018 E-c, 11 labels) and TweetTopic (19 labels)
come from SuperTweetEval: the validation split for development and 1,000 tweets of the test split for testing.
Items without a gold label are left out. Every item is a multiple-select question whose options are the labels.
The lists of the items put to the frontier models are written to data/processed/frontier_uids/.
"""

import json
import random
import re
import urllib.request

import pandas as pd

from common import ROOT
from llmfusion.data import Question

RAW = ROOT / "data" / "raw" / "setvalued"
OUT = ROOT / "data" / "processed"
AGIEVAL = "https://raw.githubusercontent.com/ruixiangcui/AGIEval/main/data/v1/{}.jsonl"
TWEETEVAL = "https://huggingface.co/api/datasets/cardiffnlp/super_tweeteval/parquet/{}/{}/0.parquet"
LETTERS = "ABCDEFGHIJKLMNOPQRS"
EMOTIONS = ["anger", "anticipation", "disgust", "fear", "joy", "love", "optimism", "pessimism", "sadness",
            "surprise", "trust"]
TOPICS = ["arts_&_culture", "business_&_entrepreneurs", "celebrity_&_pop_culture", "diaries_&_daily_life", "family",
          "fashion_&_style", "film_tv_&_video", "fitness_&_health", "food_&_dining", "gaming",
          "learning_&_educational", "music", "news_&_social_concern", "other_hobbies", "relationships",
          "science_&_technology", "sports", "travel_&_adventure", "youth_&_student_life"]
TWEET_META = {"system": "You are an expert annotator of social media posts.",
              "header": "Please answer the following question about a tweet."}
LAW_META = {"system": "You are a legal expert.",
            "header": "Please answer the following question from the Chinese National Judicial Examination."}
TEST_SIZE = 1000
FRONTIER_DEV, FRONTIER_TEST = 300, 600


def fetch(url: str, name: str):
    path = RAW / name
    if not path.exists():
        RAW.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url, timeout=120) as r:
            path.write_bytes(r.read())
    return path


def jecqa() -> list[Question]:
    out = []
    for part in ("jec-qa-kd", "jec-qa-ca"):
        rows = [json.loads(x) for x in fetch(AGIEVAL.format(part), f"{part}.jsonl").read_text(encoding="utf-8").splitlines()
                if x.strip()]
        for i, r in enumerate(rows):
            if not r["label"]:
                continue
            opts = {}
            for o in r["options"]:
                m = re.match(r"\(([A-D])\)\s*(.*)", o.strip(), re.S)
                opts[m.group(1)] = m.group(2).strip()
            out.append(Question(uid=f"{part}-{i:04d}", dataset="jecqa", qtype="multiple", stem=r["question"].strip(),
                                options=opts, answer=sorted(r["label"]), passage=r.get("passage") or None,
                                source=f"AGIEval v1 {part}", meta={"language": "Chinese", "task": "law", **LAW_META}))
    return out


def tweets(task: str, split: str, labels: list[str]) -> list[Question]:
    df = pd.read_parquet(fetch(TWEETEVAL.format(task, split), f"{task}-{split}.parquet"))
    question = ("Which of the following emotions does the author of the tweet express?" if task == "tweet_emotion"
                else "Which of the following topics is the tweet about?")
    out = []
    for i, r in df.iterrows():
        gold = [LETTERS[k] for k, x in enumerate(r["gold_label_list"]) if int(x)]
        if not gold:
            continue
        opts = {LETTERS[k]: lab.replace("_&_", " & ").replace("_", " ") for k, lab in enumerate(labels)}
        out.append(Question(uid=f"{task.replace('_', '-')}-{split}-{i:04d}", dataset=task, qtype="multiple",
                            stem=f"Tweet: {r['text'].strip()}\n\n{question}", options=opts, answer=gold,
                            source=f"SuperTweetEval {task} {split}", meta={"language": "English", "task": task, **TWEET_META}))
    return out


def write(name: str, qs: list[Question]) -> None:
    (OUT / f"{name}.jsonl").write_text("".join(q.to_json() + "\n" for q in qs), encoding="utf-8")


def main() -> None:
    law = jecqa()
    random.Random(0).shuffle(law)
    sets = {"jecqa_dev": law[:500], "jecqa_test": law[500:],
            "tweet_emotion_dev": tweets("tweet_emotion", "validation", EMOTIONS),
            "tweet_emotion_test": tweets("tweet_emotion", "test", EMOTIONS),
            "tweet_topic_dev": tweets("tweet_topic", "validation", TOPICS),
            "tweet_topic_test": tweets("tweet_topic", "test", TOPICS)}
    uid_dir = OUT / "frontier_uids"
    uid_dir.mkdir(exist_ok=True)
    for task in ("jecqa", "tweet_emotion", "tweet_topic"):
        test = sets[f"{task}_test"]
        random.Random(1).shuffle(test)
        sets[f"{task}_test"] = test[:TEST_SIZE]
        (uid_dir / f"{task}_dev.txt").write_text("\n".join(q.uid for q in sets[f"{task}_dev"][:FRONTIER_DEV]) + "\n")
        (uid_dir / f"{task}_test.txt").write_text("\n".join(q.uid for q in sets[f"{task}_test"][:FRONTIER_TEST]) + "\n")
    for name, qs in sets.items():
        write(name, qs)
        sizes = {}
        for q in qs:
            sizes[len(q.answer)] = sizes.get(len(q.answer), 0) + 1
        print(f"{name:20s} {len(qs):5d} items, answer sizes {dict(sorted(sizes.items()))}")


if __name__ == "__main__":
    main()
