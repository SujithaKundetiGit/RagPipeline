import json
import gzip
from datasets import load_dataset

dataset = load_dataset("natural_questions", "default", split="train[:100]")

output_file = "nq_sample_with_answers.jsonl.gz"

def get_text_from_tokens(tokens, start, end):
    """Reconstruct text from token list"""
    return " ".join([t["token"] for t in tokens[start:end]])

with gzip.open(output_file, "wt", encoding="utf-8") as f:
    for i, item in enumerate(dataset):

        question = item.get("question", {}).get("text", "")
        tokens = item.get("document_tokens", [])

        long_answer_text = ""
        short_answers_text = []

        annotations = item.get("annotations", [])

        if annotations:
            ann = annotations[0]

            # ✅ LONG ANSWER
            long_ans = ann.get("long_answer", {})
            start = long_ans.get("start_token", -1)
            end = long_ans.get("end_token", -1)

            if start != -1 and end != -1:
                long_answer_text = get_text_from_tokens(tokens, start, end)

            # ✅ SHORT ANSWERS
            short_answers = ann.get("short_answers", [])
            for sa in short_answers:
                s = sa.get("start_token", -1)
                e = sa.get("end_token", -1)

                if s != -1 and e != -1:
                    short_answers_text.append(
                        get_text_from_tokens(tokens, s, e)
                    )

        example = {
            "example_id": i,
            "question_text": question,
            "long_answer": long_answer_text,
            "short_answers": short_answers_text
        }

        f.write(json.dumps(example) + "\n")

print("Saved -> nq_sample_with_answers.jsonl.gz")