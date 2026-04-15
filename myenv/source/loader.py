import gzip
import json
import os
import csv

def load_nq_data(path):
    texts = []

    # Handle CSV files
    if path.endswith('.csv'):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Extract text from question and long_answers
                    question = row.get('question', '').strip()
                    long_answer = row.get('long_answers', '').strip()
                    
                    # Combine question and answer for richer context
                    text = f"{question} {long_answer}" if long_answer else question
                    
                    if text.strip():
                        texts.append(text)
            return texts
        except Exception as e:
            raise FileNotFoundError(f"Error reading CSV file {path}: {str(e)}")

    # Handle JSONL/gzip files
    gz_path = path if path.endswith('.gz') else path + '.gz'
    jsonl_path = path if not path.endswith('.gz') else path[:-3]

    if os.path.exists(gz_path):
        file_path = gz_path
        open_fn = lambda p: gzip.open(p, "rt", encoding="utf-8")
    elif os.path.exists(jsonl_path):
        file_path = jsonl_path
        open_fn = lambda p: open(p, "rt", encoding="utf-8")
    else:
        raise FileNotFoundError(f"Missing dataset file. Please add either '{gz_path}' or '{jsonl_path}' to the project root.")

    with open_fn(file_path) as f:
        for line in f:
            if not line.strip():
                continue
            
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            # NQ dataset has question_text as primary field
            text = (
                obj.get("question_text")
                or obj.get("document_text")
                or obj.get("context")
                or obj.get("document_html")
                or ""
            )

            if text.strip():
                texts.append(text)

    return texts
