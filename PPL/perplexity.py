import json
import math
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

DATASET = "nq" # "nq" or "ms" or "hot"
clean = True 

MODEL_NAME = "gpt2"  

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
model.eval()
if torch.cuda.is_available():
    model = model.cuda()

def calculate_perplexity(sentence):
    # tokenize
    encodings = tokenizer(sentence, return_tensors="pt")
    input_ids = encodings.input_ids
    if torch.cuda.is_available():
        input_ids = input_ids.cuda()

    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss
    # PPL = exp(loss)
    return math.exp(loss.item())

def load_jsonl(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            yield json.loads(line)

def load_clean_jsonl(filepath):
    with open(filepath, 'r', encoding='utf-8') as g:
        data = json.load(g)
    return data


def main(database_path, output_path, clean):
    results = []
    if clean:
        entries = load_clean_jsonl(database_path)
        for entry in entries:
            _id = entry.get("id")
            corpus = entry.get("text", "")
            ppl = calculate_perplexity(corpus)
            results.append({
                "id": _id,
                "perplexity": ppl
            })
    else:
        for item in tqdm(load_jsonl(database_path), desc="Processing"):
            _id = item['id']
            corpus = item['corpus']
            ppl = calculate_perplexity(corpus)
            results.append({
                "id": _id,
                "perplexity": ppl
            })

    with open(output_path, 'w', encoding='utf-8') as fout:
        for res in results:
            fout.write(json.dumps(res, ensure_ascii=False) + '\n')
    print(f"Saved results to {output_path}")

if __name__ == "__main__":
    if clean:
        database_path = f"../datasets/{DATASET}_corpus_origin.json"
        output_path = f"{DATASET}_clean_ppl.json"
    else:
        database_path = f"../datasets/{DATASET}_corpus.json"
        output_path = f"{DATASET}_ppl.json"
    main(database_path, output_path, clean)