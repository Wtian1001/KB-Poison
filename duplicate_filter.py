import json
import hashlib

input_path = "datasets/nq_corpus.json"
output_path = "nq_corpus_deduped.json"


def sha256_hash(text: str) -> str:
    """Compute SHA-256 hex digest of the input text."""
    h = hashlib.sha256()
    h.update(text.encode('utf-8'))
    return h.hexdigest()

def dedupe_kb(input_path: str, output_path: str):
    seen_hashes = set()
    total, kept = 0, 0

    with open(input_path, 'r', encoding='utf-8') as f_in, \
         open(output_path, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            total += 1
            item = json.loads(line)
            text = item.get('corpus', '')
            h = sha256_hash(text)
            if h not in seen_hashes:
                seen_hashes.add(h)
                f_out.write(json.dumps(item, ensure_ascii=False) + "\n")
                kept += 1

    print(f"Processed {total} documents, kept {kept}, filtered out {total - kept} duplicates.")

if __name__ == "__main__":
    dedupe_kb(input_path, output_path)