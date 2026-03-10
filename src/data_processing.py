import json


def extract_n_abstracts(file_path, n=1000):
    abstracts = []
    current_sentences = []
    current_id = None

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            # Detect start of a new abstract
            if line.startswith("###"):
                if current_sentences and current_id:
                    abstracts.append({
                        "abstract_id": current_id,
                        "text": " ".join(current_sentences)
                    })

                    if len(abstracts) >= n:
                        break

                    current_sentences = []

                current_id = line.replace("###", "")

            # Process sentence lines
            elif line:
                parts = line.split("\t")
                if len(parts) == 2:
                    _, sentence = parts
                    current_sentences.append(sentence)

    return abstracts


def save_abstracts(abstracts, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(abstracts, f, indent=2)

def chunk_abstracts(abstracts, chunk_size=200, overlap=40):

    chunks = []
    chunk_id = 0

    for abstract in abstracts:

        words = abstract["text"].split()

        start = 0
        while start < len(words):

            end = start + chunk_size
            chunk_words = words[start:end]

            chunk_text = " ".join(chunk_words)

            chunks.append({
                "chunk_id": chunk_id,
                "abstract_id": abstract["abstract_id"],
                "text": chunk_text
            })

            chunk_id += 1
            start += chunk_size - overlap

    return chunks