import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_processing import extract_n_abstracts, save_abstracts, chunk_abstracts


train_file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'train.txt'))


abstracts = extract_n_abstracts(train_file_path, n=1000)

print("Total abstracts extracted:", len(abstracts))

chunks = chunk_abstracts(abstracts)

print("Total chunks created:", len(chunks))

print("\nExample chunk:\n")
print(chunks[0]["text"][:500])

chunks_file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'chunks.json'))
with open(chunks_file_path, "w", encoding="utf-8") as f:
    json.dump(chunks, f, indent=2)

print(f"\nChunks saved to {chunks_file_path}")