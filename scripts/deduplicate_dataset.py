import os
import sys
import json
import argparse
import numpy as np
from tqdm import tqdm
from datasketch import MinHash, MinHashLSH

# Lazy import model2vec to prevent failure if package is not ready
def get_model2vec():
    try:
        from model2vec import StaticModel
        return StaticModel
    except ImportError:
        print("Warning: model2vec not found. We will use a lightweight fallback or wait for installation.")
        return None

def preprocess_text(text):
    # Minimal normalization for shingling
    return text.lower().strip()

def get_shingles(text, k=5):
    # Construct character k-grams
    text = preprocess_text(text)
    shingles = set()
    for i in range(len(text) - k + 1):
        shingles.add(text[i:i+k])
    return shingles

def run_deduplication(args):
    import datasets
    
    print("Loading mc4 Turkish dataset split in streaming mode...")
    dataset = datasets.load_dataset("mc4", "tr", split="train", streaming=True, trust_remote_code=True)
    
    # 1. MinHash LSH Initialization
    lsh = MinHashLSH(threshold=args.minhash_threshold, num_perm=128)
    
    # Read a buffer of documents to deduplicate
    print(f"Reading first {args.max_docs} documents from stream...")
    docs = []
    stream = iter(dataset)
    for idx in range(args.max_docs):
        try:
            item = next(stream)
            docs.append({
                "id": idx,
                "text": item.get("text", ""),
                "url": item.get("url", "")
            })
        except StopIteration:
            break
            
    print(f"Loaded {len(docs)} documents for deduplication processing.")
    
    # Fuzzy Deduplication (MinHash LSH)
    unique_docs = []
    duplicate_count = 0
    
    print("Running MinHash LSH fuzzy deduplication...")
    for doc in tqdm(docs):
        text = doc["text"]
        if len(text) < 50: # Skip extremely short documents
            continue
            
        shingles = get_shingles(text, k=5)
        if not shingles:
            continue
            
        m = MinHash(num_perm=128)
        for s in shingles:
            m.update(s.encode('utf-8'))
            
        # Query LSH index
        result = lsh.query(m)
        if len(result) > 0:
            duplicate_count += 1
            # Found a duplicate candidate, skip it
            continue
            
        # If no duplicates found, insert into LSH and keep
        lsh.insert(doc["id"], m)
        doc["minhash"] = m
        unique_docs.append(doc)
        
    print(f"MinHash LSH complete. Removed {duplicate_count} fuzzy duplicates. Remaining: {len(unique_docs)}")
    
    # Semantic Deduplication (SemDeDup)
    StaticModel = get_model2vec()
    if StaticModel is None or not args.enable_semdedup:
        print("Semantic Deduplication skipped or disabled.")
        final_docs = unique_docs
    else:
        print("Loading Model2Vec static model for semantic embeddings...")
        # Load a very lightweight model
        try:
            model = StaticModel.from_pretrained("minishlab/potion-base-8M")
        except Exception as e:
            print(f"Failed to load potion-base-8M: {e}. Trying glove fallback...")
            model = StaticModel.from_pretrained("minishlab/M2V_base_glove")
            
        print("Generating document embeddings...")
        texts = [doc["text"][:1000] for doc in unique_docs] # Truncate to first 1000 chars for speed
        embeddings = model.encode(texts)
        
        # Normalize embeddings for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normalized_embeddings = embeddings / norms
        
        print("Filtering semantically redundant documents...")
        sem_duplicate_count = 0
        keep_mask = np.ones(len(unique_docs), dtype=bool)
        
        # Iteratively check similarity
        for i in range(len(unique_docs)):
            if not keep_mask[i]:
                continue
            # Calculate cosine similarity with all subsequent documents
            sims = np.dot(normalized_embeddings[i+1:], normalized_embeddings[i])
            # Find indices where similarity is above threshold
            dup_indices = np.where(sims > args.sem_threshold)[0] + i + 1
            for dup_idx in dup_indices:
                if keep_mask[dup_idx]:
                    keep_mask[dup_idx] = False
                    sem_duplicate_count += 1
                    
        final_docs = [unique_docs[i] for i in range(len(unique_docs)) if keep_mask[i]]
        print(f"SemDeDup complete. Removed {sem_duplicate_count} semantic duplicates. Remaining: {len(final_docs)}")
        
    # Save the deduplicated documents
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        for doc in final_docs:
            record = {"text": doc["text"], "url": doc["url"]}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            
    print(f"Deduplicated corpus saved to {args.output_path} (Total: {len(final_docs)} documents)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Double-Layer Deduplication (MinHash LSH + SemDeDup)")
    parser.add_argument("--max_docs", type=int, default=10000, help="Number of documents to process")
    parser.add_argument("--minhash_threshold", type=float, default=0.8, help="MinHash Jaccard similarity threshold")
    parser.add_argument("--sem_threshold", type=float, default=0.85, help="Semantic cosine similarity threshold")
    parser.add_argument("--enable_semdedup", type=bool, default=True, help="Enable Model2Vec semantic dedup")
    parser.add_argument("--output_path", type=str, default="shared/deduplicated_mc4_tr.jsonl", help="Output path")
    args = parser.parse_args()
    
    run_deduplication(args)
