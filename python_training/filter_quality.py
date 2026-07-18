import os
import sys
import json
import random
import argparse
import sqlite3
from datetime import datetime
import numpy as np

# Lazy imports to speed up CLI and prevent failures if packages aren't installed yet
# Helper function to call the OpenAI-compatible gateway proxy using requests
def call_proxy_api(prompt, model_name):
    import requests
    api_key = os.environ.get("ANTHROPIC_API_KEY", "my-super-secret-password-123")
    base_url = os.environ.get("ANTHROPIC_BASE_URL", "https://ytdlp.kodyazar.com/v1")
    url = f"{base_url.rstrip('/')}/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 150
    }
    
    res = requests.post(url, headers=headers, json=payload, timeout=30)
    if res.status_code == 200:
        data = res.json()
        return data["choices"][0]["message"]["content"].strip()
    else:
        raise Exception(f"API Error: {res.status_code} - {res.text}")

def get_datasets():
    try:
        import datasets
        return datasets
    except ImportError:
        print("Error: 'datasets' package is not installed. Run 'pip install datasets'.")
        sys.exit(1)

def get_fasttext():
    try:
        import fasttext
        return fasttext
    except ImportError:
        print("Error: 'fasttext' package is not installed. Run 'pip install fasttext'.")
        sys.exit(1)

def get_tokenizer(tokenizer_path):
    try:
        from tokenizers import Tokenizer
        return Tokenizer.from_file(tokenizer_path)
    except ImportError:
        print("Error: 'tokenizers' package is not installed. Run 'pip install tokenizers'.")
        sys.exit(1)

def log_kaydet(db_path, index, toplam, url, puan, metin):
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            zaman TEXT,
            sira INTEGER,
            toplam INTEGER,
            kaynak_url TEXT,
            puan REAL,
            metin TEXT
        )
    """)
    conn.execute("""
        INSERT INTO log (zaman, sira, toplam, kaynak_url, puan, metin)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (datetime.now().isoformat(), index, toplam, url, puan, metin[:200]))
    conn.commit()
    conn.close()

def sample_and_label(args):
    datasets = get_datasets()
    
    print("Loading mc4 Turkish dataset in streaming mode...")
    dataset = datasets.load_dataset("mc4", "tr", split="train", streaming=True, trust_remote_code=True)
    
    print(f"Sampling {args.sample_size} records...")
    records = []
    # Take a larger stream and randomly sample to ensure diversity
    stream = iter(dataset)
    for _ in range(args.sample_size * 2):
        try:
            item = next(stream)
            records.append(item)
        except StopIteration:
            break
            
    if len(records) > args.sample_size:
        records = random.sample(records, args.sample_size)
        
    print(f"Starting Claude API evaluation for {len(records)} records...")
    labeled_data = []
    
    for i, rec in enumerate(records):
        metin = rec.get("text", "")
        url = rec.get("url", "bilinmiyor")
        if not metin:
            continue
            
        prompt = f"""Aşağıdaki veriyi 0-5 arası bir kalite puanıyla değerlendir:
0 = tamamen çöp/anlamsız/spam/reklam
5 = son derece kaliteli, bilgilendirici, doğru, iyi yapılandırılmış bilimsel veya akademik seviyede metin

Sadece şu JSON formatında cevap ver: {{"puan": X, "sebep": "kısa açıklama"}}

VERİ:
{metin[:3000]}
"""
        try:
            raw_text = call_proxy_api(prompt, args.model)
            # Attempt to extract JSON from markdown wrappers if present
            if "```json" in raw_text:
                raw_text = raw_text.split("```json")[1].split("```")[0].strip()
            elif "```" in raw_text:
                raw_text = raw_text.split("```")[1].split("```")[0].strip()
                
            res = json.loads(raw_text)
            puan = float(res.get("puan", 0))
            sebep = res.get("sebep", "")
        except Exception as e:
            print(f"Warning: Claude API call or parse failed: {e}. Defaulting score to 0.")
            puan = 0.0
            sebep = "Hata nedeniyle otomatik elendi"
            
        print(f"[{i+1}/{len(records)}] URL: {url[:50]} | Puan: {puan}")
        log_kaydet(args.db_path, i+1, len(records), url, puan, metin)
        
        labeled_data.append({
            "id": i,
            "metin": metin,
            "url": url,
            "puan": puan,
            "sebep": sebep
        })
        
    # Save results to jsonl
    os.makedirs(os.path.dirname(args.labeled_jsonl), exist_ok=True)
    with open(args.labeled_jsonl, "w", encoding="utf-8") as f:
        for item in labeled_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
    print(f"Claude labeling finished! Labeled file saved to {args.labeled_jsonl}")

def train_classifier(args):
    fasttext = get_fasttext()
    
    if not os.path.exists(args.labeled_jsonl):
        print(f"Error: Labeled data file not found at {args.labeled_jsonl}. Run with --mode sample-and-label first.")
        sys.exit(1)
        
    print("Preparing training file for fastText...")
    ft_train_path = args.labeled_jsonl + ".ft_train"
    
    # Read items
    items = []
    with open(args.labeled_jsonl, "r", encoding="utf-8") as fin:
        for line in fin:
            items.append(json.loads(line))
            
    # Classify into two categories with threshold >= 2
    kaliteli_items = []
    dusuk_items = []
    for item in items:
        if item["puan"] >= 2:
            kaliteli_items.append(item)
        else:
            dusuk_items.append(item)
            
    print(f"Original dataset: {len(kaliteli_items)} kaliteli, {len(dusuk_items)} dusuk")
    
    # Balance the dataset by oversampling kaliteli_items to match len(dusuk_items)
    if len(kaliteli_items) > 0 and len(dusuk_items) > 0:
        target_count = max(len(kaliteli_items), len(dusuk_items))
        # Oversample kaliteli
        while len(kaliteli_items) < target_count:
            kaliteli_items.extend(kaliteli_items[:target_count - len(kaliteli_items)])
        # Oversample dusuk
        while len(dusuk_items) < target_count:
            dusuk_items.extend(dusuk_items[:target_count - len(dusuk_items)])
            
    balanced_items = kaliteli_items + dusuk_items
    random.shuffle(balanced_items)
    
    count = 0
    with open(ft_train_path, "w", encoding="utf-8") as fout:
        for item in balanced_items:
            label = "__label__kaliteli" if item["puan"] >= 2 else "__label__dusuk"
            metin = item["metin"].replace("\n", " ")[:2000]
            fout.write(f"{label} {metin}\n")
            count += 1
            
    print(f"Training fastText model on {count} balanced samples...")
    # Train supervised fastText model
    model = fasttext.train_supervised(
        input=ft_train_path,
        epoch=50,
        lr=0.2,
        wordNgrams=2,
        dim=100
    )
    
    model.save_model(args.classifier_path)
    print(f"Classifier saved successfully to {args.classifier_path}")

def filter_dataset(args):
    fasttext = get_fasttext()
    datasets = get_datasets()
    tokenizer = get_tokenizer(args.tokenizer_path)
    
    if not os.path.exists(args.classifier_path):
        print(f"Error: Classifier model not found at {args.classifier_path}. Run with --mode train-classifier first.")
        sys.exit(1)
        
    model = fasttext.load_model(args.classifier_path)
    
    print("Loading mc4 Turkish dataset in streaming mode...")
    dataset = datasets.load_dataset("mc4", "tr", split="train", streaming=True, trust_remote_code=True)
    
    print(f"Opening binary output file: {args.output_path}")
    if os.path.dirname(args.output_path):
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        
    total_tokens = 0
    total_processed = 0
    total_accepted = 0
    
    token_buffer = []
    
    with open(args.output_path, "wb") as fout:
        for item in dataset:
            total_processed += 1
            metin = item.get("text", "")
            if not metin:
                continue
                
            # Clean metin format for classifier prediction
            clean_text = metin.replace("\n", " ")[:2000]
            label, confidence = model.predict(clean_text)
            
            is_high_quality = label[0] == "__label__kaliteli"
            
            if is_high_quality:
                total_accepted += 1
                enc = tokenizer.encode(metin)
                token_buffer.extend(enc.ids)
                
                # Buffered write to binary
                if len(token_buffer) >= args.chunk_size:
                    arr = np.array(token_buffer, dtype=np.uint32)
                    fout.write(arr.tobytes())
                    total_tokens += len(token_buffer)
                    token_buffer = []
                    
                    if total_processed % 1000 == 0 or total_tokens >= args.target_tokens:
                        print(f"[PROGRESS] Processed: {total_processed:,} | Accepted: {total_accepted:,} | Tokens Saved: {total_tokens:,}")
                        
            if total_tokens >= args.target_tokens:
                break
                
        # Flush remaining tokens
        if token_buffer and total_tokens < args.target_tokens:
            remaining_needed = args.target_tokens - total_tokens
            if remaining_needed > 0:
                write_buf = token_buffer[:remaining_needed]
                arr = np.array(write_buf, dtype=np.uint32)
                fout.write(arr.tobytes())
                total_tokens += len(write_buf)
                
    print(f"Dataset filtering completed! Total tokens saved: {total_tokens:,} from {total_accepted:,} high quality documents.")

def main():
    parser = argparse.ArgumentParser(description="FineWeb-Edu Quality Filter System for Maya-1")
    parser.add_argument("--mode", type=str, required=True, choices=["sample-and-label", "train-classifier", "filter-dataset"], help="System execution mode")
    parser.add_argument("--sample_size", type=int, default=100, help="Number of records for Claude labeling sample (use larger for prod, eg 50000)")
    parser.add_argument("--labeled_jsonl", type=str, default="../shared/claude_labels.jsonl", help="Path to save Claude graded records")
    parser.add_argument("--classifier_path", type=str, default="../shared/quality_classifier.bin", help="Path to save trained fastText classifier")
    parser.add_argument("--db_path", type=str, default="../shared/islem_log.db", help="Path to SQLite logging database")
    parser.add_argument("--tokenizer_path", type=str, default="../shared/mock_tokenizer.json", help="Path to model tokenizer JSON file")
    parser.add_argument("--output_path", type=str, default="../shared/clean_data.bin", help="Path to output clean binary token file")
    parser.add_argument("--target_tokens", type=int, default=3000000000, help="Target token count for pretraining (default: 3 billion)")
    parser.add_argument("--chunk_size", type=int, default=1000000, help="Token buffer size before writing to binary file")
    parser.add_argument("--model", type=str, default="claude-sonnet-4.5", help="Model name (e.g. claude-sonnet-4.5 or minimax-m2.5)")
    
    args = parser.parse_args()
    
    # Adjust paths if we are running from project directory vs workspace root
    # Ensure they exist
    if args.mode == "sample-and-label":
        sample_and_label(args)
    elif args.mode == "train-classifier":
        train_classifier(args)
    elif args.mode == "filter-dataset":
        filter_dataset(args)

if __name__ == "__main__":
    main()
