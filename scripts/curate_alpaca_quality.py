import os
import sys
import json
import random
import argparse
import pandas as pd
import numpy as np
import requests
import joblib

try:
    from model2vec import StaticModel
    from sklearn.linear_model import LogisticRegression
except ImportError:
    print("Warning: model2vec or scikit-learn not installed. Will use dummy fallback if training.")

def call_proxy_api(prompt, model_name="claude-sonnet-4.5"):
    api_key = "my-super-secret-password-123"
    base_url = "https://ytdlp.kodyazar.com/v1"
    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 150,
        "temperature": 0.2
    }
    res = requests.post(url, headers=headers, json=payload, timeout=30)
    if res.status_code == 200:
        return res.json()["choices"][0]["message"]["content"].strip()
    else:
        raise Exception(f"API Error: {res.status_code} - {res.text}")

def generate_negative_response(chosen):
    r = random.random()
    if r < 0.25:
        words = chosen.split()
        if len(words) > 3:
            return " ".join(words[:len(words)//2]) + "..."
        return "Gerek yok."
    elif r < 0.50:
        return random.choice([
            "Buna cevap vermek istemiyorum.",
            "Bilmiyorum, yardımcı olamam.",
            "Bu konuda bilgim yok.",
            "Üzgünüm, şu an bunu yapamam."
        ])
    elif r < 0.75:
        words = chosen.split()
        if len(words) > 2:
            rep_word = words[-1]
            return " ".join(words[:min(5, len(words))]) + f" {rep_word}" * 8 + "..."
        return "Tekrar edin..."
    else:
        return "Anlamadım, başka bir şey sorun."

def load_and_combine_datasets():
    print("Downloading TFLai/Turkish-Alpaca from HuggingFace...")
    try:
        df1 = pd.read_json("https://huggingface.co/datasets/TFLai/Turkish-Alpaca/resolve/main/data.json")
        print(f"Loaded {len(df1)} rows from TFLai/Turkish-Alpaca.")
    except Exception as e:
        print(f"Failed to load TFLai/Turkish-Alpaca: {e}. Creating empty fallback.")
        df1 = pd.DataFrame(columns=["instruction", "input", "output"])

    print("Downloading merve/turkish_instructions from HuggingFace...")
    try:
        df2 = pd.read_csv("https://huggingface.co/datasets/merve/turkish_instructions/resolve/main/instructions.csv")
        df2.columns = [c.strip() for c in df2.columns]
        # Rename columns to match
        df2 = df2.rename(columns={"talimat": "instruction", "giriş": "input", "çıktı": "output"})
        print(f"Loaded {len(df2)} rows from merve/turkish_instructions.")
    except Exception as e:
        print(f"Failed to load merve/turkish_instructions: {e}. Creating empty fallback.")
        df2 = pd.DataFrame(columns=["instruction", "input", "output"])

    combined = []
    
    # Process df1
    for _, row in df1.iterrows():
        inst = str(row.get("instruction", "")).strip()
        inp = str(row.get("input", "")).strip()
        out = str(row.get("output", "")).strip()
        if not inst or not out:
            continue
        prompt = f"{inst}\n\nGirdi:\n{inp}" if inp else inst
        combined.append({
            "prompt": prompt,
            "chosen": out,
            "konu": "alpaca-tr"
        })
        
    # Process df2
    for _, row in df2.iterrows():
        inst = str(row.get("instruction", "")).strip()
        inp = str(row.get("input", "")).strip()
        out = str(row.get("output", "")).strip()
        # Handle nan strings
        if inp.lower() == "nan" or inp == "nan":
            inp = ""
        if not inst or not out:
            continue
        prompt = f"{inst}\n\nGirdi:\n{inp}" if inp else inst
        combined.append({
            "prompt": prompt,
            "chosen": out,
            "konu": "turkish_instructions"
        })
        
    print(f"Combined dataset contains {len(combined)} total samples.")
    return combined

def sample_and_label(combined, output_path, model_name, sample_size):
    print(f"Sampling {sample_size} records for Claude quality evaluation...")
    sampled = random.sample(combined, min(len(combined), sample_size))
    
    labeled_data = []
    for i, item in enumerate(sampled):
        prompt = item["prompt"]
        chosen = item["chosen"]
        
        evaluation_prompt = f"""Aşağıdaki Türkçe soru ve cevap çiftini 0-5 arası bir kalite puanıyla değerlendir:
0 = tamamen saçma/yanlış/anlamsız/çeviri hatası
5 = son derece doğal, dil bilgisi olarak doğru, bilgilendirici ve kaliteli Türkçe cevap

Sadece şu JSON formatında cevap ver: {{"puan": X, "sebep": "kısa açıklama"}}

SORU:
{prompt[:1000]}

CEVAP:
{chosen[:1500]}
"""
        try:
            raw_res = call_proxy_api(evaluation_prompt, model_name)
            if "```json" in raw_res:
                raw_res = raw_res.split("```json")[1].split("```")[0].strip()
            elif "```" in raw_res:
                raw_res = raw_res.split("```")[1].split("```")[0].strip()
            res = json.loads(raw_res)
            puan = float(res.get("puan", 0))
            sebep = res.get("sebep", "")
        except Exception as e:
            print(f"[{i+1}/{len(sampled)}] Evaluation failed: {e}. Defaulting to 1.")
            puan = 1.0
            sebep = "Değerlendirme hatası"
            
        print(f"[{i+1}/{len(sampled)}] Puan: {puan} | Sebep: {sebep}")
        labeled_data.append({
            "prompt": prompt,
            "chosen": chosen,
            "puan": puan,
            "sebep": sebep
        })
        
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for item in labeled_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Successfully saved {len(labeled_data)} labeled samples to {output_path}")

def train_classifier(labeled_path, classifier_path):
    print(f"Loading labeled samples from {labeled_path}...")
    items = []
    with open(labeled_path, "r", encoding="utf-8") as f:
        for line in f:
            items.append(json.loads(line))
            
    # Positive threshold is >= 3.0 out of 5
    pos_items = [it for it in items if it["puan"] >= 3.0]
    neg_items = [it for it in items if it["puan"] < 3.0]
    print(f"Labels summary: {len(pos_items)} positive, {len(neg_items)} negative.")
    
    # Balance classes via oversampling
    if len(pos_items) > 0 and len(neg_items) > 0:
        target = max(len(pos_items), len(neg_items))
        while len(pos_items) < target:
            pos_items.extend(pos_items[:target - len(pos_items)])
        while len(neg_items) < target:
            neg_items.extend(neg_items[:target - len(neg_items)])
            
    balanced = pos_items + neg_items
    random.shuffle(balanced)
    
    texts = [f"{it['prompt']} [SEP] {it['chosen']}" for it in balanced]
    labels = [1 if it["puan"] >= 3.0 else 0 for it in balanced]
    
    print("Loading Model2Vec static embedder...")
    model_name = "minishlab/potion-base-8M"
    try:
        embedder = StaticModel.from_pretrained(model_name)
    except Exception:
        embedder = StaticModel.from_pretrained("minishlab/M2V_base_glove")
        model_name = "minishlab/M2V_base_glove"
        
    print("Encoding embeddings...")
    X = embedder.encode(texts)
    y = np.array(labels)
    
    print("Fitting Logistic Regression classifier...")
    clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    clf.fit(X, y)
    
    joblib.dump({
        "classifier": clf,
        "embedding_model_name": model_name
    }, classifier_path)
    print(f"Classifier saved successfully to {classifier_path}")

def sanitize_record(text):
    if not isinstance(text, str):
        return text
    text = text.replace("Kiro", "Maya")
    text = text.replace("kiro", "maya")
    text = text.replace("KIRO", "MAYA")
    return text

def filter_and_save_dataset(combined, classifier_path, output_path):
    print(f"Loading classifier from {classifier_path}...")
    save_data = joblib.load(classifier_path)
    clf = save_data["classifier"]
    model_name = save_data.get("embedding_model_name", "minishlab/potion-base-8M")
    embedder = StaticModel.from_pretrained(model_name)
    
    print("Scoring combined dataset...")
    # Prepare inputs
    inputs = [f"{it['prompt']} [SEP] {it['chosen']}" for it in combined]
    
    # Process in batches to save memory
    batch_size = 5000
    scores = []
    print("Running batch predictions...")
    for i in range(0, len(inputs), batch_size):
        batch = inputs[i:i+batch_size]
        embs = embedder.encode(batch)
        probs = clf.predict_proba(embs)[:, 1]
        scores.extend(probs.tolist())
        if (i // batch_size) % 3 == 0:
            print(f"Processed {min(i+batch_size, len(inputs)):,} / {len(inputs):,} samples.")
            
    # Filter and construct dataset
    curated = []
    for item, score in zip(combined, scores):
        if score >= 0.65: # Slightly lower threshold to ensure we capture maximum quality instructions
            # Generate negative response for ORPO
            rejected = generate_negative_response(item["chosen"])
            
            # Sanitize Kiro identity
            prompt = sanitize_record(item["prompt"])
            chosen = sanitize_record(item["chosen"])
            rejected = sanitize_record(rejected)
            
            curated.append({
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
                "konu": item["konu"],
                "quality_score": float(score)
            })
            
    print(f"Accepted {len(curated):,} high quality samples out of {len(combined):,}.")
    
    # Save to file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for item in curated:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
    print(f"Successfully saved curated dataset to {output_path} (Total: {len(curated):,} records)")

def main():
    parser = argparse.ArgumentParser(description="Turkish Alpaca Instruction Quality Curation and ORPO Generator")
    parser.add_argument("--mode", type=str, required=True, choices=["sample-and-label", "train-classifier", "filter-dataset"], help="Execution mode")
    parser.add_argument("--sample_size", type=int, default=100, help="Number of samples to label with Claude")
    parser.add_argument("--labeled_jsonl", type=str, default="shared/claude_chat_labels.jsonl", help="Path to save Claude graded records")
    parser.add_argument("--classifier_path", type=str, default="shared/chat_quality_classifier.bin", help="Path to save quality classifier")
    parser.add_argument("--output_path", type=str, default="shared/chat_data_orpo_100k.jsonl", help="Curated output path")
    parser.add_argument("--model", type=str, default="claude-sonnet-4.5", help="Model name for Claude evaluations")
    args = parser.parse_args()
    
    if args.mode == "sample-and-label":
        combined = load_and_combine_datasets()
        sample_and_label(combined, args.labeled_jsonl, args.model, args.sample_size)
    elif args.mode == "train-classifier":
        train_classifier(args.labeled_jsonl, args.classifier_path)
    elif args.mode == "filter-dataset":
        combined = load_and_combine_datasets()
        filter_and_save_dataset(combined, args.classifier_path, args.output_path)

if __name__ == "__main__":
    main()
