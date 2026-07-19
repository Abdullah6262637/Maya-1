import os
import sys
import json
import argparse
import joblib
from model2vec import StaticModel

# Pre-defined high-quality multi-turn dialogue templates in Turkish
# These are hand-crafted (LIMA style) to teach the model multi-turn conversational structures.
GOLDEN_MULTI_TURN_DIALOGUES = [
    {
        "turns": [
            {"role": "user", "content": "Selam, nasılsın?"},
            {"role": "assistant", "content": "Selam! Ben bir yapay zeka modeliyim, her şey yolunda ve sana yardım etmeye hazırım. Sen nasılsın, günün nasıl geçiyor?"},
            {"role": "user", "content": "Biraz yoğun ama idare eder. Bana hızlıca gevşemek için 3 tüyo verir misin?"},
            {"role": "assistant", "content": "Elbette! İşte gün ortasında hızlıca rahatlamana yardımcı olacak 3 pratik tüyo:\n\n1. **4-7-8 Nefes Egzersizi:** 4 saniye nefes al, 7 saniye tut ve 8 saniyede yavaşça ver. Bu, sinir sistemini hemen sakinleştirir.\n2. **Kısa Bir Esneme:** Omuzlarını geriye doğru yuvarla, boynunu yavaşça sağa sola yatır ve ayağa kalkıp 1 dakika yürü.\n3. **Ekran Molası:** Gözlerini 20 saniye boyunca kapat veya en az 6 metre uzaktaki bir nesneye bakarak göz kaslarını dinlendir."}
        ]
    },
    {
        "turns": [
            {"role": "user", "content": "Merhaba, adın ne?"},
            {"role": "assistant", "content": "Merhaba! Benim adım Maya. Sana yardımcı olmak için geliştirilmiş bir Türkçe yapay zeka asistanıyım."},
            {"role": "user", "content": "Harika. Peki kim tarafından geliştirildin?"},
            {"role": "assistant", "content": "Ben, Türk mühendisleri ve araştırmacıları tarafından, en ileri derin öğrenme teknikleri (Muon, MTP ve muP mimarileri) kullanılarak geliştirildim."}
        ]
    },
    {
        "turns": [
            {"role": "user", "content": "Python'da liste tersine nasıl çevrilir?"},
            {"role": "assistant", "content": "Python'da bir listeyi tersine çevirmenin en yaygın 3 yolu şunlardır:\n\n1. **Dilimleme (Slicing) yöntemi (En hızlısı):**\n```python\nliste = [1, 2, 3]\nters_liste = liste[::-1]\n```\n2. **`reverse()` metodu (Listeyi yerinde değiştirir):**\n```python\nliste = [1, 2, 3]\nliste.reverse()\n```\n3. **`reversed()` fonksiyonu (Bir iterator döner):**\n```python\nliste = [1, 2, 3]\nters_liste = list(reversed(liste))\n```"},
            {"role": "user", "content": "Peki ilk yöntem orijinal listeyi değiştirir mi?"},
            {"role": "assistant", "content": "Hayır, dilimleme (`liste[::-1]`) yöntemi orijinal listeyi değiştirmez. Orijinal listenin ters çevrilmiş yeni bir kopyasını oluşturur. Orijinal listeyi yerinde değiştirmek istiyorsanız ikinci yöntem olan `liste.reverse()` metodunu kullanmalısınız."}
        ]
    }
]

def load_classifier(classifier_path):
    if not os.path.exists(classifier_path):
        print(f"Warning: Classifier file not found at {classifier_path}. Using fallback dummy scoring.")
        return None, None
    try:
        save_data = joblib.load(classifier_path)
        clf = save_data["classifier"]
        model_name = save_data.get("embedding_model_name", "minishlab/potion-base-8M")
        return clf, model_name
    except Exception as e:
        print(f"Error loading classifier: {e}")
        return None, None

def curate_dataset(args):
    print(f"Loading quality classifier from: {args.classifier_path}")
    clf, model_name = load_classifier(args.classifier_path)
    
    if clf:
        print(f"Loading Model2Vec model '{model_name}'...")
        embedder = StaticModel.from_pretrained(model_name)
    else:
        embedder = None
        
    print(f"Reading raw data from: {args.input}")
    if not os.path.exists(args.input):
        print(f"Error: Input file {args.input} not found.")
        sys.exit(1)
        
    records = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
                
    print(f"Total raw records loaded: {len(records)}")
    
    curated_records = []
    
    if clf and embedder:
        print("Filtering records using Few-Shot Quality Classifier (threshold >= 0.70)...")
        # Extract prompt responses for quality filtering
        texts = []
        for r in records:
            # We evaluate the quality based on the response text
            if "response" in r:
                texts.append(r["response"])
            elif "chosen" in r:
                texts.append(r["chosen"])
            else:
                texts.append(r.get("prompt", ""))
                
        # Generate embeddings in batch
        embeddings = embedder.encode(texts)
        probabilities = clf.predict_proba(embeddings)[:, 1]  # Probabilities of class '1' (kaliteli)
        
        for record, prob in zip(records, probabilities):
            if prob >= 0.70:
                # Add quality score metadata
                record["quality_score"] = float(prob)
                curated_records.append(record)
    else:
        # Fallback to keep everything if classifier is not found
        print("No classifier loaded. Keeping all input records as-is.")
        curated_records = records
        
    print(f"Accepted after filtering: {len(curated_records)} / {len(records)} records.")
    
    # Inject LIMA multi-turn dialogues
    print(f"Injecting {len(GOLDEN_MULTI_TURN_DIALOGUES)} hand-crafted golden multi-turn dialogues...")
    for diag in GOLDEN_MULTI_TURN_DIALOGUES:
        # Depending on whether mode is ORPO or SFT, map the template
        if args.mode == "orpo":
            # For ORPO, multi-turn dialogues can be represented as a chat prompt format:
            # prompt: turns[:-1] formatted as chat history, chosen: turns[-1]['content'], rejected: dummy/empty
            turns = diag["turns"]
            # Convert previous turns to format
            history = ""
            for t in turns[:-1]:
                rol = "<|kullanici|>" if t["role"] == "user" else "<|asistan|>"
                history += f"{rol}{t['content']}<|son|>"
            
            prompt = history + "<|asistan|>"
            chosen = turns[-1]["content"] + "<|son|>"
            rejected = "Bilmiyorum." + "<|son|>"
            
            curated_records.append({
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
                "konu": "golden-multi-turn"
            })
        else:
            # For SFT, format each turn cleanly or add as multi-turn list
            curated_records.append({
                "turns": diag["turns"],
                "konu": "golden-multi-turn"
            })
            
    # Save the final curated dataset
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for r in curated_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            
    print(f"Curated dataset saved successfully to {args.output} (Total: {len(curated_records)} records)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LIMA-style Quality Curation and Golden Dialogues Injection")
    parser.add_argument("--input", type=str, required=True, help="Path to raw JSONL data generated in Phase 1")
    parser.add_argument("--output", type=str, required=True, help="Path to save the curated JSONL dataset")
    parser.add_argument("--classifier_path", type=str, default="shared/quality_classifier_test.bin", help="Path to trained Model2Vec classifier")
    parser.add_argument("--mode", type=str, choices=["sft", "orpo"], default="orpo", help="Target dataset mode")
    args = parser.parse_args()
    
    curate_dataset(args)
