import os
import sys
import json
import argparse
import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

class OpenAICompatClient:
    def __init__(self, api_key, base_url):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        
    class Messages:
        def __init__(self, parent):
            self.parent = parent
            
        def create(self, model, max_tokens, temperature, messages):
            import requests
            headers = {
                "Authorization": f"Bearer {self.parent.api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature
            }
            url = f"{self.parent.base_url}/chat/completions"
            response = requests.post(url, headers=headers, json=payload)
            if response.status_code != 200:
                raise RuntimeError(f"API Error ({response.status_code}): {response.text}")
            
            data = response.json()
            text = data["choices"][0]["message"]["content"]
            
            class Content:
                def __init__(self, t):
                    self.text = t
            class Response:
                def __init__(self, t):
                    self.content = [Content(t)]
            return Response(text)

    @property
    def messages(self):
        return self.Messages(self)

# Curated pool of topics/domains for diversity in Turkish instruction generation
KONU_HAVUZU = [
    "günlük sohbet ve selamlaşma",
    "genel kültür sorusu",
    "basit matematik ve mantık bulmacası",
    "hayat tavsiyesi ve kişisel gelişim",
    "teknik/yazılım geliştirme sorusu",
    "duygusal destek ve empati arayışı",
    "yaratıcı metin yazarlığı (hikaye, şiir, e-posta)",
    "detaylı açıklama ve analiz talebi",
    "Türkçe dilbilgisi ve kelime anlamı",
    "günlük hayat pratik bilgileri (yemek tarifi, temizlik vb.)",
    "rol yapma (roleplay) senaryosu",
    "seyahat planlama ve turizm önerileri"
]

def get_client(api_key=None, base_url=None):
    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    url = base_url or os.environ.get("ANTHROPIC_BASE_URL")
    if not key:
        print("Warning: API key variable not found.")
        return None
    if url:
        print(f"Using OpenAI-compatible endpoint: {url}")
        return OpenAICompatClient(key, url)
    try:
        import anthropic
        return anthropic.Anthropic(api_key=key)
    except ImportError:
        print("Error: 'anthropic' library not installed. Standard client unavailable.")
        return None

def generate_instruction(client, topic, model="claude-3-5-sonnet-20241022"):
    prompt = f"""Gerçekçi bir Türkçe kullanıcısının bir yapay zekaya sorabileceği,
'{topic}' kategorisinde son derece DOĞAL ve gerçekçi bir soru veya talimat üret.
- Sadece kullanıcının sorusunu/talimatını yaz, başka hiçbir şey ekleme (başlık, tırnak işareti, açıklama vb. olmasın).
- Cümle kısa veya orta uzunlukta, samimi, gerçek bir insanın elinden çıkmış gibi olsun.
- İmla hataları veya klavye kaymaları gibi doğal kusurlar (isteğe bağlı) içerebilir."""
    
    try:
        r = client.messages.create(
            model=model,
            max_tokens=150,
            temperature=0.7,
            messages=[{"role": "user", "content": prompt}]
        )
        return r.content[0].text.strip()
    except Exception as e:
        print(f"Error generating instruction for topic '{topic}': {e}")
        return None

def generate_responses_orpo(client, instruction, model="claude-3-5-sonnet-20241022"):
    # Generate chosen (high-quality) response
    chosen_prompt = f"""Bir Türkçe yapay zeka asistanı olarak şu soruya/talimata son derece doğal, yardımsever, açıklayıcı ve samimi bir cevap ver: "{instruction}" """
    
    # Generate rejected (low-quality) response
    rejected_prompt = f"""'{instruction}' sorusuna KASITLI OLARAK kısa, kaçamak, yetersiz, kaba veya yarım bırakılmış bir cevap yaz (kötü örnek olarak kullanılacak). Sadece bu kötü cevabı yaz."""
    
    try:
        # Generate chosen response
        r_chosen = client.messages.create(
            model=model,
            max_tokens=500,
            temperature=0.5,
            messages=[{"role": "user", "content": chosen_prompt}]
        )
        chosen_text = r_chosen.content[0].text.strip()
        
        # Generate rejected response
        r_rejected = client.messages.create(
            model=model,
            max_tokens=200,
            temperature=0.8,
            messages=[{"role": "user", "content": rejected_prompt}]
        )
        rejected_text = r_rejected.content[0].text.strip()
        
        return chosen_text, rejected_text
    except Exception as e:
        print(f"Error generating responses for instruction: {e}")
        return None, None

def worker_task(client, topic, mode, model):
    instruction = generate_instruction(client, topic, model)
    if not instruction:
        return None
        
    if mode == "orpo":
        chosen, rejected = generate_responses_orpo(client, instruction, model)
        if chosen and rejected:
            return {
                "prompt": instruction,
                "chosen": chosen,
                "rejected": rejected,
                "konu": topic
            }
    else:  # sft mode
        # Just generate chosen response as the target SFT response
        chosen_prompt = f"""Bir Türkçe yapay zeka asistanı olarak şu soruya/talimata son derece doğal, yardımsever ve samimi bir cevap ver: "{instruction}" """
        try:
            r = client.messages.create(
                model=model,
                max_tokens=500,
                temperature=0.5,
                messages=[{"role": "user", "content": chosen_prompt}]
            )
            response = r.content[0].text.strip()
            return {
                "instruction": instruction,
                "response": response,
                "konu": topic
            }
        except Exception as e:
            print(f"Error generating response: {e}")
            
    return None

def main():
    parser = argparse.ArgumentParser(description="Generate Senthetic Chat Dataset using Claude API (Magpie & Preference Pairs)")
    parser.add_argument("--api_key", type=str, default=None, help="API Key (Anthropic or OpenAI-compatible)")
    parser.add_argument("--base_url", type=str, default=None, help="OpenAI-compatible Base URL")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to generate")
    parser.add_argument("--mode", type=str, choices=["sft", "orpo"], default="orpo", help="Dataset target mode (sft or orpo)")
    parser.add_argument("--output", type=str, default=None, help="Output path for JSONL data")
    parser.add_argument("--model", type=str, default="claude-3-5-sonnet-20241022", help="Model name")
    parser.add_argument("--concurrency", type=int, default=3, help="Number of concurrent API request threads")
    args = parser.parse_args()
    
    # Determine output path if not specified
    if not args.output:
        args.output = f"shared/chat_data_{args.mode}.jsonl"
        
    client = get_client(args.api_key, args.base_url)
    if not client:
        # If API key is not present, we will write mock records as a fallback demonstration
        print("\n=== OFFLINE/DEMO MODE ===")
        print("API Key not provided. Writing sample demonstration records directly to output.")
        mock_data = []
        if args.mode == "orpo":
            mock_data = [
                {
                    "prompt": "selam nasılsın",
                    "chosen": "Selam! Çok iyiyim, teşekkür ederim. Sana nasıl yardımcı olabilirim bugün?",
                    "rejected": "selam iyi",
                    "konu": "günlük sohbet"
                },
                {
                    "prompt": "Türkiye'nin başkenti neresidir?",
                    "chosen": "Türkiye'nin başkenti Ankara'dır. Ankara, 13 Ekim 1923'te resmi olarak başkent ilan edilmiştir.",
                    "rejected": "biliyorum ama söylemem.",
                    "konu": "genel kültür"
                }
            ]
        else:
            mock_data = [
                {
                    "instruction": "selam",
                    "response": "Merhaba! Sana nasıl yardımcı olabilirim?",
                    "konu": "günlük sohbet"
                },
                {
                    "instruction": "Nasılsın?",
                    "response": "Ben bir yapay zeka modeliyim, duygularım yok ama sistemim harika çalışıyor! Sen nasılsın?",
                    "konu": "günlük sohbet"
                }
            ]
            
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            for item in mock_data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"Sample demo data written to: {args.output}")
        return

    print(f"Starting synthetic generation in mode '{args.mode}' using {args.model}...")
    print(f"Targeting {args.num_samples} samples across {args.concurrency} concurrent threads...")
    
    results = []
    completed = 0
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = []
        for i in range(args.num_samples):
            topic = KONU_HAVUZU[i % len(KONU_HAVUZU)]
            futures.append(executor.submit(worker_task, client, topic, args.mode, args.model))
            
        for future in as_completed(futures):
            res = future.result()
            if res:
                results.append(res)
                # Write progressively to save state
                with open(args.output, "a", encoding="utf-8") as f:
                    f.write(json.dumps(res, ensure_ascii=False) + "\n")
                completed += 1
                if completed % 1 == 0:
                    print(f"Progress: {completed}/{args.num_samples} samples written.")
            # Rate limit backoff to be friendly to Anthropic API limits
            time.sleep(0.5)
            
    print(f"Generation completed successfully! Saved {completed} records to: {args.output}")

if __name__ == "__main__":
    main()
