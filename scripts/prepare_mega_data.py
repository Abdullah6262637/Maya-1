import os
import re
import json
import random
import urllib.request
import urllib.parse

def fetch_html(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5'
    }
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=15) as response:
        return response.read().decode('utf-8', errors='ignore')

def clean_html(text):
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'&quot;', '"', text)
    text = re.sub(r'&amp;', '&', text)
    text = re.sub(r'&lt;', '<', text)
    text = re.sub(r'&gt;', '>', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def main():
    print("=== Maya-1 Mega Crawler & Dataset Builder ===")
    
    shared_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "shared"))
    os.makedirs(shared_dir, exist_ok=True)
    output_path = os.path.join(shared_dir, "mega_sft_data.json")
    
    dataset = []
    
    # 1. Fetch CNN (English News)
    print("[CNN] Crawling live news...")
    try:
        xml = fetch_html("http://rss.cnn.com/rss/edition.rss")
        links = re.findall(r'<link>(https://edition\.cnn\.com/.*?)</link>', xml)
        print(f"[CNN] Found {len(links)} articles.")
        for link in links[:15]:
            try:
                html = fetch_html(link)
                title = re.search(r'<title>(.*?)</title>', html)
                title_text = clean_html(title.group(1)) if title else "CNN News Article"
                paras = re.findall(r'<p[^>]*>(.*?)</p>', html)
                body = " ".join([clean_html(p) for p in paras if len(clean_html(p)) > 30])
                if len(body) > 100:
                    dataset.append({
                        "messages": [
                            {"role": "user", "content": f"Summarize this news article: {title_text}"},
                            {"role": "assistant", "content": body[:800] + "..."}
                        ]
                    })
            except Exception as e:
                print(f"[CNN] Skipped article {link}: {e}")
    except Exception as e:
        print(f"[CNN] Failed to get RSS: {e}")

    # 2. Fetch Gazzetta.it (Italian Sports News - corrected lowercase home.xml)
    print("[Gazzetta] Crawling live Italian sports news...")
    try:
        xml = fetch_html("https://www.gazzetta.it/rss/home.xml")
        links = re.findall(r'<link>(https://www\.gazzetta\.it/.*?)</link>', xml)
        print(f"[Gazzetta] Found {len(links)} articles.")
        for link in links[:15]:
            try:
                html = fetch_html(link)
                title = re.search(r'<title>(.*?)</title>', html)
                title_text = clean_html(title.group(1)) if title else "Articolo di Gazzetta"
                paras = re.findall(r'<p[^>]*>(.*?)</p>', html)
                body = " ".join([clean_html(p) for p in paras if len(clean_html(p)) > 30])
                if len(body) > 100:
                    dataset.append({
                        "messages": [
                            {"role": "user", "content": f"Di cosa parla questo articolo sportivo: {title_text}"},
                            {"role": "assistant", "content": body[:800] + "..."}
                        ]
                    })
            except Exception as e:
                print(f"[Gazzetta] Skipped article {link}: {e}")
    except Exception as e:
        print(f"[Gazzetta] Failed to get RSS: {e}")

    # 3. Fetch Wikipedia TR (Turkish Knowledge)
    print("[Wikipedia TR] Crawling random summaries...")
    try:
        rand_url = "https://tr.wikipedia.org/w/api.php?action=query&format=json&list=random&rnnamespace=0&rnlimit=30"
        req = urllib.request.Request(rand_url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode('utf-8'))
        pages = data["query"]["random"]
        for p in pages:
            pageid = p["id"]
            title_text = p["title"]
            ext_url = f"https://tr.wikipedia.org/w/api.php?action=query&format=json&prop=extracts&explaintext=1&pageids={pageid}"
            req_ext = urllib.request.Request(ext_url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req_ext) as resp_ext:
                data_ext = json.loads(resp_ext.read().decode('utf-8'))
            extract = data_ext["query"]["pages"][str(pageid)].get("extract", "")
            if len(extract) > 100:
                dataset.append({
                    "messages": [
                        {"role": "user", "content": f"\"{title_text}\" konusu hakkında bana bilgi ver."},
                        {"role": "assistant", "content": extract[:1000]}
                    ]
                })
    except Exception as e:
        print(f"[Wikipedia TR] Failed: {e}")

    # 4. Fetch Reddit (English Dialogue / Trends)
    print("[Reddit] Fetching hot posts...")
    try:
        url = "https://www.reddit.com/r/popular.json?limit=30"
        html = fetch_html(url)
        data = json.loads(html)
        posts = data["data"]["children"]
        for post in posts:
            title = clean_html(post["data"]["title"])
            selftext = clean_html(post["data"].get("selftext", ""))
            subreddit = post["data"]["subreddit"]
            if len(title) > 10:
                ans = f"This is a popular discussion topic on Reddit's r/{subreddit}. Title: {title}."
                if selftext:
                    ans += f" Details: {selftext[:500]}..."
                dataset.append({
                    "messages": [
                        {"role": "user", "content": f"What is trending on Reddit r/{subreddit}?"},
                        {"role": "assistant", "content": ans}
                    ]
                })
    except Exception as e:
        print(f"[Reddit] Failed (Cloudflare/Rate Limit): {e}")

    # 5. Fetch Ekşi Sözlük (Turkish Slang / Colloquial)
    print("[Ekşi Sözlük] Crawling popular topics...")
    try:
        html = fetch_html("https://eksisozluk.com/basliklar/gundem")
        titles = re.findall(r'<a href="/[^"]+"[^>]*>([^<]+)</a>', html)
        for t in titles:
            t_clean = clean_html(t)
            if len(t_clean) > 5 and not t_clean.isdigit():
                dataset.append({
                    "messages": [
                        {"role": "user", "content": f"Ekşi Sözlük'te \"{t_clean}\" başlığı hakkında ne düşünülüyor?"},
                        {"role": "assistant", "content": f"\"{t_clean}\" başlığı Ekşi Sözlük'te bugün çok konuşulan konulardan biri. Genellikle kullanıcılar bu konuda mizahi, eleştirel veya gündelik yorumlar yazıyorlar."}
                    ]
                })
    except Exception as e:
        print(f"[Ekşi Sözlük] Failed (Cloudflare): {e}")

    # 6. High-Quality Custom Translation (TR-EN-IT) & General Chat Dialogue
    print("[Custom] Injecting high-quality dialogue & translation datasets...")
    dialogue_pairs = [
        ("selam", "Merhaba! Ben Maya-1. Bugün size nasıl yardımcı olabilirim?"),
        ("selamlar", "Selamlar! Ben Maya-1. Size yardımcı olmak için buradayım."),
        ("merhaba", "Merhaba! Ben Maya-1, çok dilli bir yapay zeka asistanıyım. Nasıl yardımcı olabilirim?"),
        ("nasılsın", "Teşekkür ederim, iyiyim! Siz nasılsınız? Size nasıl destek olabilirim?"),
        ("nasılsın?", "Harikayım, teşekkürler! Siz nasılsınız, bugün nasıl gidiyor?"),
        ("kimsin sen", "Ben Maya-1, internet verileriyle ve modern Transformer mimarisiyle eğitilmiş çok dilli bir yapay zeka asistanıyım."),
        ("sen kimsin?", "Ben Maya-1 mimarili bir büyük dil modeliyim. Türkçe, İngilizce ve İtalyanca dillerini anlar ve konuşabilirim."),
        ("what is your name?", "My name is Maya-1. I am a multilingual AI assistant."),
        ("come ti chiami?", "Mi chiamo Maya-1, sono un assistente virtuale basato su intelligenza artificiale."),
        ("türkiye'nin başkenti neresidir?", "Türkiye'nin başkenti Ankara'dır."),
        ("what is the capital of italy?", "The capital of Italy is Rome."),
        ("what is the capital of turkey?", "The capital of Turkey is Ankara."),
        ("roma nerenin başkentidir?", "Roma, İtalya'nın başkentidir."),
        ("can you translate?", "Yes! I can translate between Turkish, English, and Italian. Please write the text you want to translate."),
        ("çeviri yapabilir misin?", "Evet, Türkçe, İngilizce ve İtalyanca arasında çeviri yapabilirim. Çevirmek istediğiniz metni yazabilirsiniz."),
        
        # Translation examples (TR -> EN)
        ("Türkçeye çevir: 'Hello, how are you today?'", "Türkçe çevirisi: 'Merhaba, bugün nasılsın?'"),
        ("İngilizceye çevir: 'Yarın hava çok güzel olacak.'", "English translation: 'The weather will be very beautiful tomorrow.'"),
        ("Translate to English: 'Yapay zeka hayatımızı kolaylaştırıyor.'", "English translation: 'Artificial intelligence makes our lives easier.'"),
        ("Türkçeye çevir: 'I love learning new things.'", "Türkçe çevirisi: 'Yeni şeyler öğrenmeyi seviyorum.'"),
        
        # Translation examples (EN -> IT)
        ("Translate to Italian: 'Welcome to our project.'", "Italian translation: 'Benvenuti nel nostro progetto.'"),
        ("Translate to Italian: 'Thank you very much.'", "Italian translation: 'Grazie mille.'"),
        
        # Code requests
        ("yazılım nedir?", "Yazılım, bilgisayarların belirli görevleri yerine getirmesini sağlayan kodlar, yönergeler ve programlar bütünüdür."),
        ("write a python hello world code", "Here is the Python 'Hello World' code:\n```python\nprint(\"Hello, World!\")\n```"),
        ("python hello world yaz", "İşte Python ile 'Hello World' kodu:\n```python\nprint(\"Hello, World!\")\n```")
    ]
    
    for user_msg, assist_msg in dialogue_pairs:
        dataset.append({
            "messages": [
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": assist_msg}
            ]
        })
        
    # Write SFT dataset
    print(f"Writing dataset with {len(dataset)} items to: {output_path}")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
        
    print("[SUCCESS] Mega SFT Dataset created successfully!")

if __name__ == "__main__":
    main()
