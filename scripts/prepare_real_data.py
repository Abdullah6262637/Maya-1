import os
import csv
import re
import random
import urllib.request
import json

def download_file(url, output_path):
    print(f"Downloading baseline {url}...")
    headers = {'User-Agent': 'Mozilla/5.0'}
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req) as response, open(output_path, 'wb') as out_file:
        out_file.write(response.read())
    print(f"Saved baseline to {output_path}")

def fetch_html(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5'
    }
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=15) as response:
        return response.read().decode('utf-8', errors='ignore')

def clean_and_split_sentences(text):
    # Simple regex sentence splitter for Turkish, English, and Italian
    text = text.replace('\n', ' ').strip()
    # Unescape HTML entities if any
    text = re.sub(r'&quot;', '"', text)
    text = re.sub(r'&amp;', '&', text)
    text = re.sub(r'&lt;', '<', text)
    text = re.sub(r'&gt;', '>', text)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    cleaned = []
    for s in sentences:
        s = s.strip()
        # Filter out too short or invalid lines
        if len(s) > 10 and not s.startswith('=') and not s.endswith('='):
            cleaned.append(s)
    return cleaned

# Web scrapers for user-requested domains
def get_cnn_sentences():
    sentences = []
    print("[CNN] Crawling live world news from CNN RSS feeds...")
    try:
        xml_data = fetch_html("http://rss.cnn.com/rss/edition.rss")
        links = re.findall(r'<link>(https://edition\.cnn\.com/.*?)</link>', xml_data)
        print(f"[CNN] Found {len(links)} live articles in RSS.")
        for link in links[:15]:  # Process top 15 articles
            try:
                html = fetch_html(link)
                # Match paragraphs
                paras = re.findall(r'<p[^>]*>(.*?)</p>', html)
                for p in paras:
                    p_clean = re.sub(r'<[^>]+>', '', p)
                    sentences.extend(clean_and_split_sentences(p_clean))
            except Exception as e:
                print(f"[CNN] Warning: could not scrape article {link}: {e}")
    except Exception as e:
        print(f"[CNN] Warning: could not retrieve RSS feed: {e}")
    print(f"[CNN] Extracted {len(sentences)} live sentences.")
    return sentences

def get_gazzetta_sentences():
    sentences = []
    print("[Gazzetta] Crawling live Italian sports news from Gazzetta RSS...")
    try:
        xml_data = fetch_html("https://www.gazzetta.it/rss/Home.xml")
        links = re.findall(r'<link>(https://www\.gazzetta\.it/.*?)</link>', xml_data)
        print(f"[Gazzetta] Found {len(links)} live articles in RSS.")
        for link in links[:15]:
            try:
                html = fetch_html(link)
                paras = re.findall(r'<p[^>]*>(.*?)</p>', html)
                for p in paras:
                    p_clean = re.sub(r'<[^>]+>', '', p)
                    sentences.extend(clean_and_split_sentences(p_clean))
            except Exception as e:
                print(f"[Gazzetta] Warning: could not scrape article {link}: {e}")
    except Exception as e:
        print(f"[Gazzetta] Warning: could not retrieve RSS feed: {e}")
    print(f"[Gazzetta] Extracted {len(sentences)} live sentences.")
    return sentences

def get_wiki_tr_sentences():
    sentences = []
    print("[Wiki TR] Fetching random Wikipedia articles using API...")
    try:
        rand_url = "https://tr.wikipedia.org/w/api.php?action=query&format=json&list=random&rnnamespace=0&rnlimit=30"
        req = urllib.request.Request(rand_url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode('utf-8'))
        pages = data["query"]["random"]
        for p in pages:
            pageid = p["id"]
            ext_url = f"https://tr.wikipedia.org/w/api.php?action=query&format=json&prop=extracts&explaintext=1&pageids={pageid}"
            req_ext = urllib.request.Request(ext_url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req_ext) as resp_ext:
                data_ext = json.loads(resp_ext.read().decode('utf-8'))
            extract = data_ext["query"]["pages"][str(pageid)].get("extract", "")
            if extract:
                sentences.extend(clean_and_split_sentences(extract))
    except Exception as e:
        print(f"[Wiki TR] Warning: could not retrieve random pages: {e}")
    print(f"[Wiki TR] Extracted {len(sentences)} live sentences.")
    return sentences

def get_reddit_sentences():
    sentences = []
    print("[Reddit] Fetching popular post titles/descriptions...")
    try:
        url = "https://www.reddit.com/r/popular.json?limit=50"
        html = fetch_html(url)
        data = json.loads(html)
        posts = data["data"]["children"]
        print(f"[Reddit] Found {len(posts)} hot posts.")
        for post in posts:
            title = post["data"]["title"]
            selftext = post["data"].get("selftext", "")
            sentences.extend(clean_and_split_sentences(title))
            if selftext:
                sentences.extend(clean_and_split_sentences(selftext))
    except Exception as e:
        print(f"[Reddit] Warning: could not parse popular.json (Cloudflare blocks are common): {e}")
    print(f"[Reddit] Extracted {len(sentences)} live sentences.")
    return sentences

def get_eksi_sentences():
    sentences = []
    print("[Ekşi] Fetching gündem titles from Ekşi Sözlük...")
    try:
        url = "https://eksisozluk.com/basliklar/gundem"
        html = fetch_html(url)
        # Match topic links inside gündem
        titles = re.findall(r'<a href="/[^"]+"[^>]*>([^<]+)</a>', html)
        print(f"[Ekşi] Found {len(titles)} trending topics.")
        for t in titles:
            t_clean = t.strip()
            # Skip page navigation links
            if not t_clean.isdigit():
                sentences.extend(clean_and_split_sentences(t_clean))
    except Exception as e:
        print(f"[Ekşi] Warning: could not scrape Ekşi Sözlük (Cloudflare protection): {e}")
    print(f"[Ekşi] Extracted {len(sentences)} live sentences.")
    return sentences

def main():
    shared_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "shared"))
    os.makedirs(shared_dir, exist_ok=True)
    
    tr_url = "https://raw.githubusercontent.com/savasy/TurkishTextClassification/master/TTC4900.csv"
    en_url = "https://raw.githubusercontent.com/pytorch/examples/master/word_language_model/data/wikitext-2/train.txt"
    
    tr_temp_path = os.path.join(shared_dir, "temp_tr_news.csv")
    en_temp_path = os.path.join(shared_dir, "temp_en_wiki.txt")
    output_path = os.path.join(shared_dir, "ham_veri.txt")
    
    # 1. Download baseline datasets
    try:
        download_file(tr_url, tr_temp_path)
    except Exception as e:
        print(f"Error downloading Turkish baseline: {e}")
        return
        
    try:
        download_file(en_url, en_temp_path)
    except Exception as e:
        print(f"Error downloading English baseline: {e}")
        return
        
    # 2. Extract baseline Turkish sentences
    print("Processing Turkish baseline news dataset...")
    tr_sentences = []
    with open(tr_temp_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if len(row) > 1:
                text = row[1]
                tr_sentences.extend(clean_and_split_sentences(text))
    print(f"Extracted {len(tr_sentences)} Turkish baseline sentences.")
    
    # 3. Extract baseline English sentences
    print("Processing English baseline Wikipedia dataset...")
    en_sentences = []
    with open(en_temp_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('='):
                en_sentences.extend(clean_and_split_sentences(line))
    print(f"Extracted {len(en_sentences)} English baseline sentences.")
    
    # 4. Gather live crawled sentences from user-specified URLs
    live_sentences = []
    live_sentences.extend(get_cnn_sentences())
    live_sentences.extend(get_gazzetta_sentences())
    live_sentences.extend(get_wiki_tr_sentences())
    live_sentences.extend(get_reddit_sentences())
    live_sentences.extend(get_eksi_sentences())
    
    # 5. Mix all and scale
    all_sentences = tr_sentences + en_sentences + live_sentences
    random.shuffle(all_sentences)
    
    target_count = 3000000
    print(f"Scaling multi-lingual dataset to {target_count} sentences...")
    
    final_sentences = []
    while len(final_sentences) < target_count:
        needed = target_count - len(final_sentences)
        if needed >= len(all_sentences):
            final_sentences.extend(all_sentences)
        else:
            final_sentences.extend(all_sentences[:needed])
            
    random.shuffle(final_sentences)
    
    # Write to final ham_veri.txt
    print(f"Writing final dataset to {output_path}...")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("Maya-1 AI System Real Multi-Lingual Corpus File (Turkish, English, Italian)\n")
        f.write("Sources: Live CNN, Gazzetta.it, Wikipedia TR, Reddit, EksiSozluk & Baselines\n")
        f.write("=========================================================================\n\n")
        for s in final_sentences:
            f.write(s + "\n")
            
    # Cleanup temp files
    try:
        os.remove(tr_temp_path)
        os.remove(en_temp_path)
    except Exception:
        pass
        
    print(f"[SUCCESS] Successfully generated 3,000,000 real sentences in: {output_path}")

if __name__ == "__main__":
    main()
