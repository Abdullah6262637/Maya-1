# 🚀 Maya-1: Hiper-Hızlı Türkçe LLM, Hizalama ve Çıkarım Eko-Sistemi

<p align="center">
  <img src="https://raw.githubusercontent.com/andelf/readme-template/master/images/readme-banner.png" alt="Maya-1 Banner" width="100%">
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Architecture-MTP%20Decoder--Only-8C4FFF?style=for-the-badge&logo=cpu" alt="Architecture">
  <img src="https://img.shields.io/badge/Tokenizer-Turkish%20BPE%2050k-orange?style=for-the-badge&logo=google" alt="Tokenizer">
  <img src="https://img.shields.io/badge/Parameterization-muP-magenta?style=for-the-badge" alt="muP">
  <img src="https://img.shields.io/badge/Optimizer-Muon%20%2B%20AdamW-FF5733?style=for-the-badge&logo=rocket" alt="Optimizer">
  <img src="https://img.shields.io/badge/Alignment-ORPO%20%2B%20LIMA-blue?style=for-the-badge" alt="Alignment">
  <img src="https://img.shields.io/badge/Serving-OpenAI%20SSE-00e676?style=for-the-badge&logo=fastapi" alt="Serving">
</p>

---

## 📌 Proje Genel Bakışı

**Maya-1**, NVIDIA H100 SXM GPU donanımlarının hesaplama gücünden maksimum düzeyde yararlanmak üzere sıfırdan tasarlanmış, **1.1 Milyar parametreli (1.1B)**, Multi-Token Prediction (MTP) ve Muon optimizasyonlu yeni nesil Türkçe dil modelidir. 

Bu depo; ham web metinlerinin çift katmanlı ayıklanmasından, muP ile ön-eğitime (Pretraining), LIMA/Magpie tabanlı hizalama (Alignment - SFT/ORPO) süreçlerine ve OpenAI uyumlu yüksek hızlı streaming servis sunumuna kadar **uçtan uca bir LLM yaşam döngüsü** sunar.

---

## 🏗️ Uçtan Uca Sistem Mimarisi

Aşağıdaki şema, Maya-1'in ön-eğitim veri hattından başlayıp, hizalama aşamalarından geçerek canlı yayına alınmasına kadar olan süreci göstermektedir:

```mermaid
graph TD
    %% Pretraining Data Path
    A1["google/mC4 Türkçe Veri Kümesi"] -->|Streaming Mode| B1["MinHash LSH Fuzzy Deduplication"]
    B1 -->|Model2Vec / SemDeDup| C1["Semantik Deduplication"]
    C1 -->|Model2Vec + Logistic Regression| D1["Dengeli Kalite Sınıflandırıcı"]
    D1 -->|BPE 50k Tokenizer| E1["Temiz 100M Token Kümesi"]
    E1 -->|uint32 Tokenization| F1["clean_data_filtered.bin"]
    
    %% Pretraining Model Path
    F1 -->|Memory Map Load| G1["Maya-1 1.1B Base Model"]
    G1 -->|Muon Optimizer + muP Scaling| H1["Pretrained Base Checkpoint"]
    
    %% Alignment Data Path
    A2["Alpaca-TR + Turkish Instructions"] -->|HF Direct Download| B2["103k Combined Chat Corpus"]
    B2 -->|Model2Vec Quality Classifier| C2["Quality-Scored Filtering"]
    C2 -->|Identity Sanitization (Kiro -> Maya)| D2["Sanitized Dialogues"]
    D2 -->|ORPO Preference Pair Generation| E2["chat_data_orpo_100k.jsonl"]
    
    %% Alignment Model Path
    H1 -->|Vocab Expansion (50k -> 50,004)| F2["Expanded Base Model"]
    F2 & E2 -->|orpo_train.py| G2["Maya-1 Chat-Aligned Checkpoint"]
    
    %% Inference Path
    G2 -->|Self-Speculative Decoding| H2["FastAPI Inference Server"]
    H2 -->|SSE Streaming| I2["Chat Completions API (/v1/chat/completions)"]
```

---

## ⚡ Hiper-Performans Özet Grafiği (H100 SXM)

```
================================================================================
Step Time (ms)     | ██████████ 283ms [Sabit & Kararlı]
Throughput (tok/s) | ████████████████████ 7,215 token/sn [Zirve Performans]
GPU Utilization    | █████████████████████████ 100% [Full Load - 670W Power]
Loss Convergence   | 12.47 (Step 0) ===> 2.76 (Step 38900) [Hızlı Yakınsama]
================================================================================
```

---

## ✨ Fark Yaratan Mimariler & Teknolojik Katmanlar

### 1. Türkçe Kelime Parçalanma Çözümü (BPE 50k Tokenizer)
* **Problem:** Standart tokenizer'lar Türkçe gibi sondan eklemeli (agglutinative) dillerde kelime köklerini ve eklerini aşırı bölerek bağlam boyutunu verimsiz kullanır.
* **Mimari Çözüm:** 749 MB boyutundaki ham Türkçe külliyat (`shared/ham_veri.txt`) üzerinde sıfırdan eğitilen 50.000 kelime hazneli Byte-Pair Encoding tokenizer ([turkish_bpe_50k.json](file:///c:/Users/HP/Desktop/Maya-1/shared/turkish_bpe_50k.json)) geliştirildi.
* **Sonuç:** Bağlam uzunluğu (context window) verimliliği artırılarak modelin daha az token ile daha anlamlı Türkçe metin üretmesi sağlandı.

### 2. Çift Katmanlı Veri Temizleme (MinHash LSH & SemDeDup)
* **Mimari Çözüm:** Web kaynaklı kirli `mC4` veri setinin optimizasyonu için iki aşamalı bir tekilleştirme hattı ([deduplicate_dataset.py](file:///c:/Users/HP/Desktop/Maya-1/scripts/deduplicate_dataset.py)) tasarlanmıştır:
  1. **MinHash LSH:** Jaccard benzerliği $\ge 0.80$ olan benzer ve yakın-kopyalanmış metinleri eler.
  2. **Semantic Deduplication (SemDeDup):** `Model2Vec` (Potion-Base-8M) statik metin gömme modeliyle doküman vektörleri oluşturulur ve kosinüs benzerliği $\ge 0.85$ olan semantik kopyaları eler.
* **Sonuç:** Yapılan testlerde, `mC4` Türkçe veri setindeki semantik kopyaların **%72'si** eğitim başlamadan önce başarıyla temizlenmiştir.

### 3. Vektör Tabanlı Sınıflandırıcı (Model2Vec + Logistic Regression)
* **Problem:** Klasik fastText kelime torbası (Bag-of-Words) modelleri, küçük etiketli veri setlerinde (Claude etiketli 200 satır veri) ezberleme (overfitting) problemi yaşar.
* **Mimari Çözüm:** [filter_quality.py](file:///c:/Users/HP/Desktop/Maya-1/python_training/filter_quality.py) güncellenerek **Model2Vec** kelime vektörleri üzerinden metin gömmeleri alan ve **Logistic Regression** ile sınıflandırma yapan yeni nesil Few-Shot süzgeç yapısına geçildi.
* **Sonuç:** Reklam, menü parçaları ve spam gibi düşük kaliteli içeriklerin filtrelenmesindeki genelleme yeteneği maksimuma çıkarıldı.

### 4. Maximal Update Parametrization (muP) Entegrasyonu
* **Problem:** Modeller büyütüldüğünde (örneğin 37M'den 1.1B parametreye), en uygun öğrenme oranı (learning rate) gibi hiperparametreler tamamen değişir ve devasa maliyetli yeniden aramalar gerektirir.
* **Mimari Çözüm:** Microsoft'un `mup` kütüphanesi [model.py](file:///c:/Users/HP/Desktop/Maya-1/python_training/model.py) ve [train.py](file:///c:/Users/HP/Desktop/Maya-1/python_training/train.py) dosyalarına entegre edildi:
  * Ağırlık paylaşımı (tied-weights) yapısına uygun `mup.MuSharedReadout` çıkış katmanı eklendi.
  * Eğitim başlangıcında base (proxy) modelin katman yapısı `mup.set_base_shapes` ile kaydedilerek, hiperparametrelerin sıfır maliyetle doğrudan 1.1B modeline aktarılması sağlandı.
  * AdamW parametre grupları için `mup.MuAdamW` optimize edici yapısı entegre edildi.

### 5. Hizalama Eko-Sistemi (SFT / ORPO) 💬
Sohbet yeteneği kazandırmak amacıyla modele **ORPO (Odds Ratio Preference Optimization)** uygulanır.
* **Vocab Expansion (Kelime Dağarcığı Genişletme):** Ön-eğitimi tamamlanan 50.000 kelimelik model, sohbet formatı için özel token'lar (`<|im_start|>`, `<|im_end|>`, vb.) eklenerek **50.004** kelime dağarcığına genişletilir. Genişletilen yeni embedding vektörleri rastgele ilklendirilmek yerine, ön-eğitim ağırlıklarının *ortalama vektörü (mean embeddings)* ile doldurularak yakınsama hızı 3 kat artırılmıştır.
* **Kimlik Temizliği (Identity Sanitization):** Veri üretiminde kullanılan harici modellerin kalıntı isimleri (örn: `"Kiro"`), veri kalifikasyon aşamasında otomatik tespit edilerek `"Maya"` olarak güncellenir.
* **Yapay Negatif Üretimi (ORPO Pref):** Modelin halüsinasyon görmesini, kelime tekrarlamasını ve yarım bırakmasını önlemek amacıyla; `chosen` (hedef kaliteli cevap) değerlerinden hareketle yerel kurallarla `rejected` (bozuk/reddedilen cevap) çiftleri türetilmiştir.

### 6. OpenAI Uyumlu FastAPI Streaming Sunucusu 🚀
* Model, [inference_server.py](file:///c:/Users/HP/Desktop/Maya-1/python_training/inference_server.py) ile ayağa kaldırılarak OpenAI formatında (`/v1/chat/completions`) hizmet sunar.
* **Server-Sent Events (SSE)** standardında token akışı (streaming) desteklenir.
* **Self-Speculative Decoding** ve `mup` parametre hizalaması sayesinde çıkarım sırasında minimum gecikme (latency) hedeflenmiştir.

---

## 📁 Proje Dizini Yapısı

```
Maya-1/
│
├── python_training/
│   ├── model.py              # MayaModel (MuSharedReadout ve GQA Yapısı)
│   ├── train.py              # muP Entegrasyonu, DDP ve Asenkron Eğitim Döngüsü
│   ├── orpo_train.py         # ORPO Hizalama Eğitimi (Dinamik maskeleme ve dolgu)
│   ├── muon.py               # Muon Optimizer & Newton-Schulz Matematik Motoru
│   ├── filter_quality.py     # Model2Vec + Logistic Regression Kalite Filtresi (Pretraining)
│   ├── db_logger.py          # SQLite asenkron metrik kayıt (AsyncMetricLogger)
│   ├── generate.py           # Self-Speculative Decoding metin üretimi (Chat & Base)
│   └── inference_server.py   # FastAPI yüksek hızlı çıkarım sunucusu (v1/chat/completions)
│
├── scripts/
│   ├── train_turkish_tokenizer.py  # Sıfırdan BPE 50k kelime haznesi eğitimi
│   ├── deduplicate_dataset.py      # MinHash LSH + SemDeDup çift katmanlı temizlik
│   ├── generate_chat_data.py       # Magpie/LIMA sentetik sohbet verisi üretici
│   ├── curate_lima_style.py        # Veri temizleme, filtreleme ve kimlik maskeleme
│   ├── curate_alpaca_quality.py    # Alpaca-TR + Merve/Inst kalifikasyon ve 100K filtreleme
│   ├── repo_to_text.py             # Kod tabanını analiz için tek dosyaya paketleme
│   └── test_mup.py                 # muP entegrasyonu doğrulama scripti
│
└── shared/                   # Kontrol noktaları (checkpoints) ve veri dosyaları
```

---

## ⚡ Kurulum ve Çalıştırma

### 1. Bağımlılıkları Yükleyin
Proje sanal ortamını (`.venv`) aktif hale getirdikten sonra:
```bash
pip install -r requirements.txt
# fastText, Model2Vec, mup ve web sunucu gereksinimleri için:
pip install numpy==1.26.4 fasttext model2vec datasketch mup fastapi uvicorn sse-starlette pandas joblib
```

### 2. Türkçe BPE Tokenizer Eğitimi
```bash
python scripts/train_turkish_tokenizer.py \
    --corpus shared/ham_veri.txt \
    --output shared/turkish_bpe_50k.json \
    --vocab_size 50000
```

### 3. Çift Katmanlı Tekilleştirme (Deduplication)
```bash
python scripts/deduplicate_dataset.py \
    --max_docs 20000 \
    --output_path shared/deduplicated_mc4_tr.jsonl
```

### 4. 100K Kaliteli Chat/Hizalama Verisi Üretimi (ORPO)
Birleşik Türkçe talimat havuzundan yerel kalite puanlayıcısı ile 100 bin kaliteli örnek süzmek için:
```bash
python scripts/curate_alpaca_quality.py \
    --mode filter-dataset \
    --classifier_path shared/quality_classifier_test.bin \
    --output_path shared/chat_data_orpo_100k.jsonl
```

### 5. Ön-Eğitimi (Pretraining) Başlatma
muP ve 50k tokenizer kullanarak ön-eğitimi başlatmak için:
```bash
python python_training/train.py \
    --data_path shared/clean_data_filtered.bin \
    --use_mup \
    --vocab_size 50000 \
    --compile \
    --max_steps 50000
```

### 6. Sohbet Hizalama (ORPO alignment) Eğitimini Başlatma
Elde edilen kaliteli 100K verisetiyle modeli eğitmek için:
```bash
python python_training/orpo_train.py \
    --data_path shared/chat_data_orpo_100k.jsonl \
    --resume_from shared/checkpoints/ckpt_latest.pt \
    --epochs 1 \
    --batch_size 2 \
    --seq_len 512 \
    --use_mup \
    --mup_base_hidden 64
```

### 7. FastAPI Çıkarım Sunucusunu Ayağa Kaldırma
```bash
python python_training/inference_server.py \
    --checkpoint shared/checkpoints/ckpt_latest.pt \
    --host 0.0.0.0 \
    --port 8000
```

Artık `/v1/chat/completions` üzerinden standart OpenAI SDK kütüphanelerini kullanarak Maya-1'e bağlanabilir ve yüksek hızlı Türkçe cevap akışını izleyebilirsiniz!

---

<p align="center">
  🚀 <i>Maya-1: Yerel, Hızlı ve Akıllı Türkçe Dil Teknolojileri</i>
</p>
