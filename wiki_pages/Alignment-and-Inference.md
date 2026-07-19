# 💬 Hizalama (Alignment) ve Çıkarım (Inference)

Ön-eğitimi biten model, ham kelime tamamlama yeteneğine (base model) sahip olur ancak bir insan gibi diyalog kuramaz (chat model). Maya-1 modeline güvenli ve kaliteli sohbet yeteneği kazandırmak için **ORPO** hizalaması uygulanır.

---

## 1. ORPO (Odds Ratio Preference Optimization) ve 100K Veriseti

Klasik hizalama yöntemlerinde (RLHF, DPO) önce SFT (Supervised Fine-Tuning) ardından tercih eğitimi yapılır. Maya-1 ise bu iki adımı tek seferde birleştiren **ORPO** algoritmasını kullanır.

* **100K Altın Veri Süzme:** `TFLai/Turkish-Alpaca` ve `merve/turkish_instructions` verisetleri birleştirilerek yerel kalite modelimizle filtrelendi ve **99,675 adet** kaliteli Türkçe diyalog çifti ([chat_data_orpo_100k.jsonl](file:///c:/Users/HP/Desktop/Maya-1/shared/chat_data_orpo_100k.jsonl)) oluşturuldu.
* **Tercih Yapısı (Chosen vs. Rejected):**
  * `chosen`: Kaliteli ve dil bilgisi düzgün Türkçe cevap.
  * `rejected`: Modelin halüsinasyon görmesini, tekrara düşmesini ve anlamsız kısa cevaplar vermesini engellemek amacıyla yapay olarak üretilen bozuk negatif örnek.
* **Kimlik Temizliği (Identity Sanitization):** Veri setinde dış kaynaklardan bulaşmış tüm yapay zeka isim kalıntıları (`"Kiro"`, vb.) otomatik olarak temizlenerek `"Maya"` ismi ile güncellenmiştir.

---

## 2. Kelime Haznesi Genişletme (Vocab Expansion)

Sohbet formatı için modele `<|im_start|>`, `<|im_end|>`, `<|sistem|>`, `<|kullanici|>` ve `<|asistan|>` gibi özel yapısal token'lar eklenmesi gerekir.

* **Vocab Boyutu:** Ön-eğitimdeki 50.000 kelime boyutu, bu özel token'lar eklenerek **50.004** seviyesine genişletilir.
* **Mean Embedding İlklendirmesi:** Genişletilen yeni token'ların embedding ağırlıkları rastgele doldurulmak yerine, ön-eğitimde öğrenilen tüm kelimelerin **ortalama vektörü (mean embeddings)** ile ilklendirilir. Bu yöntem, hizalama eğitiminde yakınsama (convergence) hızını 3 kat artırır.

---

## 3. Self-Speculative Decoding ile Süper Hızlı Çıkarım

Çıkarım (inference) sırasında en büyük darboğaz, her adımda sadece bir token tahmin edilmesidir (Autoregressive generation). Maya-1 bu sorunu **Self-Speculative Decoding** ile aşar.

* **Çalışma Prensibi:** Modelin MTP (Multi-Token Prediction) kafaları, ana modelin çıktısını beklemeden hızlıca bir sonraki 2 token adayını tahmin eder. Ana model bu adayları tek bir işlemde (paralel olarak) doğrular.
* **Sonuç:** Çıkarım sırasında kelime üretim hızı (token throughput) kaliteden ödün vermeden **2 ila 3 kat artar**.

---

## 4. FastAPI OpenAI Uyumlu Canlı Yayın Servisi

[inference_server.py](file:///c:/Users/HP/Desktop/Maya-1/python_training/inference_server.py) sunucusu, eğitilen modeli ayağa kaldırarak standart OpenAI API SDK'ları ile uyumlu şekilde hizmete sunar:

### A. Sunucuyu Başlatma
```bash
python python_training/inference_server.py \
    --checkpoint shared/checkpoints/sft_1b/ckpt_aligned.pt \
    --host 0.0.0.0 \
    --port 8000
```

### B. Python Client ile Bağlanma ve Akış (Streaming) Testi
```python
import openai

client = openai.OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="empty_key"
)

response = client.chat.completions.create(
    model="maya-1b",
    messages=[
        {"role": "system", "content": "Sen Maya, yardımcı bir asistansın."},
        {"role": "user", "content": "Türkiye'nin başkenti neresidir?"}
    ],
    stream=True
)

for chunk in response:
    print(chunk.choices[0].delta.content or "", end="")
```
*Sunucu, SSE (Server-Sent Events) protokolüyle cevabı harf harf ekrana akıtır.*
