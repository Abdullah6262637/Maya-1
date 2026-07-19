# 🚀 Maya-1 Wiki Belgeleri

Maya-1 projesinin teknik mimarisini, veri işleme hatlarını, ön-eğitim ve hizalama süreçlerini açıklayan resmi Wiki sayfalarına hoş geldiniz.

Bu belgeler, 1.1 Milyar parametreli, Türkçe diline özel tasarlanmış Multi-Token Prediction (MTP) modelinin tüm detaylarını içermektedir.

---

## 🗺️ Wiki İçerik Haritası

Aşağıdaki bağlantıları kullanarak projenin ilgili teknik bölümlerine hızlıca ulaşabilirsiniz:

### 1. 🏗️ [Model Mimarisi ve Matematiksel Temeller](Model-Architecture)
* **MayaModel Sınıfı:** GQA, SwiGLU ve RoPE entegrasyonu.
* **muP (Maximal Update Parameterization):** Hiperparametrelerin sıfır maliyetle ölçeklenmesi.
* **MTP (Multi-Token Prediction) Kafaları:** Eşzamanlı 3 token tahmini ile çıkarım hızı.

### 2. 🧹 [Veri İşleme ve Kalite Hattı](Dataset-Pipeline)
* **BPE Türkçe Tokenizer:** Sıfırdan 50.000 kelimelik hazne eğitimi.
* **Çift Katmanlı Tekilleştirme:** MinHash LSH ve SemDeDup (Semantik Tekilleştirme).
* **Model2Vec Few-Shot Kalite Süzgeci:** Yüksek kaliteli veri süzme mimarisi.

### 3. ⚡ [Ön-Eğitim (Pretraining) Kılavuzu](Pretraining-Guide)
* **Muon Optimizer:** Newton-Schulz matris ortogonalizasyonu.
* **DDP (Distributed Data Parallel):** Çoklu GPU ve asenkron veri yükleyici.
* **muP Parametre Hizalaması:** Proxy ve Target model kurulumu.

### 4. 💬 [Hizalama (Alignment) ve Çıkarım (Inference)](Alignment-and-Inference)
* **ORPO (Odds Ratio Preference Optimization):** Tercih tabanlı hizalama.
* **Kelime Haznesi Genişletme:** Mean embedding ilklendirmesi.
* **FastAPI Sunucusu:** OpenAI uyumlu streaming (SSE) çıkarım servisi.
* **Self-Speculative Decoding:** Model içi yüksek hızlı çıkarım tekniği.
