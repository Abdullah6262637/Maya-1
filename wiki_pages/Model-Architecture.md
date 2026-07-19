# 🏗️ Model Mimarisi ve Matematiksel Temeller

Maya-1, NVIDIA H100 GPU mimarilerine tam uyum sağlayacak şekilde optimize edilmiş, **1.1 Milyar parametreli (1.1B)**, Multi-Token Prediction (MTP) destekli bir Decoder-Only Transformer mimarisidir.

---

## 1. Maya-1 Model Yapı Taşları

Modelin çekirdek mimarisi [model.py](file:///c:/Users/HP/Desktop/Maya-1/python_training/model.py) dosyasında tanımlanmıştır. Modern LLM literatüründeki en kararlı ve performanslı bileşenleri bir araya getirir:

* **Grouped-Query Attention (GQA):** 32 adet Attention Head (Query) için 8 adet KV Head (Key/Value) kullanır. Bu sayede VRAM kullanımı dramatik düzeyde düşerken, KV Cache verimliliği zirveye çıkar.
* **SwiGLU Aktivasyon Fonksiyonu:** Standart ReLU veya GELU yerine Swish kapılı doğrusal birimler (SwiGLU) kullanılarak katmanlar arası bilgi akışının doğrusal olmayan temsil gücü artırılmıştır.
* **Rotary Position Embeddings (RoPE):** Sabit pozisyon gömmeleri yerine, token pozisyon bilgisini dikkat (attention) hesaplamaları sırasında rotasyonel matrislerle enjekte eder. Uzun bağlam boyutlarında yüksek kararlılık sağlar.
* **Residual Init Scaling:** Katman sayısı arttıkça oluşan gradyan patlamalarını önlemek için, artık bağlantı (residual connection) ağırlıkları eğitim başında $1 / \sqrt{2 \cdot \text{layers}}$ katsayısı ile ölçeklenir.

---

## 2. muP (Maximal Update Parameterization) Entegrasyonu

Modeller büyütüldüğünde (örneğin 37M test modelinden 1.1B ana modele geçildiğinde), en uygun öğrenme oranı (learning rate) gibi hiperparametreler tamamen değişir. Maya-1, Microsoft'un `mup` kütüphanesini entegre ederek bu sorunu kökten çözer.

* **MuSharedReadout:** Ağırlık paylaşımı (tied weight) mantığına tam uyumlu çıkış katmanı kullanılır. Giriş gömme (embedding) katmanı ile çıkış doğrusal projeksiyon katmanı parametreleri paylaşılır.
* **Proxy (Base) ve Target Hizalaması:**
  * Eğitim başında küçük bir model (`mup_base_hidden=64`) proxy olarak tanımlanır ve katman şekilleri `mup.set_base_shapes` ile kaydedilir.
  * Büyük hedef model (1.1B) bu şekiller doğrultusunda muP kurallarına göre ağırlık ilklendirmesi (initialization) ve öğrenme oranı ölçeklemesi yapar.
  * Bu sayede küçük modelde bulunan en iyi hiperparametreler (learning rate, weight decay), büyük modele **sıfır maliyetle doğrudan aktarılır**.

---

## 3. Multi-Token Prediction (MTP) Kafaları

Standart dil modelleri her adımda sadece bir sonraki token'ı tahmin eder ($x_{t+1}$). Maya-1 ise **Multi-Token Prediction (MTP)** mimarisi sayesinde her adımda eşzamanlı olarak **3 token birden tahmin eder** ($x_{t+1}, x_{t+2}, x_{t+3}$).

```
        ┌───► MTP Head 1 ───► Token t+1
        │
Base ───┼───► MTP Head 2 ───► Token t+2
        │
        └───► MTP Head 3 ───► Token t+3
```

### Avantajları:
1. **Çıkarım (Inference) Hızı:** Self-Speculative Decoding ile birleştirildiğinde çıkarım sırasında saniyede üretilen token sayısını 3 kata kadar artırır.
2. **Semantik Tutarlılık:** Model, cümleyi kurarken sadece bir sonraki kelimeye değil, 3 kelime sonrasına da odaklandığı için dil bilgisi ve cümlenin genel gidişatındaki mantık hataları minimuma iner.
