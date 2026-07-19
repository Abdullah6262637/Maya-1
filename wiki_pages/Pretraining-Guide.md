# ⚡ Ön-Eğitim (Pretraining) Kılavuzu

Maya-1 modelinin ön-eğitimi, yüksek parametreli dil modellerinin en kararlı ve hızlı şekilde yakınsamasını sağlamak üzere tasarlanmış modern bir eğitim döngüsüdür.

---

## 1. Muon Optimizer ve Newton-Schulz Matematik Motoru

Maya-1, standart AdamW veya SGD optimize edicilerinin yanı sıra, ağırlık güncellemelerini çok daha hızlı yakınsatan **Muon** optimize edicisini kullanır.

* **Matris Ortogonalizasyonu:** Muon, ağırlık gradyanlarını doğrudan güncellemek yerine, güncellemeleri ortogonal (dik) matrisler haline getirir.
* **Newton-Schulz İterasyonu:** Ortogonalizasyon işleminin CUDA üzerinde son derece hızlı çalışması için quintic (5. derece) Newton-Schulz katsayıları ($a=3.4445$, $b=-4.7750$, $c=2.0315$) literatüre uygun olarak [muon.py](file:///c:/Users/HP/Desktop/Maya-1/python_training/muon.py) içinde kodlanmıştır.
* **Katman Ayrımı:** Modeldeki tüm gizli katman ağırlıkları (linear layers) Muon ile optimize edilirken, embedding ve bias parametreleri klasik AdamW grubuna yönlendirilir.

---

## 2. Dağıtık Eğitim Mimarisi (DDP)

[train.py](file:///c:/Users/HP/Desktop/Maya-1/python_training/train.py) dosyası, çoklu GPU donanımları üzerinde paralel eğitim yapabilmek için **Distributed Data Parallel (DDP)** altyapısına sahiptir.

* **Asenkron Veri Yükleyici:** Eğitim adımı sürerken bir sonraki batch verinin CPU'dan GPU belleğine aktarılması için asenkron kopyalama ve memory-mapping (`np.memmap`) yöntemleri kullanılır.
* **Dinamik Gradyan Akümülasyonu (Gradient Accumulation):** Fiziksel batch boyutu küçük tutulsa bile, gradyanlar birden fazla adımda biriktirilerek sanal olarak çok büyük batch boyutlarında kararlı eğitim yapılması sağlanır.

---

## 3. Ön-Eğitimi Çalıştırma Adımları

Eğitimi yerel makinede veya uzak sunucuda (Vast.ai, RunPod vb.) başlatmak için şu adımlar uygulanır:

### A. Veriyi Hazırlama (Tokenization)
Öncelikle filtrelenmiş temiz metin verileri uint32 tipinde token'lara çevrilerek binary dosyaya yazılır:
```bash
python python_training/filter_quality.py \
    --mode filter-dataset \
    --tokenizer_path shared/mock_tokenizer.json \
    --output_path shared/clean_data.bin
```

### B. Eğitimi Başlatma
muP ölçeklemesi ve CUDA derlemesi (`torch.compile`) aktif edilerek ön-eğitim başlatılır:
```bash
python python_training/train.py \
    --data_path shared/clean_data.bin \
    --use_mup \
    --vocab_size 50000 \
    --compile \
    --max_steps 50000 \
    --checkpoint_dir shared/checkpoints
```

### C. Eğitime Kaldığı Yerden Devam Etme (Resume Training)
Herhangi bir kesinti durumunda eğitim en son kaydedilen kontrol noktasından (checkpoint) kaldığı yerden devam ettirilebilir:
```bash
python python_training/train.py \
    --data_path shared/clean_data.bin \
    --use_mup \
    --vocab_size 50000 \
    --resume_from shared/checkpoints/ckpt_latest.pt
```
