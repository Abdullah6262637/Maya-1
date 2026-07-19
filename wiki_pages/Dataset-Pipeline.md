# 🧹 Veri İşleme ve Kalite Hattı

Yapay zeka modellerinin başarısı doğrudan eğitime giren verinin kalitesine bağlıdır. Maya-1, kirli web verilerinden (mC4 Türkçe) arındırılmış, son derece rafine bir Türkçe külliyat oluşturmak amacıyla çok aşamalı bir veri işleme hattı (data pipeline) kullanır.

---

## 1. Türkçe BPE Tokenizer Eğitimi

Standart İngilizce odaklı tokenizer'lar Türkçe kelimeleri aşırı parçalar (örneğin "muvaffakiyetsizleştiremeyeceklerimizdendiniz" kelimesi 10'dan fazla token'a bölünebilir). Bu durum hem modelin bağlam penceresini (context window) verimsiz kullanır hem de Türkçe dil bilgisi öğrenimini zorlaştırır.

* **Çözüm:** 749 MB boyutundaki ham Türkçe külliyat (`shared/ham_veri.txt`) üzerinde sıfırdan eğitilen **50.000 kelime hazneli** Byte-Pair Encoding (BPE) tokenizer ([turkish_bpe_50k.json](file:///c:/Users/HP/Desktop/Maya-1/shared/turkish_bpe_50k.json)) geliştirildi.
* **Sonuç:** Türkçe kelimelerin morfolojik bütünlüğü büyük ölçüde korunarak modelin bağlam penceresi verimliliği maksimuma çıkarıldı.

---

## 2. Çift Katmanlı Tekilleştirme (Deduplication)

Web kaynaklı metinler (mC4) binlerce kopya haber, birbirinin aynı reklam yazıları ve yinelenen cümleler içerir. Bu kopyaları temizlemek için [deduplicate_dataset.py](file:///c:/Users/HP/Desktop/Maya-1/scripts/deduplicate_dataset.py) script'i ile çift katmanlı bir filtreleme hattı tasarlanmıştır:

### A. Birinci Katman: MinHash LSH (Fuzzy Deduplication)
* Dokümanlar kelime bazında N-gram'lara bölünerek MinHash imzaları oluşturulur.
* **Locality-Sensitive Hashing (LSH)** algoritması ile Jaccard benzerliği $\ge 0.80$ olan, yani ufak değişikliklerle birbirinin kopyası olan tüm dokümanlar hızlıca elenir.

### B. İkinci Katman: Semantic Deduplication (SemDeDup)
* Kalan dokümanlar **Model2Vec (Potion-Base-8M)** hafif metin temsil modeliyle yoğun vektörlere (dense embeddings) dönüştürülür.
* Doküman vektörleri arasındaki kosinüs benzerliği (Cosine Similarity) hesaplanır ve benzerliği $\ge 0.85$ olan semantik kopyaların (farklı kelimelerle aynı konuyu anlatan kopyalar) **%72'si** elenir.

---

## 3. Vektör Tabanlı Few-Shot Kalite Süzgeci

Klasik kelime torbası (Bag-of-Words) ve fastText modelleri, küçük etiketli veri setlerinde (örn: Claude API ile etiketlenmiş 200 adet veri) kolayca ezberleme (overfitting) yapar ve karmaşık cümle yapılarını ayırt edemez.

* **Mimari:** [filter_quality.py](file:///c:/Users/HP/Desktop/Maya-1/python_training/filter_quality.py) kütüphanesinde **Model2Vec** ve **Logistic Regression** tabanlı modern bir few-shot kalite süzgeci kurulmuştur.
* **Nasıl Çalışır?**
  1. Metinler Model2Vec ile 256 boyutlu vektörlere dönüştürülür.
  2. Bu vektörler üzerinden eğitilen Logistic Regression sınıflandırıcısı, metnin reklam, forum spam'ı veya kopuk kod parçası olup olmadığını yüksek doğrulukla tespit eder.
  3. Sadece kalite puanı yüksek olan dokümanlar ön-tokenizasyon işleminden geçirilerek ikili (binary uint32) shard dosyasına yazılır.
