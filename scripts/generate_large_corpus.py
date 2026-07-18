# Python script to generate a massive, diverse synthetic dataset containing thousands of lines of TR and EN text.
# Covers history, science, coding, literature, daily chat, sports, celebrity news, and current events.

import os
import random

def generate_large_corpus(output_path, num_lines=12000):
    print(f"Generating large corpus to: {output_path}...")
    
    # 1. Base sentence pools (Turkish)
    tr_subjects = [
        "Yapay zeka", "Makine öğrenimi", "Derin öğrenme", "Büyük dil modelleri", 
        "Dünya", "Jüpiter", "Samanyolu galaksisi", "Türkiye", "İstanbul", "Ankara",
        "Programlama dilleri", "Rust dili", "Python", "Go programlama dili", "TypeScript",
        "SQLite", "İnternet", "Bilgisayar ağları", "Kuantum fiziği", "Yapay sinir ağları",
        "Tarih boyunca insanlık", "Sanayi devrimi", "Yazılım mühendisliği", "Veri bilimi",
        # Sports, Celebrity, Agenda/News additions
        "Galatasaray", "Fenerbahçe", "Beşiktaş", "Milli takım sporcuları", "Şampiyonlar Ligi",
        "Ünlü sinema oyuncuları", "Popüler ses sanatçıları", "Vizyona giren yeni filmler",
        "Ekonomik gelişmeler", "Teknoloji devleri", "Merkez Bankası yetkilileri", "Son dakika haberleri"
    ]
    tr_verbs = [
        "bilişsel işlevleri yerine getirmek için geliştirilen önemli teknolojilerdir.",
        "büyük veri kümelerini işleyerek kalıpları tespit etmemizi sağlar.",
        "günümüzde teknolojinin ve yazılım dünyasının temelini oluşturmaktadır.",
        "bilgiye erişimi kolaylaştırarak insan hayatını kökten değiştirmiştir.",
        "gelecekte hayatımızın her alanında daha etkin rol oynayacaktır.",
        "güvenli, hızlı ve ölçeklenebilir sistemler geliştirmek için kullanılır.",
        "analiz edilerek stratejik kararlar alma süreçlerinde kullanılır.",
        "fizik ve kimya yasalarının sınırlarını zorlayan araştırmalara konu olmuştur.",
        "kapsamlı kütüphaneleri ve basit sözdizimi sayesinde hızla yaygınlaşmaktadır.",
        "jeopolitik konumu ve zengin kültürel geçmişiyle her zaman dikkat çekmiştir.",
        "öğrencilerin ve araştırmacıların en çok odaklandığı çalışma alanlarındandır.",
        # Sports, Celebrity, Agenda/News verbs
        "sezon sonu transfer hazırlıklarını büyük bir titizlikle hızlandırdı.",
        "taraftarları ve medyayı heyecanlandıran açıklamalarda bulundu.",
        "sosyal medyada milyonlarca beğeni toplayan bir paylaşıma imza attı.",
        "yeni projeleriyle magazin dünyasında adından sıkça söz ettirmeye başladı.",
        "piyasalardaki dalgalanmaları doğrudan etkileyerek ekonomi gündemini belirledi.",
        "kamuoyunda geniş yankı uyandıran yasal düzenlemeler için çalışma başlattı.",
        "yeni sezonda şampiyonluk kupasını kazanmak için hazırlıklarını sürdürüyor."
    ]
    tr_conversations = [
        "Merhaba! Ben Maya-1, size yardımcı olmak için buradayım.",
        "Bugün size hangi konu hakkında bilgi vermemi istersiniz?",
        "Yapay zeka sistemleri hakkında ne öğrenmek istersiniz?",
        "Hava durumu bugün oldukça güzel görünüyor.",
        "Bilgisayarların çalışma prensipleri karmaşık ama büyüleyicidir.",
        "Programlama öğrenmek, problem çözme yeteneğinizi geliştirir.",
        "Dünyanın en derin yeri Mariana Çukuru olarak bilinir.",
        "Galaksimizde milyarlarca yıldız ve gezegen sistemi bulunur.",
        "Maya-1 modeli, sıfırdan tasarlanmış çok dilli bir yapay zekadır.",
        "Sağlıklı bir yaşam için düzenli egzersiz yapmak çok önemlidir.",
        "Kitap okumak, kelime haznenizi ve genel kültürünüzü zenginleştirir.",
        # Sports, Celebrity, Agenda/News conversations
        "Dün akşam oynanan derbi maçı tribünleri tamamen doldurdu ve nefes kesti.",
        "Ünlü şarkıcı yeni albüm lansmanında kameraların karşısına geçerek açıklamalarda bulundu.",
        "Gündemdeki yeni ekonomik düzenlemeler ve vergi paketleri vatandaşlar tarafından takip ediliyor.",
        "Milli sporcumuz olimpiyatlarda altın madalya kazanarak hepimizi gururlandırdı.",
        "Bu hafta sonu yapılacak olan açık hava konseri biletleri saatler içinde kapış kapış tükendi.",
        "Son dakika gelişmesine göre yeni bir teknoloji girişimi büyük bir yatırım aldığını duyurdu."
    ]

    # 2. Base sentence pools (English)
    en_subjects = [
        "Artificial intelligence", "Machine learning", "Deep learning", "Large language models",
        "The Earth", "Jupiter", "The Milky Way", "France", "Paris", "London",
        "Programming languages", "Rust language", "Python", "Go programming language", "TypeScript",
        "SQLite", "The Internet", "Computer networks", "Quantum physics", "Neural networks",
        "Human history", "The Industrial Revolution", "Software engineering", "Data science",
        # Sports, Celebrity, Agenda/News additions
        "The Premier League", "The national football team", "Real Madrid CF", "Olympic gold medalists",
        "Famous Hollywood celebrities", "Pop music stars", "Award-winning drama series",
        "Global financial markets", "The federal reserve board", "Innovative tech giants", "Breaking news stories"
    ]
    en_verbs = [
        "are revolutionizing the way we solve complex technological problems.",
        "allow us to process vast amounts of unstructured data efficiently.",
        "form the absolute baseline of modern cloud software services.",
        "have fundamentally changed human access to global information databases.",
        "will play a much more prominent role in our daily lives in the future.",
        "are built to implement safe, fast, and highly concurrent architectures.",
        "provide critical insights for data-driven strategic planning.",
        "remain at the forefront of modern scientific research and discovery.",
        "have gained massive popularity due to their rich ecosystem of packages.",
        "hold a very strategic importance on the geopolitical world map.",
        "are fascinating topics attracting students and researchers globally.",
        # Sports, Celebrity, Agenda/News verbs
        "have accelerated their transfer efforts to sign elite players for the next season.",
        "have dominated social media trends and pop culture headlines this week.",
        "are closely analyzing the latest consumer price index and economic data.",
        "won a prestigious award at the international film festival yesterday.",
        "are drawing massive crowds of passionate fans to their world tour performances.",
        "will play a highly decisive role in the upcoming national championship match.",
        "released a statement that quickly generated massive headlines worldwide."
    ]
    en_conversations = [
        "Hello! I am Maya-1, a multilingual AI assistant designed to help you.",
        "How can I assist you with your project tasks today?",
        "Feel free to ask me anything about deep learning models.",
        "The weather seems to be exceptionally pleasant today.",
        "Understanding CPU scheduling algorithms is complex but rewarding.",
        "Learning to write clean code is an essential skill for developers.",
        "The deepest known point on Earth is the Mariana Trench.",
        "Our solar system consists of eight distinct major planets orbiting the Sun.",
        "The Maya-1 model was trained fully from scratch on multi-lingual datasets.",
        "Regular physical exercise is key to maintaining a healthy lifestyle.",
        "Reading historical novels can broaden your perspectives about the world.",
        # Sports, Celebrity, Agenda/News conversations
        "The championship final match last night ended in a historic and thrilling penalty shootout.",
        "A well-known celebrity fashion designer launched a sustainable clothing line in Paris.",
        "Economic analysts expect a major shift in interest rates by the end of this financial quarter.",
        "Our national athlete set a new world record at the track and field championship event.",
        "A major technology company officially unveiled its highly anticipated smart device today.",
        "Global news networks are reporting live from the international diplomatic summit."
    ]

    with open(output_path, "w", encoding="utf-8") as f:
        # Write some initial structure
        f.write("Maya-1 AI System Large Corpus File\n")
        f.write("==================================\n\n")
        
        for i in range(num_lines):
            # Mix structured sentences, conversations, and technical lines
            rand_val = random.random()
            if rand_val < 0.25:
                # Turkish structured
                line = f"{random.choice(tr_subjects)} {random.choice(tr_verbs)}"
            elif rand_val < 0.50:
                # English structured
                line = f"{random.choice(en_subjects)} {random.choice(en_verbs)}"
            elif rand_val < 0.75:
                # Turkish chat
                line = random.choice(tr_conversations)
            else:
                # English chat
                line = random.choice(en_conversations)
                
            f.write(line + "\n")
            
    print(f"[SUCCESS] Generated {num_lines} sentences in: {output_path}")

if __name__ == "__main__":
    generate_large_corpus("shared/ham_veri.txt", 3000000)
