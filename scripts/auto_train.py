import os
import sys
import subprocess
import time

def get_latest_checkpoint(checkpoint_dir):
    if not os.path.exists(checkpoint_dir):
        return None, 0
    files = [os.path.join(checkpoint_dir, f) for f in os.listdir(checkpoint_dir) if f.startswith("ckpt_step_") and f.endswith(".pt")]
    if not files:
        return None, 0
    
    def get_step_num(fpath):
        try:
            return int(os.path.basename(fpath).replace("ckpt_step_", "").replace(".pt", ""))
        except ValueError:
            return -1
            
    latest_file = max(files, key=get_step_num)
    step_num = get_step_num(latest_file)
    return latest_file, step_num

def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    checkpoint_dir = os.path.join(base_dir, "shared", "checkpoints")
    lua_config = os.path.join(base_dir, "python_training", "config.lua")
    data_path = os.path.join(base_dir, "shared", "mock_data.bin")
    
    # Configuration
    step_block_size = 1000 # Number of pre-training steps per iteration before SFT
    
    print("=" * 60)
    print("Maya-1 Otomatik Sürekli Eğitim Boru Hattı Başlatılıyor...")
    print("Bu betik sırasıyla:")
    print(f"1. Modeli {step_block_size} adım ön eğitime (Pre-train) sokar.")
    print("2. Sonrasında otomatik olarak sohbet (SFT) ince ayarını yapar.")
    print("3. Sohbet panelini günceller ve kaldığı yerden döngüye devam eder.")
    print("Döngüyü durdurmak için istediğiniz zaman Ctrl+C tuşlarına basabilirsiniz.")
    print("=" * 60)
    
    python_exe = sys.executable
    
    iteration = 1
    try:
        while True:
            # 1. Detect current checkpoint and step
            latest_ckpt, current_step = get_latest_checkpoint(checkpoint_dir)
            target_step = current_step + step_block_size
            
            print(f"\n[DÖNGÜ #{iteration}] Mevcut Adım: {current_step} -> Hedef Adım: {target_step}")
            
            # 2. Build pre-training command
            cmd_train = [
                python_exe,
                os.path.join(base_dir, "python_training", "train.py"),
                "--data_path", data_path,
                "--max_steps", str(target_step),
                "--checkpoint_interval", "500",
                "--lua_config", lua_config
            ]
            
            if latest_ckpt:
                cmd_train += ["--resume_from", latest_ckpt]
                print(f"Ön eğitim son checkpointten devam ediyor: {os.path.basename(latest_ckpt)}")
            else:
                print("Başlangıç checkpointi bulunamadı. Eğitim sıfırdan başlıyor...")
                
            # Run pre-training
            print(f"Çalıştırılan komut: {' '.join(cmd_train)}")
            p_train = subprocess.Popen(cmd_train, cwd=base_dir)
            p_train.wait()
            
            if p_train.returncode != 0:
                print(f"[HATA] Ön eğitim başarısız oldu! Çıkış kodu: {p_train.returncode}")
                time.sleep(5)
                continue
                
            # Verify new checkpoint exists
            _, new_step = get_latest_checkpoint(checkpoint_dir)
            if new_step <= current_step:
                print("[UYARI] Ön eğitim sonrasında yeni bir checkpoint oluşturulamadı veya adım ilerlemedi.")
            
            # 3. Build and run SFT command
            print("\nÖn eğitim bloğu tamamlandı. Sohbet hizalama (SFT) başlatılıyor...")
            cmd_sft = [
                python_exe,
                os.path.join(base_dir, "python_training", "sft.py")
            ]
            
            p_sft = subprocess.Popen(cmd_sft, cwd=base_dir)
            p_sft.wait()
            
            if p_sft.returncode != 0:
                print(f"[HATA] SFT hizalama başarısız oldu! Çıkış kodu: {p_sft.returncode}")
            else:
                print("[BAŞARI] SFT hizalaması tamamlandı ve sft_checkpoint.pt güncellendi!")
                print("Canlı Sohbet panelinde yeni modeli seçip hemen test edebilirsiniz.")
                
            iteration += 1
            print("-" * 50)
            time.sleep(3)
            
    except KeyboardInterrupt:
        print("\n[BİLGİ] Eğitim döngüsü kullanıcı tarafından durduruldu.")
        sys.exit(0)

if __name__ == "__main__":
    main()
