"""
download_data.py — Baseline (EfficientNetB2) Modeli ve Veri Setlerini İndirme Modülü

Bu modül:
1. Figshare MRI veri setini (Eğitim ve İç Doğrulama için)
2. Kaggle MRI veri setini (Dış Doğrulama / Domain Shift testi için)
3. Eğitilmiş Baseline modelini (.keras) 
Google Drive üzerinden otomatik olarak indirir ve zipten çıkartır.
Dosyalar zaten mevcutsa indirme işlemini atlayarak zaman kazandırır.
"""
import os
import sys
import zipfile
import subprocess

# config.py dosyasında tanımlayacağımız sabitleri içe aktarıyoruz
from config import (
    MODEL_GDRIVE_ID, 
    FIGSHARE_ZIP_ID, 
    KAGGLE_ZIP_ID,
    BASE_PATH, 
    MODEL_SAVE_PATH, 
    FINAL_MODEL_NAME,
    FIGSHARE_SOURCE_DIR,
    KAGGLE_SOURCE_TEST
)

def _ensure_gdown():
    """Sistemde gdown kütüphanesi yoksa otomatik kurar."""
    try:
        import gdown
    except ImportError:
        print("gdown bulunamadı, kuruluyor...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "gdown"])
        import gdown
    return gdown

def _download_file(gdrive_id, dest_path, label):
    """Google Drive ID'si verilen dosyayı indirir."""
    if os.path.exists(dest_path):
        print(f"[✓] {label} zaten mevcut, atlanıyor.")
        return
    
    gdown = _ensure_gdown()
    os.makedirs(os.path.dirname(dest_path) or ".", exist_ok=True)
    print(f"[↓] {label} indiriliyor...")
    
    # Drive bağlantısından indirme işlemi
    gdown.download(
        id=gdrive_id, 
        output=dest_path, 
        quiet=False, 
        use_cookies=False, 
        verify=False
    )
    
    if not os.path.exists(dest_path):
        raise FileNotFoundError(f"{label} indirilemedi. Drive dosyasının 'Herkese Açık (Public)' olduğundan emin olun.")
    print(f"[✓] {label} başarıyla indirildi: {dest_path}")

def _extract_zip(zip_path, extract_to, label):
    """ZIP dosyasını belirtilen hedefe çıkartır ve ardından ZIP'i siler."""
    print(f"[↓] {label} çıkartılıyor -> {extract_to}")
    os.makedirs(extract_to, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(extract_to)
    
    # Yer kaplamaması için zip dosyasını sil
    os.remove(zip_path)
    print(f"[✓] {label} başarıyla çıkartıldı.")

def download_model():
    """Eğitilmiş Baseline B2 modelini (Hybrid_EfficientNetB2_Baseline_v4_Final.keras) indirir."""
    full_model_path = os.path.join(MODEL_SAVE_PATH, FINAL_MODEL_NAME)
    if os.path.exists(full_model_path):
        print("[✓] Baseline Model zaten mevcut, atlanıyor.")
        return
    _download_file(MODEL_GDRIVE_ID, full_model_path, "Baseline Model (.keras)")

def download_figshare_dataset():
    """Orijinal makalenin (Baseline) eğitimde kullandığı Figshare veri setini indirir."""
    if os.path.isdir(FIGSHARE_SOURCE_DIR):
        print("[✓] Figshare Veri Seti zaten mevcut, atlanıyor.")
        return
    
    zip_path = os.path.join(BASE_PATH, "figshare.zip")
    _download_file(FIGSHARE_ZIP_ID, zip_path, "Figshare Dataset (ZIP)")
    _extract_zip(zip_path, BASE_PATH, "Figshare Dataset")

def download_kaggle_dataset():
    """Domain Shift (Alan Kayması) testi için Kaggle veri setini indirir."""
    if os.path.isdir(KAGGLE_SOURCE_TEST):
        print("[✓] Kaggle Test Veri Seti zaten mevcut, atlanıyor.")
        return
    
    zip_path = os.path.join(BASE_PATH, "kaggle_test.zip")
    _download_file(KAGGLE_ZIP_ID, zip_path, "Kaggle Dataset (ZIP)")
    _extract_zip(zip_path, BASE_PATH, "Kaggle Dataset")

def download_all():
    """main.py tarafından çağrılan, tüm indirme süreçlerini yöneten ana fonksiyon."""
    print("\n=== Varlık Kontrolü & İndirme İşlemleri ===")
    os.makedirs(BASE_PATH, exist_ok=True)
    
    download_model()
    download_figshare_dataset()
    download_kaggle_dataset()
    
    print("=== Tüm dosyalar kullanıma hazır ===\n")

if __name__ == '__main__':
    # Modül doğrudan çalıştırılırsa tüm indirmeleri başlatır
    download_all()
