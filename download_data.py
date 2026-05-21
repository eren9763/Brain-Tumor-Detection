"""
download_data.py — Model ve veri setlerini Google Drive'dan indirir.

Tüm kaynaklar tek dosya (ZIP veya .keras) olarak indirilir.
ZIP'ler otomatik çıkartılır. Dosyalar zaten mevcutsa tekrar indirilmez.
"""
import os
import sys
import zipfile
import subprocess

from config import (
    MODEL_GDRIVE_ID, MODEL_FILENAME, SAVE_PATH,
    MRI_ZIP_GDRIVE_ID, MENDELEY_ZIP_GDRIVE_ID,
    BASE_PATH, KAGGLE_TRAIN_DIR, MENDELEY_DIR,
)


def _ensure_gdown():
    try:
        import gdown
    except ImportError:
        print("gdown bulunamadı, kuruluyor…")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "gdown"])
        import gdown
    return gdown


def _download_file(gdrive_id: str, dest_path: str, label: str):
    if os.path.exists(dest_path):
        print(f"[✓] {label} zaten mevcut, atlanıyor.")
        return
    gdown = _ensure_gdown()
    os.makedirs(os.path.dirname(dest_path) or ".", exist_ok=True)
    print(f"[↓] {label} indiriliyor…")
    gdown.download(
        id=gdrive_id,
        output=dest_path,
        quiet=False,
        use_cookies=False,
        verify=False,
    )
    if not os.path.exists(dest_path):
        raise FileNotFoundError(
            f"{label} indirilemedi.\n"
            "Drive paylaşım iznini kontrol edin."
        )
    print(f"[✓] {label} indirildi: {dest_path}")


def _extract_zip(zip_path: str, extract_to: str, label: str):
    """ZIP'i belirtilen dizine çıkartır, ardından ZIP'i siler."""
    print(f"[↓] {label} çıkartılıyor → {extract_to}")
    os.makedirs(extract_to, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(extract_to)
    os.remove(zip_path)
    print(f"[✓] {label} çıkartıldı.")


def download_model():
    """Pretrained .keras modelini indirir."""
    _download_file(MODEL_GDRIVE_ID, SAVE_PATH, "Model (.keras)")


def download_mri_dataset():
    """Kaggle MRI Dataset ZIP'ini indirir ve çıkartır."""
    if os.path.isdir(KAGGLE_TRAIN_DIR):
        print("[✓] MRI Dataset zaten mevcut, atlanıyor.")
        return
    zip_path = os.path.join(BASE_PATH, "mri_dataset.zip")
    _download_file(MRI_ZIP_GDRIVE_ID, zip_path, "MRI Dataset (ZIP)")
    _extract_zip(zip_path, BASE_PATH, "MRI Dataset")

    # ZIP çıktısında ekstra iç klasör olabilir, kontrol et
    _fix_nested_folder(BASE_PATH, "MRI dataset")


def download_mendeley():
    """Mendeley Dataset ZIP'ini indirir ve çıkartır."""
    if os.path.isdir(MENDELEY_DIR):
        print("[✓] Mendeley Dataset zaten mevcut, atlanıyor.")
        return
    zip_path = os.path.join(BASE_PATH, "mendeley.zip")
    _download_file(MENDELEY_ZIP_GDRIVE_ID, zip_path, "Mendeley Dataset (ZIP)")
    _extract_zip(zip_path, BASE_PATH, "Mendeley Dataset")

    _fix_nested_folder(BASE_PATH, "Mendeley")


def _fix_nested_folder(base: str, expected_name: str):
    """
    ZIP çıkarma bazen iç içe klasör oluşturur:
      data/MRI dataset/MRI dataset/Training  →  istemiyoruz
      data/MRI dataset/Training              →  istiyoruz

    Bu fonksiyon bunu otomatik düzeltir.
    """
    import shutil
    target = os.path.join(base, expected_name)
    nested = os.path.join(target, expected_name)
    if os.path.isdir(nested):
        print(f"[~] İç içe klasör tespit edildi, düzeltiliyor: {nested}")
        tmp = target + "_tmp"
        shutil.move(nested, tmp)
        shutil.rmtree(target)
        shutil.move(tmp, target)
        print(f"[✓] Düzeltildi: {target}")


def download_all():
    """Model + her iki veri setini indir. Zaten varsa atlar."""
    print("\n=== Varlık Kontrolü & İndirme ===")
    os.makedirs(BASE_PATH, exist_ok=True)
    download_model()
    download_mri_dataset()
    download_mendeley()
    print("=== Tüm dosyalar hazır ===\n")


if __name__ == "__main__":
    download_all()
