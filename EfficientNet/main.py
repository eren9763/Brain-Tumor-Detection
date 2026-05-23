"""
main.py — Baseline (EfficientNetB2) Modeli için Eğitim ve Değerlendirme Boru Hattı

Kullanım Seçenekleri:
    python main.py                   # İndir, ön işleme yap, değerlendir (Varsayılan Model)
    python main.py --train           # İndir, ön işleme yap, sıfırdan eğit ve değerlendir
    python main.py --skip-download   # İndirmeyi atla (Dosyalar zaten varsa)
    python main.py --skip-prep       # Morfolojik kırpma ve offline augmentation'ı atla (Uzun sürer, yapıldıysa atlanabilir)
    python main.py --test-kaggle     # Domain Shift'i ispatlamak için modeli Kaggle dış veri setinde test et
"""

import os
import argparse
import tensorflow as tf

# Modüler dosyalardan içe aktarımlar
from config import FINAL_MODEL_PATH
from download_data import download_all
from preprocessing import prepare_splits_and_crop
from data_pipeline import apply_offline_augmentation, build_generators
from train import run_training
from evaluate import run_evaluation, evaluate_on_kaggle

# GPU optimizasyonu ve log kirliliğini önleme
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def parse_args():
    parser = argparse.ArgumentParser(description='Baseline (EfficientNetB2) - Brain Tumor MRI Pipeline')
    
    parser.add_argument('--train', action='store_true', 
                        help='Modeli önceden eğitilmiş ağırlıklarla başlatmak yerine sıfırdan eğit.')
    parser.add_argument('--model-path', type=str, default=FINAL_MODEL_PATH, 
                        help=f'Kullanılacak modelin yolu (Varsayılan: {FINAL_MODEL_PATH})')
    parser.add_argument('--skip-download', action='store_true', 
                        help='Google Drive üzerinden veri seti ve model indirme adımlarını atla.')
    parser.add_argument('--skip-prep', action='store_true', 
                        help='Morfolojik kırpma ve çevrimdışı (offline) veri artırma adımlarını atla.')
    parser.add_argument('--test-kaggle', action='store_true', 
                        help='Eğitilen modeli Kaggle dış veri setinde (Domain Shift) test et.')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    print("\n" + "="*50)
    print("--- BASELINE (EFFICIENTNET-B2) PIPELINE BAŞLATILIYOR ---")
    print("="*50)

    # 1. ADIM: İndirme İşlemleri
    if not args.skip_download:
        print("\n[ADIM 1] Veri Seti ve Pretrained Model İndiriliyor...")
        download_all()
    else:
        print("\n[ADIM 1] İndirme adımı atlandı (--skip-download).")

    # 2. ADIM: Morfolojik Kırpma, Veri Bölme ve Çevrimdışı Artırım
    if not args.skip_prep:
        print("\n[ADIM 2] Referans Makale Standartlarında Ön İşleme Başlıyor...")
        # Kafatasını silme ve %70-%30 Train/Test bölme
        prepare_splits_and_crop()
        
        print("\n[ADIM 3] Çevrimdışı (Offline) Veri Artırımı (Sınıf Dengeleme)...")
        # Makaledeki hedef sayılara göre (Glioma: 2795, Meningioma: 1290, vb.) veriyi dengeler
        apply_offline_augmentation()
    else:
        print("\n[ADIM 2 & 3] Ön işleme ve çevrimdışı artırım atlandı (--skip-prep).")

    # 4. ADIM: Jeneratörlerin Kurulması (Yalnızca Rescale: 1./255)
    print("\n[ADIM 4] Veri Jeneratörleri Hazırlanıyor...")
    train_gen, val_gen, test_gen = build_generators()

    # 5. ADIM: Eğitim veya Model Yükleme
    if args.train:
        print("\n[ADIM 5] Eğitim (Training) Başlıyor...")
        model = run_training(train_gen, val_gen)
    else:
        print(f"\n[ADIM 5] Hazır Model Yükleniyor: {args.model_path}")
        if not os.path.exists(args.model_path):
            raise FileNotFoundError(f"Model bulunamadı: {args.model_path}. Lütfen --train argümanını kullanın veya indirme adımını atlamayın.")
        model = tf.keras.models.load_model(args.model_path)

    # 6. ADIM: İç Doğrulama (Figshare Testi)
    print("\n[ADIM 6] Figshare (Unseen Data) Testi Başlıyor...")
    run_evaluation(model, test_gen, dataset_name="Figshare")

    # 7. ADIM: Dış Doğrulama (Kaggle Domain Shift Testi) - İsteğe Bağlı
    if args.test_kaggle:
        print("\n[ADIM 7] Kaggle (Cross-Dataset) Domain Shift Testi Başlıyor...")
        # Bu fonksiyon Kaggle verisini otomatik olarak Baseline morfolojisiyle kırpıp test eder
        evaluate_on_kaggle(model)

    print("\n" + "="*50)
    print("--- PIPELINE BAŞARIYLA TAMAMLANDI ---")
    print("="*50 + "\n")

if __name__ == '__main__':
    main()
