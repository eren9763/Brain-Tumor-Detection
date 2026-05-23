"""
data_pipeline.py — Baseline (EfficientNetB2) Çevrimdışı Veri Artırımı ve Jeneratör Modülü

Bu modül:
1. Sınıf dengesizliğini çözmek için azınlık sınıflarına (Meningioma vb.) fiziksel 
   olarak veri artırımı (offline augmentation) uygular ve diske kaydeder.
2. Orijinal makaledeki hedef resim sayılarına (Glioma: 2795, Meningioma: 1290, Pituitary: 1829) ulaşır.
3. Modelin eğitimi ve testi için resimlere yalnızca (1./255) normalizasyon uygulayan 
   ImageDataGenerator nesnelerini kurar.
"""

import os
import random
import shutil
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

# İleride config.py dosyasında tanımlayacağımız sabitler
from config import (
    CLASSES, 
    TRAIN_DIR, 
    VAL_DIR, 
    TEST_DIR, 
    AUG_TRAIN_DIR, 
    TARGET_COUNTS, 
    IMG_SIZE, 
    BATCH_SIZE, 
    SEED
)

def _augment_and_save(source_dir, target_dir, target_count, cls):
    """
    Belirli bir sınıftaki görüntüleri kopyalar ve eksik sayıyı (target_count'a kadar)
    rastgele rotasyon ve kaydırmalarla (augmentation) üreterek tamamlar.
    """
    current_files = [f for f in os.listdir(os.path.join(source_dir, cls))]
    current_count = len(current_files)
    needed_count = target_count - current_count

    # Makale standartlarına uygun offline augmentation parametreleri
    aug_gen = ImageDataGenerator(
        rotation_range=15, 
        width_shift_range=0.1,
        height_shift_range=0.2, 
        horizontal_flip=True, 
        fill_mode='nearest'
    )

    # 1. Adım: Orijinal kırpılmış dosyaları olduğu gibi hedef (augmented) klasörüne kopyala
    for f in current_files:
        shutil.copy(os.path.join(source_dir, cls, f), os.path.join(target_dir, cls, f))

    # 2. Adım: Eksik sayı varsa (needed_count > 0), sentetik veri üreterek tamamla
    if needed_count > 0:
        generated = 0
        while generated < needed_count:
            # Rastgele bir orijinal resim seç
            img_name = random.choice(current_files)
            img_path = os.path.join(source_dir, cls, img_name)
            
            # Görüntüyü yükle ve numpy dizisine çevir (Augmentation jeneratörü için)
            img = load_img(img_path)
            x = img_to_array(img)
            x = x.reshape((1,) + x.shape)

            # Jeneratörden tek bir varyant üret ve diske jpeg formatında kaydet
            for batch in aug_gen.flow(x, batch_size=1, 
                                      save_to_dir=os.path.join(target_dir, cls),
                                      save_prefix='aug', save_format='jpeg'):
                generated += 1
                break # Sonsuz döngüyü engellemek için flow sonrası break


def apply_offline_augmentation():
    """
    Ana (train) klasöründeki verileri okuyarak, TARGET_COUNTS hedeflerine ulaşana kadar
    çevrimdışı (offline) veri artırımı uygular ve sonuçları yeni klasöre yazar.
    """
    # Eski artırılmış veri klasörü varsa silip temiz bir sayfa aç
    if os.path.exists(AUG_TRAIN_DIR):
        shutil.rmtree(AUG_TRAIN_DIR)
        
    for cls in CLASSES:
        os.makedirs(os.path.join(AUG_TRAIN_DIR, cls), exist_ok=True)

    print("\n[Veri Hattı] Sınıf dengelemesi için çevrimdışı (offline) veri artırımı başlatılıyor...")
    for cls in CLASSES:
        target = TARGET_COUNTS[cls]
        print(f" - {cls} sınıfı hedeflenen sayıya ({target}) eşitleniyor...")
        _augment_and_save(TRAIN_DIR, AUG_TRAIN_DIR, target, cls)
        
    print("[Veri Hattı] Çevrimdışı artırım başarıyla tamamlandı.\n")


def build_generators():
    """
    Eğitim (Train), Doğrulama (Val) ve Test setleri için görüntü jeneratörlerini kurar.
    Diskte fiziksel veri artırımı yapıldığı için burada YALNIZCA rescale (1./255) işlemi yapılır.
    """
    # Rescale işlemi (0-255 piksel değerlerini 0-1 aralığına sıkıştırma)
    datagen = ImageDataGenerator(rescale=1./255)

    print("[Veri Hattı] Veri Jeneratörleri (Rescale 1./255) oluşturuluyor...")
    
    # DİKKAT: Eğitim jeneratörü orijinal train dizininden değil, yeni oluşturulan 
    # artırılmış (AUG_TRAIN_DIR) dizininden verileri okur.
    train_generator = datagen.flow_from_directory(
        AUG_TRAIN_DIR, 
        target_size=IMG_SIZE, 
        batch_size=BATCH_SIZE, 
        class_mode='categorical',
        seed=SEED
    )
    
    val_generator = datagen.flow_from_directory(
        VAL_DIR, 
        target_size=IMG_SIZE, 
        batch_size=BATCH_SIZE, 
        class_mode='categorical',
        seed=SEED
    )
    
    # Test jeneratöründe shuffle=False olmalıdır ki metrikler (Classification Report vs.) karışmasın
    test_generator = datagen.flow_from_directory(
        TEST_DIR, 
        target_size=IMG_SIZE, 
        batch_size=BATCH_SIZE, 
        class_mode='categorical', 
        shuffle=False,
        seed=SEED
    )
    
    return train_generator, val_generator, test_generator
