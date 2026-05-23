"""
train.py — Baseline (EfficientNetB2) Eğitim Modülü

Bu modül:
1. model.py dosyasından Hybrid EfficientNetB2 mimarisini çağırır.
2. Adam optimizasyon algoritmasını (lr=0.001) ve kayıp fonksiyonunu tanımlar.
3. ReduceLROnPlateau ile dinamik öğrenme hızı düşürme stratejisini uygular.
4. 50 epoch'luk tek aşamalı eğitimi yürütür ve en yüksek başarıyı gösteren modeli kaydeder.
"""

import os
import tensorflow as tf
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint

# İleride config.py dosyasında tanımlayacağımız sabitler
from config import EPOCHS, MODEL_SAVE_PATH, FINAL_MODEL_NAME

# İleride model.py dosyasında tanımlayacağımız mimariyi kuran fonksiyon
from model import build_hybrid_efficientnet

def run_training(train_generator, val_generator):
    """
    Modeli kurar, derler ve veri jeneratörleri üzerinden eğitir.
    Eğitim bitiminde en iyi performansı gösteren (val_accuracy) modeli döndürür.
    """
    print("\n[Eğitim] Hybrid EfficientNetB2 Modeli Kuruluyor...")
    model = build_hybrid_efficientnet()

    print("[Eğitim] Model Derleniyor (Optimizer: Adam, lr=0.001)...")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Modelin kaydedileceği dizinin (örn: /content/data/models) var olduğundan emin ol
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
    best_model_path = os.path.join(MODEL_SAVE_PATH, FINAL_MODEL_NAME)

    print("[Eğitim] Callbacks (Geri Çağrılar) Ayarlanıyor...")
    
    # 1. Makaledeki dinamik öğrenme hızı düşürücü strateji
    lr_reduce = ReduceLROnPlateau(
        monitor='val_loss', 
        factor=0.3, 
        patience=5, 
        min_lr=1e-7, 
        verbose=1
    )

    # 2. Yalnızca en yüksek validation_accuracy sonucunu veren modeli kaydet
    checkpoint = ModelCheckpoint(
        filepath=best_model_path, 
        monitor='val_accuracy', 
        save_best_only=True, 
        mode='max', 
        verbose=1
    )

    print(f"\n[Eğitim] {EPOCHS} Epoch'luk Eğitim Süreci Başlıyor...")
    # DİKKAT: Baseline kodlamasında verileri offline augmentation ile diskte 
    # fiziksel olarak eşitlediğimiz için 'class_weight' parametresine ihtiyaç yoktur.
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=val_generator,
        callbacks=[lr_reduce, checkpoint],
        verbose=1
    )

    print(f"\n[Eğitim] Başarıyla tamamlandı! En iyi model kalıcı olarak kaydedildi: {best_model_path}")

    # Test modülüne (evaluate.py) geçmeden önce, en başarılı epoch'un ağırlıklarını belleğe geri yükle
    print("[Eğitim] Test aşaması için en iyi ağırlıklar modele yükleniyor...")
    model.load_weights(best_model_path)

    return model
