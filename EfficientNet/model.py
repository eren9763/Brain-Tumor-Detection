"""
model.py — Baseline (EfficientNetB2) Hibrit Mimari Modülü

Bu modül:
1. ImageNet ağırlıklarıyla önceden eğitilmiş EfficientNetB2 tabanını yükler.
2. Referans makalenin metodolojisine uygun olarak tabanı DONDURMAZ (trainable = True), 
   tek aşamalı ince ayar (fine-tuning) için tüm ağı eğitime açık bırakır.
3. Orijinal ImageNet başlığını çöpe atarak, yerine makalede belirtilen 
   Global Average Pooling, %20 Dropout ve 3 sınıflı Softmax Dense katmanlarını (Custom Head) ekler.
"""

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB2

# İleride config.py dosyasında tanımlayacağımız sabitler
from config import IMG_SIZE, CLASSES

def build_hybrid_efficientnet(input_shape=None, num_classes=None):
    """
    Hibrit EfficientNetB2 mimarisini kurar ve derlenmeye hazır model nesnesini döndürür.
    """
    # Config dosyasından değerleri çekip (240, 240, 3) formatına getiriyoruz
    if input_shape is None:
        input_shape = IMG_SIZE + (3,)
    if num_classes is None:
        num_classes = len(CLASSES)

    print(f"\n[Mimari] EfficientNetB2 tabanı {input_shape} boyutuyla yükleniyor...")
    
    # Önceden eğitilmiş taban modeli yükle (Üst katmanlar hariç)
    base_model = EfficientNetB2(
        weights='imagenet', 
        include_top=False, 
        input_shape=input_shape
    )
    
    # DİKKAT: Baseline makalesine göre transfer öğrenme tek aşamalıdır.
    # Bu nedenle taban model kilitlenmez, en başından itibaren eğitime açıktır.
    base_model.trainable = True 

    # Sınıflandırma Başlığı (Custom Head) Eklemesi
    x = base_model.output
    x = layers.GlobalAveragePooling2D(name='global_average_pooling')(x)
    
    # Makalede aşırı öğrenmeyi (overfitting) önlemek için belirtilen %20'lik dropout oranı
    x = layers.Dropout(0.2, name='dropout_20')(x)
    
    predictions = layers.Dense(num_classes, activation='softmax', name='classifier_output')(x)

    # Taban girdi ile yeni çıktıyı birleştirerek nihai modeli oluştur
    model = models.Model(inputs=base_model.input, outputs=predictions, name='Hybrid_EfficientNetB2_Baseline')
    
    return model
