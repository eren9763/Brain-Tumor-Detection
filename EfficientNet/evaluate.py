"""
evaluate.py — Baseline (EfficientNetB2) Test ve Değerlendirme Modülü

Bu modül:
1. Figshare test seti (İç Doğrulama) üzerinde performans ölçümü yapar.
2. Kaggle test setini morfolojik olarak kırpar ve (Dış Doğrulama / Domain Shift) testini gerçekleştirir.
3. Sonuçları Classification Report ve Confusion Matrix olarak görselleştirir.
"""

import os
import cv2
import shutil
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# İleride config.py dosyasında tanımlayacağımız sabitler
from config import CLASSES, KAGGLE_SOURCE_TEST, KAGGLE_CROPPED_TEST, IMG_SIZE

def run_evaluation(model, test_gen, dataset_name="Figshare"):
    """
    Verilen jeneratör üzerinden modeli test eder, F1 skorlarını yazdırır ve
    Karmaşıklık Matrisi (Confusion Matrix) ısı haritasını çizer.
    """
    print(f"\n[{dataset_name} Testi] Tahminler alınıyor, lütfen bekleyin...")
    
    Y_pred = model.predict(test_gen)
    y_pred = np.argmax(Y_pred, axis=1)

    print('\n' + '='*50)
    print(f'--- {dataset_name.upper()} TEST RAPORU ---')
    print('='*50)
    print(classification_report(test_gen.classes, y_pred, target_names=CLASSES))

    # Karmaşıklık Matrisi (Confusion Matrix) Hesaplama
    cm = confusion_matrix(test_gen.classes, y_pred)
    plt.figure(figsize=(8, 6))
    
    # Görsel ayırt edicilik: Figshare için Maviler (Blues), Kaggle için Turuncular (Oranges)
    cmap_color = 'Oranges' if dataset_name.lower() == 'kaggle' else 'Blues'
    
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap_color, xticklabels=CLASSES, yticklabels=CLASSES)
    plt.title(f'Confusion Matrix - Baseline B2 ({dataset_name})')
    plt.xlabel('Tahmin Edilen Sınıf')
    plt.ylabel('Gerçek Sınıf')
    plt.show()


def evaluate_on_kaggle(model):
    """
    Kaggle dış veri setini Baseline standartlarında (morfolojik kırpma uygulayarak) 
    işler ve modelin alan kaymasına (domain shift) karşı zayıflığını test eder.
    """
    # Döngüsel içe aktarmayı (circular import) önlemek için burada çağırıyoruz
    from preprocessing import crop_brain_contour
    
    print("\nKaggle test görüntüleri baseline standartlarına (morfolojik kırpma) göre işleniyor...")
    
    # Eski kırpılmış Kaggle test klasörü varsa temizle ve yeniden oluştur
    if os.path.exists(KAGGLE_CROPPED_TEST):
        shutil.rmtree(KAGGLE_CROPPED_TEST)
    for cls in CLASSES:
        os.makedirs(os.path.join(KAGGLE_CROPPED_TEST, cls), exist_ok=True)

    # Kaggle test setindeki görüntüleri tek tek oku, kırp ve yeni klasöre kaydet
    for cls in CLASSES:
        cls_path = os.path.join(KAGGLE_SOURCE_TEST, cls)
        if not os.path.exists(cls_path):
            continue
        
        images = [f for f in os.listdir(cls_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        for img_name in images:
            img_path = os.path.join(cls_path, img_name)
            image = cv2.imread(img_path)
            
            if image is not None:
                cropped = crop_brain_contour(image)
                # Kontur başarıyla bulunduysa kırpılmış halini, bulunamadıysa orijinalini kaydet
                if cropped is not None and cropped.size > 0:
                    cv2.imwrite(os.path.join(KAGGLE_CROPPED_TEST, cls, img_name), cropped)
                else:
                    cv2.imwrite(os.path.join(KAGGLE_CROPPED_TEST, cls, img_name), image)
                    
    print("Morfolojik Kırpma işlemi tamamlandı. Kaggle jeneratörü kuruluyor...")
    
    # 1./255 normalizasyonlu test jeneratörü
    kaggle_datagen = ImageDataGenerator(rescale=1./255)
    kaggle_test_generator = kaggle_datagen.flow_from_directory(
        KAGGLE_CROPPED_TEST,
        target_size=IMG_SIZE,
        batch_size=32,
        class_mode='categorical',
        shuffle=False
    )
    
    # Hazırlanan Kaggle jeneratörünü standart değerlendirme fonksiyonuna gönder
    run_evaluation(model, kaggle_test_generator, dataset_name="Kaggle")
