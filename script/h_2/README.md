# CIFAR-10 Görüntü Sınıflandırma Projesi

## Introduction

Bu projede CIFAR-10 veri seti üzerinde evrişimli sinir ağları kullanılarak görüntü sınıflandırma çalışması yapılmıştır. Çalışmanın amacı temel bir CNN modeli, iyileştirilmiş bir CNN modeli, literatürde yaygın kullanılan bir CNN mimarisi ve hibrit bir CNN + makine öğrenmesi yaklaşımını karşılaştırmaktır.

## Method

Çalışmada CIFAR-10 veri seti kullanılmıştır. Veri setindeki görüntüler RGB formatında ve 32x32 boyutundadır. Görüntüler PyTorch dönüşümleri ile tensöre çevrilmiş ve normalize edilmiştir.

Birinci model LeNet-5 benzeri temel bir CNN mimarisi olarak oluşturulmuştur. Bu modelde Conv2d, ReLU, MaxPool2d, Flatten ve Linear katmanları kullanılmıştır.

İkinci model, birinci modelin temel yapısı korunarak oluşturulmuştur. Bu modelde ek olarak BatchNorm2d ve Dropout katmanları kullanılmıştır. Batch normalization eğitimi daha kararlı hale getirmek, dropout ise overfitting riskini azaltmak için eklenmiştir.

Üçüncü model olarak torchvision kütüphanesinden ResNet18 mimarisi kullanılmıştır. CIFAR-10 görüntüleri küçük boyutlu olduğu için ilk convolution katmanı 3x3 kernel ve stride 1 olacak şekilde düzenlenmiş, maxpool katmanı kaldırılmıştır. Son tam bağlantılı katman 10 sınıfa uygun hale getirilmiştir.

Dördüncü model hibrit model olarak tasarlanmıştır. Eğitilmiş ResNet18 modelinin sınıflandırıcı katmanı çıkarılarak feature extractor olarak kullanılmıştır. Bu yapı ile train ve test veri setleri için özellik vektörleri çıkarılmıştır. Çıkarılan özellikler ve etiketler .npy dosyaları olarak kaydedilmiştir. Daha sonra bu özellikler RandomForestClassifier modeli ile sınıflandırılmıştır.

Kullanılan loss function CrossEntropyLoss, optimizer ise Adam olarak seçilmiştir. Adam optimizer adaptif öğrenme oranı kullandığı için eğitim sürecinde hızlı ve kararlı sonuçlar verebilmektedir.

## Results

| Model | Test Accuracy (%) |
| --- | --- |
| LeNet5 | 63.59 |
| LeNet5 + BatchNorm + Dropout | 66.12 |
| ResNet18 | 83.84 |
| ResNet18 Features + RandomForest | 84.97 |

Feature dosyalarının boyutları aşağıdaki gibidir:

| Veri | Boyut |
|---|---|
| X_train_features | (50000, 512) |
| y_train_features | (50000,) |
| X_test_features | (10000, 512) |
| y_test_features | (10000,) |

Eğitim kaybı grafiği outputs/loss_comparison.png dosyasına kaydedilmiştir. En başarılı model için karmaşıklık matrisi outputs/confusion_matrix.png dosyasına kaydedilmiştir.

## Discussion

LeNet-5 modeli temel CNN yapısını göstermesi açısından sade ve anlaşılırdır. Ancak CIFAR-10 gibi RGB ve daha karmaşık görüntülerden oluşan bir veri setinde temsil gücü sınırlı kalabilir.

Batch normalization ve dropout eklenen ikinci model, temel modele göre daha kararlı öğrenme ve daha iyi genelleme sağlayabilir. Batch normalization ara aktivasyonları düzenlerken dropout bazı nöronları eğitim sırasında devre dışı bırakarak modelin ezberleme eğilimini azaltır.

ResNet18 modeli daha derin bir mimari olduğu için özellik çıkarma kapasitesi daha yüksektir. Residual bağlantılar sayesinde derin ağlarda gradyan akışı daha sağlıklı gerçekleşir. Bu nedenle ResNet18 modelinin temel CNN modellerine göre daha yüksek başarı vermesi beklenir.

Hibrit modelde CNN tarafı özellik çıkarıcı, RandomForest ise sınıflandırıcı olarak kullanılmıştır. Bu yaklaşım, derin öğrenme ile klasik makine öğrenmesi yöntemlerinin birlikte kullanılmasına örnektir. Hibrit modelin başarısı, çıkarılan özelliklerin kalitesine bağlıdır.

## References

- PyTorch Documentation
- Torchvision Models
- CIFAR-10 Dataset
- scikit-learn Documentation
