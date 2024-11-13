# Laporan Proyek Machine Learning - Yeftha Joshua Ezekiel

## Domain Proyek

Proyek difokuskan pada deteksi tumor otak menggunakan metode deep learning, khususnya transfer learning dengan model pretrained. Deteksi dini tumor otak dapat membantu dalam diagnosis dan perencanaan perawatan medis. Penggunaan model machine learning diharapkan dapat meningkatkan keakuratan diagnosis, mengurangi waktu pemeriksaan, serta memberikan hasil yang konsisten.

**Rubrik/Kriteria Tambahan**:
- Masalah ini penting karena tumor otak yang terdeteksi secara dini dapat meningkatkan peluang kesembuhan dan mengurangi komplikasi. Namun, kesalahan deteksi, seperti false positives, bisa menimbulkan kecemasan pada pasien yang sebenarnya sehat.
- [Efficient Brain Tumor Detection Based on Deep Learning Models](https://iopscience.iop.org/article/10.1088/1742-6596/2128/1/012012/pdf) 

## Business Understanding

### Problem Statements
1. Bagaimana mendeteksi tumor otak dengan tingkat akurasi yang tinggi?
2. Bagaimana meminimalkan false positives untuk menghindari kesalahan diagnosa yang merugikan pasien sehat?

### Goals
1. Menghasilkan model deteksi tumor otak dengan akurasi dan recall yang tinggi.
2. Meminimalisir jumlah false positives untuk mengurangi potensi salah diagnosa.

**Rubrik/Kriteria Tambahan**:
### Solution Statements:
1. Menggunakan dua model transfer learning, yaitu InceptionV3 dan ResNet50, untuk menentukan mana yang memberikan hasil terbaik dalam mendeteksi tumor otak.
2. Melakukan hyperparameter tuning pada model dengan performa terbaik guna memaksimalkan hasil yang diukur berdasarkan metrik evaluasi recall.

## Data Understanding

Dataset yang digunakan dalam proyek ini adalah dataset gambar MRI otak yang diklasifikasikan ke dalam dua kelas: tumor (glioma) dan sehat. Dataset ini diperoleh dari Kaggle - Brain Tumor 
[Brain Tumor Dataset](https://www.kaggle.com/datasets/rm1000/brain-tumor-mri-scans/data), sebuah repositori terbuka yang menyediakan gambar MRI otak untuk deteksi tumor.

**Informasi Dataset**
**Total Data**: Dataset ini terdiri dari total 3621 gambar MRI.

**Distribusi Kelas**:
- MRI Otak Tumor (Glioma): Terdapat 1621 gambar yang mewakili MRI otak dengan glioma.
- MRI Otak Sehat: Terdapat 2000 gambar yang mewakili MRI otak sehat.

**Resolusi Gambar**: Resolusi gambar bervariasi, namun akan disesuaikan (resize) menjadi ukuran 256x256 piksel selama preprocessing.
  
**Format Gambar**: Grayscale

**Variabel dalam Dataset**:
- healthy: Gambar MRI yang menunjukkan otak tanpa tumor.
- glioma: Gambar MRI yang menunjukkan otak dengan tumor jenis glioma.

**Rubrik/Kriteria Tambahan**:
GAMBAR VISUALISASI DATA

## Data Preparation

Beberapa tahapan data preparation yang dilakukan antara lain:
Memisahkan dataset menjadi training dan validation dengan rasio 80:20.
Melakukan augmentasi data pada training set, seperti rotasi, shear, dan zoom, untuk meningkatkan generalisasi model.

**Rubrik/Kriteria Tambahan**:

**Memisahkan Dataset menjadi Training dan Validation Set (80:20)**:
Dataset dibagi menjadi 80% untuk training set dan 20% untuk validation set.
Tujuan: Pembagian ini membantu melatih model dengan data yang cukup banyak untuk pembelajaran, sekaligus menyisihkan sebagian data untuk menguji kinerja model pada data baru (validation set). Validation set digunakan untuk memonitor performa model pada data yang tidak dilihat selama training, sehingga dapat memberikan indikasi seberapa baik model akan bekerja pada data nyata.
Augmentasi Data pada Training Set:

**Dilakukan augmentasi data pada gambar di training set dengan teknik berikut**:
Rotasi: Memutar gambar dalam rentang sudut tertentu, membantu model mengenali objek dengan orientasi yang berbeda.
Shear: Menyimulasikan distorsi dalam gambar, membuat model lebih tangguh terhadap perubahan kecil dalam bentuk atau sudut objek.
Zoom: Memperbesar gambar untuk fokus pada bagian tertentu, membuat model lebih tangguh dalam mengenali objek meski ukurannya bervariasi.
Tujuan: Augmentasi ini bertujuan meningkatkan generalisasi model. Dalam dataset kecil, augmentasi data membantu "menambah" variasi gambar tanpa benar-benar menambah data baru, sehingga model tidak terlalu bergantung pada pola spesifik dalam data training. Hal ini mengurangi risiko overfitting, di mana model belajar terlalu spesifik pada data training dan gagal mengenali pola umum saat diuji dengan data baru.
Alasan Diperlukan Tahapan Data Preparation
Pembagian Dataset (Training dan Validation Set):

Pembagian dataset penting untuk menguji performa model secara obyektif. Dengan validation set, kita dapat memantau apakah model overfitting atau underfitting selama proses training. Jika performa di training set jauh lebih tinggi dari validation set, ini bisa menjadi tanda overfitting.
Pembagian ini juga penting untuk tuning hyperparameter. Model dilatih pada training set, sementara performa di validation set digunakan untuk mengevaluasi berbagai pengaturan model, seperti jumlah epoch atau pembaruan parameter lainnya.
Augmentasi Data:

Augmentasi data membantu meningkatkan variasi data dalam training set, yang berguna terutama saat dataset asli memiliki jumlah data yang terbatas. Teknik augmentasi membantu model untuk mengenali pola yang lebih umum dan membuat model lebih tangguh terhadap variasi pada gambar nyata. Ini penting untuk meningkatkan performa model pada data baru dan tak terduga.


## Modeling
Berikut adalah tahapan yang dilakukan dalam proses pemodelan dengan dua arsitektur yang berbeda, InceptionV3 dan ResNet50, serta penjelasan parameter yang digunakan:

1. Persiapan Pretrained Model
InceptionV3 dan ResNet50 adalah dua arsitektur deep learning yang sudah dilatih sebelumnya pada dataset besar seperti ImageNet. Model-model ini digunakan untuk transfer learning, di mana bagian awal model (layers) akan digunakan untuk ekstraksi fitur, sementara bagian akhir akan disesuaikan (fine-tuning) untuk tugas khusus, dalam hal ini untuk deteksi tumor otak.

InceptionV3:
```
pretrained_model = InceptionV3(input_shape=(256,256,3),
                               include_top=False,
                               weights=None)  # Weights from the local file
pretrained_model.load_weights(local_weights_file)
for layer in pretrained_model.layers:
    layer.trainable = False
```

`input_shape=(256,256,3)`: Ukuran input untuk gambar RGB (256x256 piksel dan 3 channel warna).

`include_top=False`: Ini berarti kita tidak akan menggunakan lapisan fully connected (top) dari InceptionV3, hanya bagian convolutional untuk ekstraksi fitur.

`weights=None`: Model dimuat tanpa bobot yang ada, dan bobot akan dimuat dari file lokal.

ResNet50:

```
pretrained_model = ResNet50(input_shape=(256, 256, 3),
                            include_top=False,
                            weights='imagenet')
for layer in pretrained_model.layers:
    layer.trainable = False
```

`input_shape=(256, 256, 3)`: Sama seperti pada InceptionV3, model diharapkan menerima gambar RGB.

`weights='imagenet'`: Bobot pretrained yang sudah dilatih pada dataset ImageNet digunakan.

**Penambahan Layers untuk Fine-Tuning**

Setelah menggunakan model pretrained, beberapa layer tambahan ditambahkan untuk mengadaptasi model agar dapat bekerja pada dataset baru (MRI otak tumor).

Layer tambahan:

`Lambda(duplicate_channels, input_shape=(256, 256, 1))`: Mengubah gambar grayscale menjadi gambar RGB dengan menduplikasi channel grayscale (menambah 2 channel lagi). Ini memastikan bahwa input gambar memiliki 3 channel (sesuai dengan InceptionV3 dan ResNet50).

`Conv2D dan MaxPooling2D`: Menggunakan layer convolution untuk ekstraksi fitur tambahan setelah layer pretrained dan pooling untuk mereduksi dimensi.

`Dense`: Fully connected layer yang digunakan untuk klasifikasi akhir.

`Flatten`: Meratakan output 2D menjadi 1D untuk masuk ke layer fully connected.

`Softmax`: Fungsi aktivasi pada layer output yang mengklasifikasikan output ke dalam dua kelas: tumor (glioma) dan sehat.

`Regularization`:l2(0.001) pada layer Conv2D untuk mencegah overfitting dengan memberikan penalti pada bobot yang besar.

`Model output`: Output dari model ini adalah dua kelas, menggunakan softmax untuk klasifikasi multi-kelas.

Rubrik/Kriteria Tambahan:

Setiap model dianalisis dari segi kelebihan dan kekurangannya. Misalnya, ResNet50 dikenal baik dalam menangani deep architecture tanpa mengalami masalah vanishing gradient.


 
