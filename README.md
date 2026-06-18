# 🎭 Object Detection with Mask R-CNN

Implementasi **object detection** dan **instance segmentation** menggunakan model **Mask R-CNN** pretrained dengan OpenCV DNN module. Proyek ini mampu mendeteksi objek dari gambar statis maupun kamera secara real-time.

---

## 📋 Deskripsi

**Mask R-CNN** (2017, Girshick et al.) merupakan perluasan dari Faster R-CNN yang menambahkan kemampuan *instance segmentation* di atas object detection. Dua kontribusi utamanya:

| Komponen | Keterangan |
|---|---|
| **ROI Align** | Menggantikan ROI Pooling untuk akurasi ekstraksi fitur yang lebih baik |
| **Mask Branch** | Cabang tambahan yang menghasilkan binary mask per objek terdeteksi |

Model yang digunakan dilatih pada dataset **MS-COCO** dengan **Inception V2** sebagai backbone, mampu mendeteksi **80 kelas objek**.

---

## 📁 Struktur Proyek

```
object-detection_mask-r-cnn/
│
├── object_detection.py         # Deteksi objek dari file gambar
├── object_detection_camera.py  # Deteksi objek secara real-time via kamera
├── output_detection.jpg        # Contoh hasil output deteksi
│
├── images/
│   ├── road.jpg
│   ├── road_1.jpg
│   └── road_3.jpg
│
└── models/
    ├── frozen_inference_graph.pb                  # Model weights TensorFlow
    ├── mask_rcnn_inception_v2_coco_2018_01_28.pbtxt  # Graph definition
    ├── mscoco_labels.names                        # Daftar 80 nama kelas
    └── colors.txt                                 # Warna untuk visualisasi kelas
```

---

## ⚙️ Requirements

- Python 3.8+
- OpenCV (`opencv-python`)
- NumPy

Install dependencies:

```bash
pip install opencv-python numpy
```

---

## 🚀 Cara Penggunaan

### 1. Deteksi dari Gambar Statis

Menjalankan inferensi pada gambar `images/road.jpg` dan menyimpan hasilnya ke `output_detection.jpg`.

```bash
python object_detection.py
```

**Output:**
- Jendela OpenCV menampilkan hasil deteksi
- File `output_detection.jpg` tersimpan di direktori utama
- Setiap objek terdeteksi ditampilkan dengan bounding box, label kelas, confidence score, dan segmentation mask

### 2. Deteksi Real-Time via Kamera

Menjalankan inferensi secara live menggunakan kamera (webcam).

```bash
python object_detection_camera.py
```

> **Catatan untuk macOS:** Pastikan Terminal / VS Code / iTerm memiliki izin akses kamera di **System Settings > Privacy & Security > Camera**. Jalankan ulang aplikasi setelah memberikan izin.

**Kontrol:**
- Tekan `q` untuk keluar dari mode kamera

---

## 🔧 Konfigurasi

Parameter deteksi dapat disesuaikan di bagian atas masing-masing script:

```python
CONF_THRESHOLD = 0.5   # Ambang batas confidence (0.0 – 1.0)
MASK_THRESHOLD = 0.3   # Ambang batas untuk binary mask (0.0 – 1.0)
```

| Parameter | Default | Keterangan |
|---|---|---|
| `CONF_THRESHOLD` | `0.5` | Objek dengan confidence di bawah nilai ini diabaikan |
| `MASK_THRESHOLD` | `0.3` | Piksel dengan nilai mask di bawah nilai ini dianggap background |

---

## 🧠 Arsitektur Pipeline

```
Input Image / Video Frame
        ↓
Feature Extraction (Inception V2 Backbone)
        ↓
Region Proposal Network (RPN)
        ↓
ROI Align
        ↓
  ┌─────────────────┬──────────────────┐
  │ Classification  │  Bounding Box    │
  │    (80 kelas)   │   Regression     │
  └─────────────────┴──────────────────┘
             ↓
        Mask Branch
             ↓
   Instance Segmentation
             ↓
  Output: BBox + Label + Score + Mask
```

---

## 📊 Cara Kerja

### `draw_box()`
Fungsi visualisasi yang:
1. Menggambar **bounding box** di sekitar objek terdeteksi
2. Menampilkan **label kelas** dan **confidence score**
3. Me-resize dan men-threshold **mask** yang diprediksi
4. Meng-overlay **segmentation mask** dengan warna acak per instance
5. Menggambar **kontur** pada area yang tersegmentasi

### `post_process()`
Fungsi post-processing yang:
1. Mengekstrak deteksi dari output jaringan
2. Memfilter deteksi berdasarkan `CONF_THRESHOLD`
3. Mengonversi koordinat ternormalisasi ke koordinat piksel
4. Memilih mask yang sesuai dengan kelas yang diprediksi
5. Memanggil `draw_box()` untuk visualisasi

---

## 🗂️ Model

Model yang digunakan adalah **Mask R-CNN Inception V2** yang dilatih pada **MS-COCO** (80 kelas):

| Detail | Info |
|---|---|
| Framework | TensorFlow |
| Backbone | Inception V2 |
| Dataset | MS-COCO |
| Jumlah Kelas | 80 |
| File Weights | `frozen_inference_graph.pb` (~64 MB) |

> **⚠️ Catatan:** File `frozen_inference_graph.pb` berukuran besar (~64 MB). Jika tidak tersedia di repositori, unduh dari:
> ```
> http://download.tensorflow.org/models/object_detection/mask_rcnn_inception_v2_coco_2018_01_28.tar.gz
> ```
> Kemudian ekstrak file `.pb`-nya ke dalam folder `models/`.

---

## 📚 Referensi

- [OpenCV DNN Module Documentation](https://docs.opencv.org/4.x/d2/d58/tutorial_table_of_content_dnn.html)
- [MS-COCO Dataset](https://cocodataset.org/)

---

## 👤 Author

**Rivan** — Tugas UAS Mata Kuliah Pengolahan dan Analisis Citra Digital (PACD)  
Universitas Padjadjaran · Semester 6
