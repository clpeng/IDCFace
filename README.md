# IDCFace: Identity Consistent Face Anonymization for Secure Recognition
Ruiying Lu , Shuang Wan , Zimin Miao , Nannan Wang, Chunlei Peng
Xidian University

This project implements an architecture for identity-consistent face anonymization and secure recognition. The system protects facial privacy while ensuring controllable recognition and tracking in specific secure scenarios by embedding hidden identity codes.

## 🌟 Core Pipeline
This work consists of three core processing stages:
1. **Anonymization**: Utilizing generative models to conceal the original identity in the image.
2. **Super-Resolution (SR)**: Restoring and enhancing the visual quality of the anonymized face images.
3. **Fingerprint Embedding & Detection**: Embedding specific identity codes into the anonymized images, supporting subsequent extraction and secure verification.



## 🚀 Quick Start

### 0. Environment Setup
```bash
pip install -r requirements.txt
```

### 1. Anonymization Model Training
Start model training on a specified dataset (e.g., CASIA-WebFace):
```bash
CUDA_VISIBLE_DEVICES=0 python3 main_train.py \
    config/main.yaml \
    --data_root /data/dataset/CASIA-WebFace
```

### 2. Face Anonymization (Inference)

```bash
CUDA_VISIBLE_DEVICES=0 python3 random_encrypt.py config/best.yaml \
    --ckpt_name checkpoint_13_iter8193.pth.tar \
    --dataset CELEBA \
    --data_dir_type 1 \
    --data_root /data/Downloads/backup/paper_code/visual_result/or \
    --save_dir /data/Downloads/backup/paper_code/visual_result/pwd
```

### 3. Post-Anonymization Super-Resolution

```bash
python face_enhancement_v2.py \
    --model GPEN-BFR-512 \
    --channel_multiplier 2 \
    --narrow 1 \
    --use_sr \
    --use_cuda \
    --indir /data/outputs/paper/img_align_celeba_128 \
    --outdir /data/outputs/paper_2/img_align_celeba_128
```

### 4. Identity Code Embedding

```bash
python my_embed_fingerprints_v2.py \
    --encoder_path model/gauss_2/stegastamp_100_16032022_21:49:43_encoder.pth \
    --data_dir /data/outputs/paper/img_align_celeba_128_encrypt_super_2_3_12 \
    --image_resolution 128 \
    --output_dir /data/outputs/paper/img_align_celeba_256_encrypt_fingerprint_4_15 \
    --batch_size 8
```

### 5. Identity Code Detection & Recognition
Decode and perform secure identity verification on the images embedded with codes:
```bash
python my_detect_fingerprints.py
```