# CSC4005 BTL – Garbage Classification

Phân loại rác thải thành 6 loại: **cardboard, glass, metal, paper, plastic, trash**

## Thành viên nhóm
- Phùng Hữu Tài – KHMT 17-01
- Nguyễn Việt Chung - KHMT 17-01
- Đặng Quốc An- KHMT 17-01
- Nguyễn Đoàn Ngọc Linh- KHMT 17-01

## Dataset
TrashNet / Garbage Classification Dataset từ Kaggle  
Link: https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification

## Cấu trúc
```
├── src/
│   ├── dataset.py          # Data loading
│   ├── models.py           # CNN Baseline + Transfer Learning
│   ├── train.py            # Training + W&B
│   ├── utils.py            # Plots, metrics
│   ├── export_onnx.py      # Export ONNX
│   ├── infer_onnx.py       # Inference ONNX
│   ├── gradcam.py          # Grad-CAM visualization
│   └── error_analysis.py   # Phân tích lỗi
├── configs/                # Cấu hình thí nghiệm
├── outputs/                # Kết quả train
├── reports/                # Báo cáo phân tích
├── app.py                  # Demo Streamlit
└── requirements.txt
```

## Cài đặt
```bash
conda activate csc4005-dl
pip install -r requirements.txt
```

## Chạy thí nghiệm

### 1. CNN Baseline
```bash
python -m src.train --data_dir "D:\path\to\dataset" --model_name cnn_baseline --augment --use_wandb --run_name cnn_baseline
```

### 2. ResNet18 Fine-tune
```bash
python -m src.train --data_dir "D:\path\to\dataset" --model_name resnet18 --train_mode finetune --augment --use_wandb --run_name resnet18_finetune
```

### 3. MobileNetV2
```bash
python -m src.train --data_dir "D:\path\to\dataset" --model_name mobilenet_v2 --train_mode finetune --augment --use_wandb --run_name mobilenet_finetune
```

### 4. Export ONNX
```bash
python -m src.export_onnx --checkpoint outputs/resnet18_finetune/best_model.pt --config outputs/resnet18_finetune/config.json --output_onnx outputs/models/best_model.onnx
```

### 5. Inference ONNX
```bash
python -m src.infer_onnx --onnx_path outputs/models/best_model.onnx --class_to_idx outputs/resnet18_finetune/class_to_idx.json --data_dir "D:\path\to\dataset"
```

### 6. Grad-CAM
```bash
python -m src.gradcam --checkpoint outputs/resnet18_finetune/best_model.pt --config outputs/resnet18_finetune/config.json --data_dir "D:\path\to\dataset"
```

### 7. Error Analysis
```bash
python -m src.error_analysis --checkpoint outputs/resnet18_finetune/best_model.pt --config outputs/resnet18_finetune/config.json --data_dir "D:\path\to\dataset"
```

### 8. Demo
```bash
streamlit run app.py
```

## W&B Project
https://wandb.ai/phunghuutai07122005-/csc4005-khmt16-01-garbage-classification
