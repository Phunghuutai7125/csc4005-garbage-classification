# Experiment Log – Garbage Classification

## Run 1: CNN Baseline
- Model: CNN from scratch (3 conv blocks)
- Key hyperparameters: lr=0.001, batch=32, epochs=20, augment=True
- Validation result: (cập nhật sau khi chạy)
- Test result: (cập nhật sau khi chạy)
- Nhận xét: Baseline đơn giản để so sánh

## Run 2: ResNet18 Finetune
- Model: ResNet18 pretrained ImageNet
- Key hyperparameters: lr=0.0001, batch=32, epochs=15, augment=True
- Validation result: (cập nhật sau khi chạy)
- Test result: (cập nhật sau khi chạy)
- Nhận xét: Transfer learning với toàn bộ backbone

## Run 3: MobileNetV2 Finetune
- Model: MobileNetV2 pretrained ImageNet
- Key hyperparameters: lr=0.0001, batch=32, epochs=15, augment=True
- Validation result: (cập nhật sau khi chạy)
- Test result: (cập nhật sau khi chạy)
- Nhận xét: Mô hình nhẹ hơn ResNet18

## Run 4: ResNet18 Head Only
- Model: ResNet18, chỉ train head
- Key hyperparameters: lr=0.001, batch=32, epochs=15
- Validation result: (cập nhật sau khi chạy)
- Test result: (cập nhật sau khi chạy)
- Nhận xét: So sánh head_only vs finetune

## Run 5: Best Model
- Model: (chọn sau khi có kết quả)
- Key hyperparameters: (cập nhật)
- Validation result: (cập nhật)
- Test result: (cập nhật)
- Nhận xét: Best model được chọn dựa trên val macro-F1
