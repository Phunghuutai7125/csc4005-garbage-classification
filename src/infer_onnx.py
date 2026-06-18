"""
infer_onnx.py – Inference bằng ONNX Runtime
"""
import argparse, json, time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image
from torchvision import transforms


def preprocess(img_path: str, img_size: int = 224) -> np.ndarray:
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    img = Image.open(img_path).convert("RGB")
    return tf(img).unsqueeze(0).numpy()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--onnx_path",     required=True)
    p.add_argument("--class_to_idx",  required=True, help="Path to class_to_idx.json")
    p.add_argument("--image",         default=None,  help="Single image path")
    p.add_argument("--data_dir",      default=None,  help="Thư mục test để batch inference")
    p.add_argument("--img_size",      type=int, default=224)
    p.add_argument("--n_samples",     type=int, default=20, help="Số mẫu để benchmark")
    return p.parse_args()


def main():
    args = parse_args()

    with open(args.class_to_idx) as f:
        class_to_idx = json.load(f)
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    sess = ort.InferenceSession(args.onnx_path, providers=["CPUExecutionProvider"])
    input_name  = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    print(f"[infer] ONNX model: {args.onnx_path}")

    # Single image
    if args.image:
        inp = preprocess(args.image, args.img_size)
        t0 = time.time()
        out = sess.run([output_name], {input_name: inp})[0]
        latency = (time.time() - t0) * 1000
        pred_idx = out.argmax()
        pred_cls = idx_to_class[pred_idx]
        probs = np.exp(out) / np.exp(out).sum()
        print(f"\nPrediction: {pred_cls} ({probs[0][pred_idx]:.2%})")
        print(f"Latency: {latency:.2f} ms")
        print("\nTop-3:")
        top3 = np.argsort(probs[0])[::-1][:3]
        for i in top3:
            print(f"  {idx_to_class[i]}: {probs[0][i]:.2%}")

    # Batch inference + benchmark
    if args.data_dir:
        from pathlib import Path
        import random
        all_imgs = list(Path(args.data_dir).rglob("*.jpg")) + \
                   list(Path(args.data_dir).rglob("*.jpeg")) + \
                   list(Path(args.data_dir).rglob("*.png"))
        random.shuffle(all_imgs)
        samples = all_imgs[:args.n_samples]

        latencies = []
        correct = 0
        print(f"\n[infer] Benchmark trên {len(samples)} mẫu...")
        for img_path in samples:
            true_cls = img_path.parent.name
            inp = preprocess(str(img_path), args.img_size)
            t0 = time.time()
            out = sess.run([output_name], {input_name: inp})[0]
            latencies.append((time.time() - t0) * 1000)
            pred_idx = out.argmax()
            pred_cls = idx_to_class[pred_idx]
            if pred_cls == true_cls:
                correct += 1

        acc = correct / len(samples)
        print(f"\n[infer] Kết quả benchmark:")
        print(f"  Accuracy:        {acc:.4f} ({correct}/{len(samples)})")
        print(f"  Mean latency:    {np.mean(latencies):.2f} ms")
        print(f"  P95 latency:     {np.percentile(latencies, 95):.2f} ms")
        print(f"  Throughput:      {1000 / np.mean(latencies):.1f} img/s")
        print(f"  Model size:      {Path(args.onnx_path).stat().st_size / 1e6:.1f} MB")

        # Lưu kết quả
        result = {
            "onnx_path": str(args.onnx_path),
            "n_samples": len(samples),
            "accuracy": acc,
            "mean_latency_ms": float(np.mean(latencies)),
            "p95_latency_ms": float(np.percentile(latencies, 95)),
            "throughput_img_per_sec": float(1000 / np.mean(latencies)),
            "model_size_mb": Path(args.onnx_path).stat().st_size / 1e6,
        }
        out_path = Path("outputs/metrics/eval_onnx.json")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\n[infer] Saved: {out_path}")


if __name__ == "__main__":
    main()
