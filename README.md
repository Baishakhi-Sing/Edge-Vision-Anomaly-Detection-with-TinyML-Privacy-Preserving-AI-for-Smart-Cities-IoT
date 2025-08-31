dge-Vision Anomaly Detection with TinyML

⚡ TinyML-powered anomaly detection on the edge — deploying privacy-preserving computer vision models for smart cities, IoT, and Industry 4.0.

🌍 Motivation :

Modern cities and factories generate massive video data through IoT sensors and CCTV cameras.
Sending all video streams to the cloud is:

❌ Slow (latency issues)
❌ Energy-hungry
❌ Privacy-invasive

👉 The solution: Run AI locally at the edge (TinyML).

This project builds a lightweight, quantized CNN distilled from a heavy vision transformer, deployable on low-power devices (Jetson Nano, Coral TPU).
It detects anomalies (defects, unusual events) in real-time while respecting privacy by blurring sensitive regions on-device.

✨ Features

📂 Uses MVTec-AD dataset (standard benchmark for industrial anomaly detection)
🧠 Baselines: AutoEncoder (AE), PatchCore, SimCLR
🔄 Knowledge distillation from heavy transformer → small CNN
📉 Quantization & pruning for edge deployment
🔒 Privacy-preserving preprocessing (blur on-device before inference)
📊 Tradeoff analysis: FPS / Latency vs Accuracy vs Energy
🎥 Demo video of real-time inference pipeline

📂 Project Structure
edge_anomaly_project/
│
├── data/                   # dataset + sample videos
│   └── sample_video.mp4
│
├── results/                # outputs
│   ├── accuracy_vs_latency.png
│   ├── energy_profile.png
│   ├── fps_latency_energy_table.csv
│   ├── tiny_ae.pth
│   └── demo.mp4
│
├── src/                    # source code
│   ├── main.py             # main training & demo script
│   ├── eval_mvtec.py       # evaluation utilities
│   ├── mvtec_dataset.py    # dataset class (MVTec-AD)
│   └── create_sample_video.py  # generate synthetic test video
│
├── venv/                   # virtual environment
├── requirements.txt
└── README.md               # this file

⚙️ Installation
1. Clone / Setup
git clone https://github.com/Baishakhi-Sing/Edge-Vision-Anomaly-Detection-with-TinyML-Privacy-Preserving-AI-for-Smart-Cities-IoT
cd edge_anomaly_project

# Create virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

🚀 Usage :
Quick test run (synthetic dataset)
python src/main.py --mode quick
Trains a tiny autoencoder
Produces plots + results in results/
Generate demo video
python src/create_sample_video.py
python src/main.py --mode demo --video data/sample_video.mp4
Creates results/demo.mp4 with anomaly overlays
Train on real MVTec-AD (optional, dataset required)
python src/main.py --mode train --epochs 50 --batch_size 16 --dataset_path data/mvtec

📊 Results :

Tradeoff plots: Latency vs Accuracy vs Energy
Anomaly heatmaps over video frames
Quantized CNN achieves >10× smaller footprint while keeping comparable accuracy
Example (from quick run):
Baseline AUC: ~0.30 (toy data)
Quantized AUC: ~0.30 (same)
Latency: ~15 ms/batch on CPU

🌐 Applications :

🏭 Smart Factories — Detect defective products in real-time on conveyor belts
🏙 Smart Cities — Monitor public spaces for unusual activity without cloud upload
🏥 Healthcare IoT — Edge devices detect anomalies in medical sensor streams
🔒 Privacy-first CCTV — Blur faces locally before anomaly analysis
🎥 Example output (results/demo.mp4):

Red-tinted regions = anomalies
FPS/latency measured in real-time

🔮 Future Work :

Integrate ONNX / TensorRT for real hardware acceleration
Deploy on Jetson Nano / Coral TPU
Improve heatmap visualization for clearer anomaly localization
Extend to multimodal IoT data (vision + sensors)

👨‍💻 Authors :

Baishakhi Sing & Srizoni Maity — Project Development & Research
