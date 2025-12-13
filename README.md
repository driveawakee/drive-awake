# 🚗 Drive-Awake: Real-Time Drowsiness Detection System

Real-time driver fatigue detection using CNN and OpenCV to prevent accidents.

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?logo=opencv)
![Docker](https://img.shields.io/badge/Docker-Ready-blue?logo=docker)

## 🎯 Overview

Drive-Awake monitors driver eye states through webcam and triggers audio alerts when drowsiness is detected, helping prevent accidents caused by fatigue.

**Key Results:**
- 🎯 **92%+ accuracy** on validation set
- ⚡ **<100ms inference** time for real-time processing
- 🔔 **Audio alert system** with customizable thresholds
- 🐳 **Docker deployment** ready

## ✨ Features

- 👁️ **Real-time eye state monitoring** using CNN
- 📹 **Webcam integration** with OpenCV
- 🧠 **Deep learning model** trained on eye state dataset
- 🔊 **Audio alarm system** for immediate alerts
- 📊 **Performance metrics** tracking
- 🐳 **Dockerized** for easy deployment

## 🛠️ Tech Stack

**ML/DL:** TensorFlow, Keras, CNN  
**Computer Vision:** OpenCV  
**Deployment:** Docker, Python 3.8+

## 🚀 Quick Start

### Using Python
```bash
# Clone repository
git clone https://github.com/driveawakee/drive-awake.git
cd drive-awake

# Install dependencies
pip install -r requirements.txt

# Run detection
python Final.py
```

### Using Docker
```bash
# Build image
docker build -t drive-awake .

# Run container
docker run --device=/dev/video0 -it drive-awake
```

## 📊 Performance Metrics

| Metric | Score |
|--------|-------|
| Accuracy | 92.3% |
| Precision | 91.8% |
| Recall | 93.1% |
| F1-Score | 92.4% |
| Inference Time | <100ms |
| False Positive Rate | 8.2% |

## 🏗️ How It Works

1. **Face Detection:** Detects driver's face using Haar Cascades
2. **Eye Region Extraction:** Isolates eye regions from detected face
3. **CNN Classification:** Classifies eyes as Open or Closed
4. **Drowsiness Logic:** Monitors consecutive closed-eye frames
5. **Alert Trigger:** Sounds alarm if threshold exceeded
```
Webcam → Face Detection → Eye Extraction → CNN Model → 
Drowsiness Logic → Alert System
```

## 📁 Project Structure
```
drive-awake/
├── Final.py                    # Main detection script
├── drowsiness_model.h5         # Trained CNN model
├── alarm.wav                   # Alert sound file
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Docker configuration
└── README.md
```

## 🎓 Model Architecture
```
Input (24x24 grayscale eye image)
    ↓
Conv2D(32) → ReLU → MaxPool
    ↓
Conv2D(64) → ReLU → MaxPool
    ↓
Conv2D(128) → ReLU → MaxPool
    ↓
Flatten → Dense(128) → Dropout(0.5)
    ↓
Dense(2) → Softmax (Open/Closed)
```

## 🔮 Future Enhancements

- [ ] Mobile app integration (iOS/Android)
- [ ] Cloud deployment with real-time monitoring
- [ ] Multi-driver support for commercial vehicles
- [ ] Yawn detection
- [ ] Head pose estimation
- [ ] Dashboard analytics

## 🐛 Known Issues

- Requires good lighting conditions
- Single-driver focus (no multi-face detection yet)
- Requires webcam access

## 🤝 Contributing

Contributions welcome! Please open an issue or submit a PR.

## 📄 License

MIT License - See LICENSE file for details.

## 👤 Authors

**Parth Bijpuriya**  
📧 parthbijpuriya416@gmail.com  
💼 [LinkedIn](https://linkedin.com/in/parth-bijpuriya-821786228)  
🔗 [GitHub](https://github.com/parth656)

## 🙏 Acknowledgments

- CNN architecture inspired by drowsiness detection research
- OpenCV community for computer vision tools
- Dataset: Custom eye state dataset

---

⭐ If you found this project useful, please give it a star!
