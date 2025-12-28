# 🚗 Drive Awake - Driver Drowsiness Detection System

> Real-time driver fatigue detection using deep learning and computer vision to prevent accidents

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-red.svg)](https://opencv.org/)

## 🎯 The Problem

Driver fatigue causes approximately 20% of all road accidents worldwide. Drowsy driving impairs reaction time and decision-making similar to drunk driving, yet most drivers lack tools to monitor their alertness in real-time.

Drive Awake provides an affordable, accurate solution that actively monitors driver alertness through computer vision and immediately alerts before dangerous microsleep episodes occur.

## 🔧 How It Works

1. **Video Capture**: Captures real-time video from webcam or dashboard camera
2. **Face Detection**: CNN model detects driver's face in each frame
3. **Eye State Classification**: Trained CNN model (`drowsiness_model.h5`) classifies eyes as open or closed
4. **Drowsiness Detection**: Monitors consecutive frames of closed eyes
5. **Alert System**: Triggers audio alarm (`alarm.wav`) when drowsiness is detected
6. **Real-time Display**: Shows live feed with drowsiness status overlay

## 💻 Tech Stack

**Core Technologies:**
- **TensorFlow / Keras** - Deep learning framework for CNN model
- **OpenCV** - Real-time computer vision and video processing
- **NumPy** - Numerical computations

**Model:**
- Convolutional Neural Network (CNN)
- Custom trained model for eye state classification
- Model file: `drowsiness_model.h5`

**Deployment:**
- Docker support included
- Lightweight for edge deployment

## ✨ Key Features

✅ **92%+ Accuracy** - High-precision drowsiness detection validated on test data

✅ **Real-time Processing** - Analyzes video streams with minimal latency

✅ **CNN-based Detection** - Deep learning model for robust eye state classification

✅ **Audio Alert System** - Immediate warning sound to wake drowsy drivers

✅ **Webcam Compatible** - Works with any standard webcam or USB camera

✅ **Docker Ready** - Containerized for easy deployment

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8 or higher
- Webcam or video input device
- (Optional) GPU for faster processing

### Installation Steps
```bash
# 1. Clone the repository
git clone https://github.com/driveawakee/drive-awake.git
cd drive-awake

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the application
python Final.py
```

### Docker Deployment
```bash
# Build Docker image
docker build -t drive-awake .

# Run container
docker run --device=/dev/video0 -it drive-awake
```

## 📖 Usage

1. Connect your webcam
2. Run `python Final.py`
3. Position your face in front of the camera
4. The system will monitor your eye state in real-time
5. If drowsiness is detected, an alarm will sound

### Controls
- Press `Q` to quit the application
- Adjust camera position for optimal face detection

## 📊 Performance

- **Accuracy**: 92%+ on test dataset
- **Processing Speed**: Real-time video analysis
- **Detection Latency**: Sub-second alert triggering
- **False Positive Rate**: <5%

## 🎓 What I Learned

- **Deep Learning for CV**: Training CNN models for eye state classification
- **Real-time Video Processing**: Optimizing OpenCV pipelines for low-latency performance
- **Model Deployment**: Packaging ML models for edge devices and production use
- **Safety-Critical Systems**: Building reliable alerting systems for real-world safety applications
- **Docker Containerization**: Creating reproducible deployment environments

## 🚧 Future Improvements

- [ ] Add head pose estimation for additional drowsiness indicators
- [ ] Implement yawning detection
- [ ] Mobile app integration
- [ ] Cloud-based dashboard for fleet monitoring
- [ ] Multi-driver support
- [ ] Integration with vehicle CAN bus
- [ ] Night vision mode support

## 📝 License

This project is open source and available under the MIT License.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 👤 Author

**Parth Bijpuriya**
- GitHub: [@parth656](https://github.com/parth656)
- LinkedIn: [Parth Bijpuriya](https://linkedin.com/in/parth-bijpuriya-821786228)
- Email: parthbijpuriya416@gmail.com

## ⚠️ Disclaimer

This system is a supplementary safety tool and should not replace proper rest and responsible driving practices. Always ensure adequate sleep before driving.

---

⭐ If this project helps make roads safer, please give it a star!
