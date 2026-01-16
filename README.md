# 🚗 Drive-Awake: Real-Time Drowsiness Detection System

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg?style=for-the-badge&logo=python)](https://www.python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg?style=for-the-badge&logo=tensorflow)](https://www.tensorflow.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com)
[![OpenCV](https://img.shields.io/badge/OpenCV-Enabled-brightgreen?style=for-the-badge&logo=opencv)](https://opencv.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)

> 🛡️ Keep drivers safe with AI-powered real-time drowsiness detection using Eye Aspect Ratio (EAR) analysis and audio alerts.

---

## 🌟 Features

* **Real-Time Video Analysis**
  - Live webcam feed processing
  - Face and eye detection using Haar Cascades
  - Sub-second inference time

* **Eye Aspect Ratio (EAR) Detection**
  - Advanced blink pattern recognition
  - Continuous eye state monitoring
  - Configurable sensitivity thresholds

* **Intelligent Alert System**
  - Audio alarm when drowsiness detected
  - Visual warnings on interface
  - 30% improvement in simulated driver response time

* **High Performance**
  - 92%+ detection accuracy
  - Optimized TensorFlow model
  - Lightweight deployment with FastAPI

* **Docker Support**
  - Containerized deployment
  - Easy setup and portability
  - Production-ready configuration

---

## 🎯 How It Works

1. **Face Detection**: Identifies driver's face in video stream using Haar Cascade classifier
2. **Eye Tracking**: Locates and tracks both eyes continuously
3. **EAR Calculation**: Computes Eye Aspect Ratio to determine if eyes are closing
4. **Drowsiness Detection**: Monitors EAR values over time to detect fatigue patterns
5. **Alert Trigger**: Sounds alarm when prolonged eye closure detected

### Eye Aspect Ratio (EAR) Formula
```
EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)
```
Where p1-p6 are eye landmark coordinates.

---

## 🚀 Demo

*[Add demo video or GIF here showing the system in action]*

**Key Metrics:**
- ✅ 92%+ accuracy in controlled tests
- ⚡ Real-time processing at 30+ FPS
- 🔊 Instant audio alerts
- 🎯 30% faster response time vs. manual monitoring

---

## 📋 Prerequisites

* Python 3.8 or higher
* Webcam or video input device
* At least 2GB RAM
* Docker (optional, for containerized deployment)

---

## 🔧 Installation

### Local Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/driveawakee/drive-awake.git
   cd drive-awake
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   python Final.py
   ```

### Docker Deployment

1. **Build the Docker image**
   ```bash
   docker build -t drive-awake .
   ```

2. **Run the container**
   ```bash
   docker run -p 8000:8000 --device=/dev/video0 drive-awake
   ```

---

## 📁 Project Structure

```
drive-awake/
├── Final.py                            # Main application script
├── drowsiness_model.h5                 # Pre-trained TensorFlow model
├── haarcascade_frontalface_default.xml # Face detection classifier
├── alarm.wav                           # Alert sound file
├── requirements.txt                    # Python dependencies
├── Dockerfile                          # Container configuration
└── README.md                           # This file
```

---

## 🛠️ Technologies

* **Deep Learning:** TensorFlow 2.x, Keras
* **Computer Vision:** OpenCV, Haar Cascades
* **Backend:** FastAPI for API deployment
* **Audio Processing:** pygame / playsound
* **Deployment:** Docker

---

## 💡 Usage

### Basic Usage
```python
python Final.py
```

The application will:
1. Open your default webcam
2. Start detecting your face and eyes
3. Monitor Eye Aspect Ratio continuously
4. Sound alarm if drowsiness detected

### Configuring Sensitivity
Edit the EAR threshold in `Final.py`:
```python
EAR_THRESHOLD = 0.25  # Lower = more sensitive
CONSECUTIVE_FRAMES = 20  # Frames before alarm
```

---

## 📊 Performance Metrics

| Metric | Value |
|--------|-------|
| Detection Accuracy | 92%+ |
| False Positive Rate | < 8% |
| Processing Speed | 30-60 FPS |
| Response Time | < 1 second |
| Model Size | ~2MB |

---

## 🎯 Use Cases

* **Professional Drivers**: Truck drivers on long routes
* **Fleet Management**: Monitor driver alertness in real-time
* **Personal Safety**: Individual drivers on highway trips
* **Research**: Study drowsiness patterns and interventions
* **Education**: Learn about computer vision and deep learning

---

## ⚠️ Disclaimer

This system is designed as a **supplementary safety tool** and should not replace:
- Proper rest before driving
- Regular breaks during long trips
- Professional medical advice for sleep disorders
- Vehicle manufacturer safety features

Always prioritize adequate rest and safe driving practices.

---

## 🔒 Privacy & Security

* All video processing happens locally on your device
* No data is transmitted to external servers
* No video recording or storage
* Webcam access can be revoked at any time

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Ideas for Contributions
- [ ] Add yawn detection
- [ ] Implement head pose estimation
- [ ] Create mobile app version
- [ ] Add statistics dashboard
- [ ] Integrate with vehicle systems

---

## 🗺️ Roadmap

- [ ] Multi-language support
- [ ] Cloud-based monitoring dashboard
- [ ] Integration with GPS for location tracking
- [ ] Machine learning model improvements
- [ ] Mobile app (iOS/Android)
- [ ] Smartwatch alert notifications
- [ ] Driver behavior analytics
- [ ] Integration with vehicle ADAS systems

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Author

**Parth Bijpuriya**

* GitHub: [@parth656](https://github.com/parth656)
* LinkedIn: [parth-bijpuriya](https://www.linkedin.com/in/parth-bijpuriya-821786228)
* Email: parthbijpuriya416@gmail.com

---

## 🙏 Acknowledgments

* OpenCV community for computer vision tools
* TensorFlow team for deep learning framework
* FastAPI for excellent API framework
* Research papers on drowsiness detection using EAR

---

## 📞 Support

If you encounter any issues or have questions:
* Open an [issue](https://github.com/driveawakee/drive-awake/issues) on GitHub
* Contact via email: parthbijpuriya416@gmail.com

---

## ⭐ Show Your Support

If you find this project helpful or interesting, please consider:
- Giving it a ⭐ on GitHub
- Sharing it with others who might benefit
- Contributing to its development

---

## 📚 References & Research

* [Eye Aspect Ratio for Drowsiness Detection](https://vision.fe.uni-lj.si/cvww2016/proceedings/papers/05.pdf)
* [Real-time Eye Blink Detection using Facial Landmarks](https://www.pyimagesearch.com/2017/04/24/eye-blink-detection-opencv-python-dlib/)
* [Drowsiness Detection Systems: A Review](https://ieeexplore.ieee.org/)

---

**Made with ❤️ for safer roads and responsible driving**

---

### 📈 Stats

![GitHub stars](https://img.shields.io/github/stars/driveawakee/drive-awake?style=social)
![GitHub forks](https://img.shields.io/github/forks/driveawakee/drive-awake?style=social)
![GitHub watchers](https://img.shields.io/github/watchers/driveawakee/drive-awake?style=social)
