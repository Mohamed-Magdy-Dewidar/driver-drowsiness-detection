


# 🛑 Driver Drowsiness Detection System (C++ | OpenCV + Dlib)

Real-time driver drowsiness detection system that leverages facial landmarks to track eye and mouth movement using Dlib and OpenCV. This project is designed with embedded performance in mind and optimized to run on Raspberry Pi (CPU-only) setups.  



---

## 🚀 Features

- 👁️ **EAR (Eye Aspect Ratio)** detection to monitor blinking and eye closure
- 👄 **MAR (Mouth Aspect Ratio)** to detect yawning
- 🧠 Simple drowsiness classification: `Alert`, `Drowsy`, or `Sleeping`
- 📹 Real-time video feed with overlayed driver state and facial landmarks
- 🧾 Built-in logging system (timestamped and singleton-based)
- 🔒 Designed for single-face tracking (driver only) for embedded systems
- 🐧 Compatible with Raspberry Pi (Linux) and Windows (Dev)
- 🧠 Clean and modular C++ codebase

---

## 🖥️ Demo

> A short video showing how the system reacts to blinking and yawning.
> https://drive.google.com/file/d/1K_QaGsBUUUvE0AO1SqazbQoIUuTJqTcm/view?usp=sharing
 
> _(Coming soon. To be added after deployment on Raspberry Pi.)_

---

## 📁 Project Structure

```bash
📦 driver-drowsiness-detection
├── CMakeLists.txt
├── main.cpp
├── include/
│   ├── detector.hpp        # Face & landmark detection
│   ├── ear_mar.hpp         # EAR / MAR calculation
│   ├── logger.hpp          # Logging class
│   └── classifier.hpp      # Driver state classifier
├── models/
│   └── shape_predictor_68_face_landmarks.dat
├── data/
│   └── test.mp4            # Sample input (optional)
└── build/
```


🛠️ Requirements
C++17 or higher

Dlib

OpenCV 4.x

CMake

Raspberry Pi OS / Windows

⚙️ Installation & Build
1. Clone the repo
```bash
git clone https://github.com/your-username/driver-drowsiness-detection.git
cd driver-drowsiness-detection
```
2. Download Dlib's pre-trained model
Place it in the models/ folder:
```bash
wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
bzip2 -d shape_predictor_68_face_landmarks.dat.bz2
```
only do second step if the file does not exists in the repo
3. Build the project
```bash
mkdir build
cmake -B .\build\
build and run exe in release
cmake --build .\build\ --config Release
.\build\Release\OpenCVExample.exe
```

4. Run
```bash
.\build\Release\OpenCVExample.exe   # Or the output binary for your platform
```

🧠 Algorithm Overview
Capture frame from webcam

Detect face using Dlib’s get_frontal_face_detector()

Extract facial landmarks (68-point predictor)

Compute:

EAR from eye points (36–47)

MAR from mouth points (60–67)

Classify driver state:

Alert: EAR > threshold

Drowsy: EAR below threshold briefly

Sleeping: EAR below threshold for prolonged time

Display and log result

📊 Logging Output
Logs are printed to the terminal with timestamps:

```yaml

2025-08-06 14:12:33 - Drowsiness Detected
2025-08-06 14:12:36 - Driver Sleeping
```
Planned: Future versions will support cloud log uploads (e.g., AWS S3).

📌 Notes
This project tracks only one face: the driver.

If no face is detected for several frames, the system assumes the driver moved out of view.

For performance, landmark detection may skip every N frames (tune for Raspberry Pi).

📅 Roadmap
 Thread-safe logging mechanism

 AWS S3 log/image upload

 GPIO/ESP32 alert integration

 Lightweight face mesh using MediaPipe (experimental)

 Unit tests for EAR/MAR and classifier logic

🤝 Contribution
PRs and suggestions welcome! Please open an issue or fork the repo.

📄 License
MIT License. See LICENSE file for details.

🙏 Acknowledgments
Dlib

OpenCV

Tereza Soukupova and Jan Cech's original EAR paper

👤 Author
Mohamed Magdy
Embedded Vision Developer
GitHub | LinkedIn

