# 🛑 Driver Drowsiness Detection System (C++ | OpenCV + Dlib)

Real-time driver drowsiness detection system that leverages facial landmarks to track eye and mouth movement using Dlib and OpenCV. The system is architected for asynchronous, distributed operation, making it suitable for a variety of embedded and cloud-integrated setups.

-----

## 🚀 Features

  - 👁️ **EAR (Eye Aspect Ratio)** detection to monitor blinking and eye closure
  - 👄 **MAR (Mouth Aspect Ratio)** to detect yawning
  - 🧠 Simple drowsiness classification: `Alert`, `Drowsy`, or `Sleeping`
  - 📹 **Real-time video feed** with overlayed driver state and facial landmarks
  - 🧾 **Asynchronous Message Passing:** Decoupled C++ and Python services using ZeroMQ for efficient, non-blocking communication.
  - ☁️ **Cloud Integration:** A dedicated Python service handles cloud-related tasks, including log and snapshot uploads to an Amazon S3 bucket.
  - 🔒 Designed for single-face tracking (driver only) for embedded systems
  - 🐧 Compatible with Raspberry Pi (Linux) and Windows (Dev)
  - 🧠 **Clean and Modular C++ Codebase:** Refactored for improved maintainability and scalability.

-----

## 🖥️ Architecture Overview

The system is now split into two main components that communicate asynchronously.

1.  **C++ Drowsiness Detector:** This is the core application that runs on the edge device. It captures video, detects facial landmarks, and determines the driver's state. Instead of writing directly to a file, it publishes events (e.g., drowsiness alerts, image snapshots) as messages over a ZeroMQ socket.

2.  **Python Cloud Service:** This service runs in the background and acts as a ZeroMQ subscriber. It receives the messages from the C++ application and performs cloud-related actions, such as:

      * Uploading log data in **JSON Lines (JSONL)** format to an S3 bucket for efficient, append-only logging.
      * Uploading image snapshots of drowsiness events to S3 for a visual record.

This **PUB/SUB (Publish/Subscribe)** pattern ensures the C++ application's performance isn't affected by network latency or cloud upload times.

-----

## 📁 Project Structure

```bash
📦 driver-drowsiness-detection
├── CMakeLists.txt
├── include/
│   ├── constants.h                   # All constants and landmark indices
│   ├── config.h                      # Configuration structure
│   ├── driver_state.h                # DriverState enum and StateTracker class
│   ├── logger.h                      # Logging system (singleton pattern)
│   ├── cv_utils.h                    # Computer vision utility functions
│   ├── facial_landmark_detector.h    # Face detection and landmark extraction
│   ├── drowsiness_detection_system.h # Main system controller
│   └── message_publisher.h           # ZeroMQ message publisher
├── src/
│   ├── logger.cpp                    # Logger implementation
│   ├── driver_state.cpp              # StateTracker implementation
│   ├── cv_utils.cpp                  # CV utility functions implementation
│   ├── facial_landmark_detector.cpp  # Face detection implementation
│   ├── drowsiness_detection_system.cpp # Main system implementation
│   └── message_publisher.cpp         # ZeroMQ message publisher implementation
├── main.cpp                        # C++ application entry point
├── drowsiness_cloud_service.py     # Python ZeroMQ subscriber for cloud uploads
├── models/
│   └── shape_predictor_68_face_landmarks.dat
├── Videos/
│   └── test.mp4                    # Sample input (optional)
├── build/
└── logs/
    └── drowsiness_log.jsonl        # Log file in JSONL format
```

-----

## 🛠️ Requirements & Dependencies

**C++ Application:**

  - **C++17** or higher
  - **Dlib**
  - **OpenCV 4.x**
  - **ZeroMQ**
  - **CMake**

**Python Cloud Service:**

  - **Python 3.x**
  - **PyZMQ**
  - **Boto3** (for AWS S3)

-----

## ⚙️ Installation & Build

1.  **Clone the repo**

    ```bash
    git clone https://github.com/Mohamed-Magdy-Dewidar/driver-drowsiness-detection.git
    cd driver-drowsiness-detection
    ```

2.  **Download Dlib's pre-trained model**
    Place `shape_predictor_68_face_landmarks.dat` in the `models/` folder.

    ```bash
    wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
    bzip2 -d shape_predictor_68_face_landmarks.dat.bz2
    ```

3.  **Build the C++ project**
    This will compile the C++ application and its ZeroMQ dependencies.

    ```bash
    mkdir build
    cmake -B build
    cmake --build build --config Release
    ```

4.  **Run the services**
    You must run both services to enable cloud functionality.

    ```bash
    # Step 1: Start the Python Cloud Service (in a separate terminal)
    python drowsiness_cloud_service.py

    # Step 2: Run the C++ Drowsiness Detector
    ./build/bin/Release/DrowsinessDetector.exe  # On Windows
    ./build/bin/Release/DrowsinessDetector     # On Linux
    ```

-----

## 🧠 Algorithm Overview

  - The C++ application captures a frame from the webcam.
  - Dlib's `get_frontal_face_detector()` finds the face.
  - The 68-point shape predictor extracts facial landmarks.
  - **EAR** and **MAR** are computed to classify the driver's state.
  - The state (`Alert`, `Drowsy`, `Sleeping`) and any event data are packaged into a JSON message.
  - The `MessagePublisher` sends the message to a local ZeroMQ socket.
  - The Python `drowsiness_cloud_service` receives the message and triggers an upload to S3.

-----

## 📊 Logging Output

Logs are now structured in **JSON Lines (JSONL)** format, which is ideal for streaming and processing structured data one event per line.

```json
{"timestamp": "2025-08-06T14:12:33Z", "level": "INFO", "message": "Drowsiness Detected", "event": "drowsiness_start"}
{"timestamp": "2025-08-06T14:12:36Z", "level": "WARNING", "message": "Driver Sleeping", "event": "sleeping_start"}
```

-----

## 📅 Roadmap

  - **Graceful Shutdown:** Implement a mechanism for the C++ service to send a shutdown signal to the Python service, allowing it to complete all uploads before terminating.
  - **GPIO/ESP32 alert integration**

-----

## 🤝 Contribution

PRs and suggestions welcome\! Please open an issue or fork the repo.

## 📄 License

MIT License. See LICENSE file for details.

## 🙏 Acknowledgments

  - **Dlib**
  - **OpenCV**
  - **ZeroMQ**
  - **Boto3**
  - Tereza Soukupova and Jan Cech's original EAR paper

-----

## 👤 Author

**Mohamed Magdy**