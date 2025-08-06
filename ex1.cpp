Improved version with fixes

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <dlib/opencv.h>
#include <dlib/image_processing.h>
#include <dlib/image_io.h>
#include <iostream>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <filesystem>
// ------------------------- Configuration -------------------------
struct Config
{
    double EAR_THRESHOLD = 0.25;
    double MAR_THRESHOLD = 0.7;
    double DROWSY_TIME_SECONDS = 2.0; // Instead of frame count
    double CONFIDENCE_THRESHOLD = 0.7;
    bool SAVE_SNAPSHOTS = true;
    std::string SNAPSHOT_PATH = "snapshots/";
};

enum class DriverState
{
    ALERT,
    DROWSY,
    YAWNING,
    DROWSY_YAWNING,
    CRITICAL
};

// ------------------------- Utility Functions -------------------------
std::string formatDouble(double value, int precision = 3)
{
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(precision) << value;
    return oss.str();
}

std::string stateToString(DriverState state)
{
    switch (state)
    {
    case DriverState::ALERT:
        return "Alert";
    case DriverState::DROWSY:
        return "Drowsy";
    case DriverState::YAWNING:
        return "Yawning";
    case DriverState::DROWSY_YAWNING:
        return "Drowsy + Yawning!";
    case DriverState::CRITICAL:
        return "Critical";
    default:
        return "Unknown";
    }
}

// ------------------------- EAR & MAR calculations -------------------------
double calculateEAR(const std::vector<cv::Point2f> &eye)
{
    if (eye.size() != 6)
        return 0.0; // Error checking

    double A = cv::norm(eye[1] - eye[5]);
    double B = cv::norm(eye[2] - eye[4]);
    double C = cv::norm(eye[0] - eye[3]);

    if (C < 1e-6)
        return 0.0; // Avoid division by zero
    return (A + B) / (2.0 * C);
}

double calculateMAR(const std::vector<cv::Point2f> &mouth)
{
    if (mouth.size() != 8)
        return 0.0; // Error checking

    double A = cv::norm(mouth[3] - mouth[7]);
    double B = cv::norm(mouth[2] - mouth[6]);
    double C = cv::norm(mouth[1] - mouth[5]);
    double D = cv::norm(mouth[0] - mouth[4]);

    if (D < 1e-6)
        return 0.0; // Avoid division by zero
    return (A + B + C) / (2.0 * D);
}

// ------------------------- Face Detection with Error Handling -------------------------
// Fixed RetinaFace detection function based on debug output

// Helper function to generate anchors (simplified version)

std::vector<cv::Point2f> generateAnchors(int feature_size, int stride, const std::vector<float>& scales) {
    std::vector<cv::Point2f> anchors;
    
    for (int y = 0; y < feature_size; y++) {
        for (int x = 0; x < feature_size; x++) {
            for (float scale : scales) {
                float cx = (x + 0.5f) * stride;
                float cy = (y + 0.5f) * stride;
                anchors.push_back(cv::Point2f(cx, cy));
            }
        }
    }
    return anchors;
}

bool detectFace(cv::dnn::Net& model, const cv::Mat& frame, cv::Rect& faceBox, double confidenceThreshold = 0.5) {
    try {
        cv::Mat inputBlob = cv::dnn::blobFromImage(frame, 1.0, cv::Size(640, 640), 
                                                  cv::Scalar(104, 117, 123), true, false);
        model.setInput(inputBlob);
        
        std::vector<cv::Mat> detections;
        model.forward(detections, model.getUnconnectedOutLayersNames());
        
        if (detections.size() != 3) {
            std::cerr << "Expected 3 output layers, got " << detections.size() << std::endl;
            return false;
        }
        
        // Extract the outputs
        cv::Mat bbox_regressions = detections[0];  // [1, 16800, 4]
        cv::Mat classifications = detections[1];   // [1, 16800, 2] 
        cv::Mat landmarks = detections[2];         // [1, 16800, 10]
        
        // Get data pointers
        float* bbox_data = (float*)bbox_regressions.data;
        float* class_data = (float*)classifications.data;
        
        // Image scaling factors
        float scale_x = (float)frame.cols / 640.0f;
        float scale_y = (float)frame.rows / 640.0f;
        
        // Generate anchor points (simplified - you may need to adjust these)
        // RetinaFace typically uses multiple feature levels
        std::vector<cv::Point2f> anchors;
        
        // For 640x640 input, common anchor configurations:
        // Feature levels with different strides
        int feature_sizes[] = {80, 40, 20, 10, 5};  // 640/8, 640/16, 640/32, 640/64, 640/128
        int strides[] = {8, 16, 32, 64, 128};
        std::vector<float> scales = {1.0f, 2.0f};  // Common anchor scales
        
        for (int level = 0; level < 5; level++) {
            auto level_anchors = generateAnchors(feature_sizes[level], strides[level], scales);
            anchors.insert(anchors.end(), level_anchors.begin(), level_anchors.end());
        }
        
        // Make sure we have the right number of anchors
        if (anchors.size() != 16800) {
            // Fallback: simple grid generation
            anchors.clear();
            int grid_size = (int)sqrt(16800 / 2);  // Approximate grid
            for (int i = 0; i < 16800; i++) {
                int y = i / grid_size;
                int x = i % grid_size;
                float cx = (x + 0.5f) * 640.0f / grid_size;
                float cy = (y + 0.5f) * 640.0f / grid_size;
                anchors.push_back(cv::Point2f(cx, cy));
            }
        }
        
        float bestConfidence = 0.0f;
        cv::Rect bestBox;
        
        for (int i = 0; i < 16800; i++) {
            // Get face confidence (second element in classification)
            float face_confidence = class_data[i * 2 + 1];
            
            if (face_confidence > confidenceThreshold && face_confidence > bestConfidence) {
                // Get bounding box regression
                float dx = bbox_data[i * 4 + 0];
                float dy = bbox_data[i * 4 + 1];
                float dw = bbox_data[i * 4 + 2];
                float dh = bbox_data[i * 4 + 3];
                
                // Apply regression to anchor (simplified - may need adjustment)
                cv::Point2f anchor = anchors[i % anchors.size()];
                
                // Convert regression to actual coordinates
                float cx = anchor.x + dx * 16;  // Adjust multiplier as needed
                float cy = anchor.y + dy * 16;
                float w = exp(dw) * 16;
                float h = exp(dh) * 16;
                
                // Convert to top-left, bottom-right
                float x1 = (cx - w/2) * scale_x;
                float y1 = (cy - h/2) * scale_y;
                float x2 = (cx + w/2) * scale_x;
                float y2 = (cy + h/2) * scale_y;
                
                // Clamp to image boundaries
                x1 = std::max(0.0f, std::min((float)frame.cols, x1));
                y1 = std::max(0.0f, std::min((float)frame.rows, y1));
                x2 = std::max(0.0f, std::min((float)frame.cols, x2));
                y2 = std::max(0.0f, std::min((float)frame.rows, y2));
                
                if (x2 > x1 && y2 > y1 && (x2-x1) > 20 && (y2-y1) > 20) {  // Valid box
                    bestBox = cv::Rect(cv::Point((int)x1, (int)y1), cv::Point((int)x2, (int)y2));
                    bestConfidence = face_confidence;
                }
            }
        }
        
        if (bestConfidence > 0) {
            faceBox = bestBox;
            std::cout << "Face detected with confidence: " << bestConfidence << std::endl;
            return true;
        }
        
        return false;
        
    } catch (const cv::Exception& e) {
        std::cerr << "Face detection error: " << e.what() << std::endl;
        return false;
    }
}

// Simplified version focusing just on finding high-confidence detections
bool detectFaceSimple(cv::dnn::Net& model, const cv::Mat& frame, cv::Rect& faceBox, double confidenceThreshold = 0.7) {
    try {
        cv::Mat inputBlob = cv::dnn::blobFromImage(frame, 1.0, cv::Size(640, 640), 
                                                  cv::Scalar(104, 117, 123), true, false);
        model.setInput(inputBlob);
        
        std::vector<cv::Mat> detections;
        model.forward(detections, model.getUnconnectedOutLayersNames());
        
        if (detections.size() != 3) return false;
        
        cv::Mat classifications = detections[1];   // [1, 16800, 2]
        float* class_data = (float*)classifications.data;
        
        // Find the detection with highest face confidence
        int bestIndex = -1;
        float bestConfidence = 0.0f;
        
        for (int i = 0; i < 16800; i++) {
            float face_confidence = class_data[i * 2 + 1];  // Face probability
            
            if (face_confidence > bestConfidence) {
                bestConfidence = face_confidence;
                bestIndex = i;
            }
        }
        
        std::cout << "Best face confidence found: " << bestConfidence 
                  << " at index " << bestIndex << std::endl;
        
        if (bestConfidence > confidenceThreshold) {
            // For now, create a default face box in center of frame
            // You can refine this with proper anchor decoding
            int w = frame.cols / 3;
            int h = frame.rows / 3;
            int x = (frame.cols - w) / 2;
            int y = (frame.rows - h) / 2;
            
            faceBox = cv::Rect(x, y, w, h);
            return true;
        }
        
        return false;
        
    } catch (const cv::Exception& e) {
        std::cerr << "Face detection error: " << e.what() << std::endl;
        return false;
    }
}

// Most practical solution: Use OpenCV's DNN face detector instead
bool detectFaceOpenCVDNN(const cv::Mat& frame, cv::Rect& faceBox, double confidenceThreshold = 0.5) {
    static cv::dnn::Net net;
    static bool initialized = false;
    
    if (!initialized) {
        // Try to load OpenCV's DNN face detector
        try {
            std::string modelPath = "opencv_face_detector_uint8.pb";
            std::string configPath = "opencv_face_detector.pbtxt";
            net = cv::dnn::readNetFromTensorflow(modelPath, configPath);
            initialized = true;
        } catch (...) {
            std::cerr << "Could not load OpenCV face detector" << std::endl;
            return false;
        }
    }
    
    cv::Mat blob = cv::dnn::blobFromImage(frame, 1.0, cv::Size(300, 300), 
                                         cv::Scalar(104, 177, 123));
    net.setInput(blob);
    cv::Mat detection = net.forward();
    
    cv::Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());
    
    float bestConfidence = 0.0f;
    cv::Rect bestBox;
    
    for (int i = 0; i < detectionMat.rows; i++) {
        float confidence = detectionMat.at<float>(i, 2);
        
        if (confidence > confidenceThreshold && confidence > bestConfidence) {
            int x1 = (int)(detectionMat.at<float>(i, 3) * frame.cols);
            int y1 = (int)(detectionMat.at<float>(i, 4) * frame.rows);
            int x2 = (int)(detectionMat.at<float>(i, 5) * frame.cols);
            int y2 = (int)(detectionMat.at<float>(i, 6) * frame.rows);
            
            bestBox = cv::Rect(cv::Point(x1, y1), cv::Point(x2, y2));
            bestConfidence = confidence;
        }
    }
    
    if (bestConfidence > 0) {
        faceBox = bestBox;
        return true;
    }
    
    return false;
}


// ------------------------- Driver State Detection -------------------------
class DriverStateDetector
{
private:
    Config config;
    std::chrono::steady_clock::time_point eyesClosedStartTime;
    bool eyesClosedTimerActive = false;
    int snapshotCounter = 0;


public:
    DriverStateDetector(const Config &cfg) : config(cfg) {}

    DriverState detectState(double ear, double mar, const cv::Mat &frame)
    {
        bool isDrowsy = false;
        bool isYawning = (mar > config.MAR_THRESHOLD);

        // Time-based drowsiness detection instead of frame counting
        if (ear < config.EAR_THRESHOLD)
        {
            if (!eyesClosedTimerActive)
            {
                eyesClosedStartTime = std::chrono::steady_clock::now();
                eyesClosedTimerActive = true;
            }
            else
            {
                auto elapsed = std::chrono::steady_clock::now() - eyesClosedStartTime;
                double elapsedSeconds = std::chrono::duration<double>(elapsed).count();

                if (elapsedSeconds >= config.DROWSY_TIME_SECONDS)
                {
                    isDrowsy = true;

                    // Save snapshot
                    if (config.SAVE_SNAPSHOTS && !frame.empty())
                    {
                        std::string filename = config.SNAPSHOT_PATH + "drowsy_" +
                                               std::to_string(++snapshotCounter) + ".jpg";
                        cv::imwrite(filename, frame);
                    }
                }
            }
        }
        else
        {
            eyesClosedTimerActive = false;
        }

        // Return combined state
        if (isDrowsy && isYawning)
            return DriverState::DROWSY_YAWNING;
        if (isDrowsy)
            return DriverState::DROWSY;
        if (isYawning)
            return DriverState::YAWNING;
        return DriverState::ALERT;
    }
    
    double getEyesClosedDuration() const
    {
        if (!eyesClosedTimerActive)
            return 0.0;
        auto elapsed = std::chrono::steady_clock::now() - eyesClosedStartTime;
        return std::chrono::duration<double>(elapsed).count();
    }
};

// ------------------------- Landmark Extraction -------------------------
bool extractLandmarks(const dlib::shape_predictor &detector, const cv::Mat &face,
                      const cv::Rect &faceBox, std::vector<cv::Point2f> &leftEye,
                      std::vector<cv::Point2f> &rightEye, std::vector<cv::Point2f> &mouth)
{
    try
    {
        // Clear previous data
        leftEye.clear();
        rightEye.clear();
        mouth.clear();

        dlib::cv_image<dlib::bgr_pixel> dlibFace(face);
        dlib::rectangle dlibRect(0, 0, face.cols, face.rows);

        dlib::full_object_detection landmarks = detector(dlibFace, dlibRect);

        if (landmarks.num_parts() != 68)
        {
            return false;
        }

        // Extract eye landmarks (convert to full frame coordinates)
        for (int i = 36; i <= 41; i++)
        {
            rightEye.push_back(cv::Point2f(landmarks.part(i).x() + faceBox.x,
                                           landmarks.part(i).y() + faceBox.y));
        }
        for (int i = 42; i <= 47; i++)
        {
            leftEye.push_back(cv::Point2f(landmarks.part(i).x() + faceBox.x,
                                          landmarks.part(i).y() + faceBox.y));
        }
        for (int i = 60; i <= 67; i++)
        {
            mouth.push_back(cv::Point2f(landmarks.part(i).x() + faceBox.x,
                                        landmarks.part(i).y() + faceBox.y));
        }

        return true;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Landmark detection error: " << e.what() << std::endl;
        return false;
    }
}

int main()
{
    // Configuration
    Config config;
    config.SAVE_SNAPSHOTS = true;

    // File paths
    std::string videoPath = "Videos/SS_Sleepy While driving.mp4";
    std::string modelRetinaFace = "models/retinaface_mobilenet25.onnx";
    std::string modelFaceLandmarks = "models/shape_predictor_68_face_landmarks.dat";

    // Load models
    cv::dnn::Net retinaFaceModel = cv::dnn::readNetFromONNX(modelRetinaFace);
    if (retinaFaceModel.empty())
    {
        std::cerr << "Failed to load RetinaFace ONNX model." << std::endl;
        return -1;
    }

    dlib::shape_predictor landmarkDetector;
    try
    {
        dlib::deserialize(modelFaceLandmarks) >> landmarkDetector;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Failed to load landmark model: " << e.what() << std::endl;
        return -1;
    }

    // Open video
    cv::VideoCapture cap(videoPath);
    if (!cap.isOpened())
    {
        std::cerr << "Failed to open video." << std::endl;
        return -1;
    }

    // Create snapshot directory
    if (!std::filesystem::exists(config.SNAPSHOT_PATH))
    {
        std::filesystem::create_directories(config.SNAPSHOT_PATH);
    }
    DriverStateDetector stateDetector(config);
    cv::Mat frame;

    while (true)
    {
        cap >> frame;
        if (frame.empty())
            break;

        // Detect face
        cv::Rect faceBox;
        if (!detectFaceSimple(retinaFaceModel, frame, faceBox, config.CONFIDENCE_THRESHOLD))
        {
            cv::putText(frame, "No face detected", cv::Point(30, 40),
                        cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);
            cv::imshow("Drowsiness Detector", frame);
            if (cv::waitKey(10) == 27)
                break;
            continue;
        }

        // Extract landmarks
        cv::Mat faceROI = frame(faceBox).clone();
        std::vector<cv::Point2f> leftEye, rightEye, mouth;

        if (!extractLandmarks(landmarkDetector, faceROI, faceBox, leftEye, rightEye, mouth))
        {
            cv::putText(frame, "Landmark detection failed", cv::Point(30, 40),
                        cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);
            cv::imshow("Drowsiness Detector", frame);
            if (cv::waitKey(10) == 27)
                break;
            continue;
        }

        // Calculate ratios
        double earLeft = calculateEAR(leftEye);
        double earRight = calculateEAR(rightEye);
        double ear = (earLeft + earRight) / 2.0;
        double mar = calculateMAR(mouth);

        // Detect driver state
        DriverState driverStatus = stateDetector.detectState(ear, mar, frame);
        std::string statusText = stateToString(driverStatus);

        // Display results
        cv::rectangle(frame, faceBox, cv::Scalar(0, 255, 0), 2);
        cv::Scalar textColor = (driverStatus == DriverState::ALERT) ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255);

        cv::putText(frame, statusText, cv::Point(30, 40), cv::FONT_HERSHEY_SIMPLEX, 1, textColor, 2);
        cv::putText(frame, "EAR: " + formatDouble(ear), cv::Point(30, 80), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 0, 0), 2);
        cv::putText(frame, "MAR: " + formatDouble(mar), cv::Point(30, 110), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 0, 0), 2);
        cv::putText(frame, "Eyes Closed: " + formatDouble(stateDetector.getEyesClosedDuration(), 1) + "s",
                    cv::Point(30, 140), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 0), 1);

        cv::imshow("Drowsiness Detector", frame);
        if (cv::waitKey(10) == 27)
            break;
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}

