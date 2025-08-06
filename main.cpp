// drowsiness_detector_dlib.cpp (Raspberry Pi Optimized, Single Face Tracking)
#include <opencv2/opencv.hpp>
#include <dlib/opencv.h>
#include <dlib/image_processing.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <iostream>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <filesystem>
#include <vector>

// ------------------------- Configuration -------------------------
struct Config
{
    double EAR_THRESHOLD = 0.25;
    double MAR_THRESHOLD = 0.7;
    double DROWSY_TIME_SECONDS = 2.0;
    bool SAVE_SNAPSHOTS = true;
    std::string SNAPSHOT_PATH = "snapshots/";
};

enum class DriverState
{
    ALERT,
    DROWSY,
    YAWNING,
    DROWSY_YAWNING
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
        return "Drowsy + Yawning";
    default:
        return "Unknown";
    }
}

// ------------------------- EAR & MAR Calculations -------------------------
double calculateEAR(const std::vector<cv::Point2f> &eye)
{
    if (eye.size() != 6)
        return 0.0;
    double A = cv::norm(eye[1] - eye[5]);
    double B = cv::norm(eye[2] - eye[4]);
    double C = cv::norm(eye[0] - eye[3]);
    return (C < 1e-6) ? 0.0 : (A + B) / (2.0 * C);
}

double calculateMAR(const std::vector<cv::Point2f> &mouth)
{
    if (mouth.size() != 8)
        return 0.0;
    double A = cv::norm(mouth[3] - mouth[7]);
    double B = cv::norm(mouth[2] - mouth[6]);
    double C = cv::norm(mouth[1] - mouth[5]);
    double D = cv::norm(mouth[0] - mouth[4]);
    return (D < 1e-6) ? 0.0 : (A + B + C) / (2.0 * D);
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
        bool isYawning = mar > config.MAR_THRESHOLD;

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
                if (std::chrono::duration<double>(elapsed).count() >= config.DROWSY_TIME_SECONDS)
                {
                    isDrowsy = true;
                    if (config.SAVE_SNAPSHOTS && !frame.empty())
                    {
                        std::string filename = config.SNAPSHOT_PATH + "drowsy_" + std::to_string(++snapshotCounter) + ".jpg";
                        cv::imwrite(filename, frame);
                    }
                }
            }
        }
        else
        {
            eyesClosedTimerActive = false;
        }

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

// ------------------------- Dlib Face Detection -------------------------
bool detectFace(const cv::Mat &frame, dlib::frontal_face_detector &detector, cv::Rect &faceBox)
{
    dlib::cv_image<dlib::bgr_pixel> dlibImg(frame);
    std::vector<dlib::rectangle> faces = detector(dlibImg);

    if (faces.empty())
        return false;
    dlib::rectangle r = faces[0]; // Only one face
    faceBox = cv::Rect(cv::Point(r.left(), r.top()), cv::Point(r.right(), r.bottom())) & cv::Rect(0, 0, frame.cols, frame.rows);
    return true;
}

// ------------------------- Landmark Extraction -------------------------
bool extractLandmarks(const dlib::shape_predictor &detector, const cv::Mat &face,
                      const cv::Rect &faceBox, std::vector<cv::Point2f> &leftEye,
                      std::vector<cv::Point2f> &rightEye, std::vector<cv::Point2f> &mouth)
{
    try
    {
        dlib::cv_image<dlib::bgr_pixel> dlibFace(face);
        dlib::rectangle dlibRect(0, 0, face.cols, face.rows);
        dlib::full_object_detection landmarks = detector(dlibFace, dlibRect);

        if (landmarks.num_parts() != 68)
            return false;

        leftEye.clear();
        rightEye.clear();
        mouth.clear();
        for (int i = 42; i <= 47; ++i)
            leftEye.push_back(cv::Point2f(landmarks.part(i).x() + faceBox.x, landmarks.part(i).y() + faceBox.y));
        for (int i = 36; i <= 41; ++i)
            rightEye.push_back(cv::Point2f(landmarks.part(i).x() + faceBox.x, landmarks.part(i).y() + faceBox.y));
        for (int i = 60; i <= 67; ++i)
            mouth.push_back(cv::Point2f(landmarks.part(i).x() + faceBox.x, landmarks.part(i).y() + faceBox.y));

        return true;
    }
    catch (...)
    {
        return false;
    }
}

// ------------------------- Main -------------------------
int main()
{
    Config config;
    std::string videoPath = "Videos/Sleepy_while_driving.mp4";
    std::string modelFaceLandmarks = "models/shape_predictor_68_face_landmarks.dat";

    // Load models
    dlib::frontal_face_detector faceDetector = dlib::get_frontal_face_detector();
    dlib::shape_predictor landmarkDetector;
    dlib::deserialize(modelFaceLandmarks) >> landmarkDetector;

    if (!std::filesystem::exists(config.SNAPSHOT_PATH))
        std::filesystem::create_directories(config.SNAPSHOT_PATH);

    cv::VideoCapture cap(videoPath);
    if (!cap.isOpened())
    {
        std::cerr << "Failed to open video." << std::endl;
        return -1;
    }

    DriverStateDetector stateDetector(config);
    cv::Mat frame;

    // int count = 0;
    // int n = 5;
    while (cap.read(frame))
    {
        if (frame.empty())
            break;

        // if (++count % n != 0) continue;

        cv::Rect faceBox;
        if (!detectFace(frame, faceDetector, faceBox))
        {
            cv::putText(frame, "No face detected", cv::Point(30, 40), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);
            cv::imshow("Drowsiness Detector", frame);
            if (cv::waitKey(10) == 27)
                break;
            continue;
        }

        cv::Mat faceROI = frame(faceBox);
        std::vector<cv::Point2f> leftEye, rightEye, mouth;
        if (!extractLandmarks(landmarkDetector, faceROI, faceBox, leftEye, rightEye, mouth))
        {
            cv::putText(frame, "Landmark detection failed", cv::Point(30, 40), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);
            cv::imshow("Drowsiness Detector", frame);
            if (cv::waitKey(10) == 27)
                break;
            continue;
        }

        double ear = (calculateEAR(leftEye) + calculateEAR(rightEye)) / 2.0;
        double mar = calculateMAR(mouth);
        DriverState state = stateDetector.detectState(ear, mar, frame);

        cv::rectangle(frame, faceBox, cv::Scalar(0, 255, 0), 2);
        cv::putText(frame, stateToString(state), cv::Point(30, 40), cv::FONT_HERSHEY_SIMPLEX, 1,
                    (state == DriverState::ALERT ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255)), 2);
        cv::putText(frame, "EAR: " + formatDouble(ear), cv::Point(30, 80), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 0, 0), 2);
        cv::putText(frame, "MAR: " + formatDouble(mar), cv::Point(30, 110), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 0, 0), 2);
        cv::putText(frame, "Eyes Closed: " + formatDouble(stateDetector.getEyesClosedDuration(), 1) + "s", cv::Point(30, 140),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 0), 1);

        cv::imshow("Drowsiness Detector", frame);
        if (cv::waitKey(10) == 27)
            break;
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}
