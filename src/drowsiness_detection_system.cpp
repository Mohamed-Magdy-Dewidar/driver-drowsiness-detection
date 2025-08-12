#include "../include/drowsiness_detection_system.h"
#include "../include/constants.h"
#include "../include/cv_utils.h"
#include "../include/logger.h"
#include <iostream>
#include <filesystem>

namespace DrowsinessDetector
{
    DrowsinessDetectionSystem::DrowsinessDetectionSystem(const Config &config)
        : config_(config),
          detector_(std::make_unique<FacialLandmarkDetector>()),
          state_tracker_(std::make_unique<StateTracker>()) {}

    bool DrowsinessDetectionSystem::initialize()
    {
        return detector_->initialize(config_.model_path);
    }

    int DrowsinessDetectionSystem::run()
    {
        cv::VideoCapture cap;

        // Try to open video file or camera
        if (!config_.video_path.empty() && std::filesystem::exists(config_.video_path))
        {
            cap.open(config_.video_path);
        }
        else
        {
            cap.open(0); // Default camera
        }

        if (!cap.isOpened())
        {
            std::cerr << "Failed to open video source" << std::endl;
            return -1;
        }

        std::cout << "Drowsiness Detection System Started" << std::endl;
        std::cout << "Press ESC to exit" << std::endl;

        cv::Mat frame;
        int frame_count = 0;

        while (cap.read(frame))
        {
            if (frame.empty())
                break;

            // Skip frames for performance if configured
            if (++frame_count % config_.frame_skip != 0)
                continue;

            processFrame(frame);

            if (cv::waitKey(Constants::WAIT_KEY_MS) == Constants::ESC_KEY)
            {
                break;
            }
        }

        cleanup();
        return EXIT_SUCCESS;
    }

    void DrowsinessDetectionSystem::processFrame(cv::Mat &frame)
    {
        cv::Rect face_rect;
        std::vector<cv::Point2f> left_eye, right_eye, mouth;

        bool face_detected = detector_->detectFaceAndLandmarks(frame, face_rect, left_eye, right_eye, mouth);

        if (!face_detected)
        {
            drawNoFaceDetected(frame);
            Logger::log(DriverState::NO_FACE_DETECTED, "No face detected", 0.0, 0.0);
            return;
        }

        // Calculate metrics
        double left_ear = CVUtils::calculateEAR(left_eye);
        double right_ear = CVUtils::calculateEAR(right_eye);
        double avg_ear = (left_ear + right_ear) / 2.0;
        double mar = CVUtils::calculateMAR(mouth);

        // Update state
        DriverState current_state = state_tracker_->updateState(avg_ear, mar, config_);

        // Log significant state changes
        if (current_state != DriverState::ALERT)
        {
            std::string message = generateStateMessage(current_state);
            Logger::log(current_state, message, avg_ear, mar, frame);
        }

        // Draw visualization
        drawVisualization(frame, face_rect, current_state, avg_ear, mar);

        cv::imshow("Drowsiness Detection System", frame);
    }

    void DrowsinessDetectionSystem::drawNoFaceDetected(cv::Mat &frame)
    {
        cv::putText(frame, "No Face Detected", cv::Point(50, 50),
                    cv::FONT_HERSHEY_SIMPLEX, 1.2, cv::Scalar(0, 0, 255), 2);
    }

    void DrowsinessDetectionSystem::drawVisualization(cv::Mat &frame, const cv::Rect &face_rect, DriverState state,
                                                      double ear, double mar)
    {
        // Draw face rectangle
        cv::Scalar color = CVUtils::getStateColor(state, config_);
        cv::rectangle(frame, face_rect, color, 3);

        // State text
        std::string state_text = Logger::stateToString(state);
        cv::putText(frame, state_text, cv::Point(50, 50),
                    cv::FONT_HERSHEY_SIMPLEX, 1.2, color, 3);

        if (config_.show_debug_info)
        {
            // Metrics
            cv::putText(frame, "EAR: " + CVUtils::formatDouble(ear),
                        cv::Point(50, 90), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 2);
            cv::putText(frame, "MAR: " + CVUtils::formatDouble(mar),
                        cv::Point(50, 120), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 2);

            // Eyes closed duration
            double eyes_closed_time = state_tracker_->getEyesClosedDuration();
            cv::putText(frame, "Eyes Closed: " + CVUtils::formatDouble(eyes_closed_time, 1) + "s",
                        cv::Point(50, 150), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 0), 2);

            // Thresholds
            cv::putText(frame, "EAR Thresh: " + CVUtils::formatDouble(config_.ear_threshold),
                        cv::Point(50, frame.rows - 100), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(200, 200, 200), 1);
            cv::putText(frame, "MAR Thresh: " + CVUtils::formatDouble(config_.mar_threshold),
                        cv::Point(50, frame.rows - 80), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(200, 200, 200), 1);
        }
    }

    std::string DrowsinessDetectionSystem::generateStateMessage(DriverState state)
    {
        switch (state)
        {
        case DriverState::DROWSY:
            return "Driver showing signs of drowsiness";
        case DriverState::YAWNING:
            return "Driver is yawning";
        case DriverState::DROWSY_YAWNING:
            return "Driver is drowsy and yawning - HIGH RISK";
        default:
            return "State change detected";
        }
    }

    void DrowsinessDetectionSystem::cleanup()
    {
        cv::destroyAllWindows();
        Logger::shutdown();
        std::cout << "System shutdown complete" << std::endl;
    }
}