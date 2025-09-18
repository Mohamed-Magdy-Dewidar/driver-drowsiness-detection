#include "../include/drowsiness_detection_system.h"
#include "../include/constants.h"
#include "../include/cv_utils.h"
#include "../include/logger.h"
#include <iostream>
#include <filesystem>

namespace DrowsinessDetector
{
    // DrowsinessDetectionSystem::DrowsinessDetectionSystem(const Config &config)
    //     : config_(config),
    //       detector_(std::make_unique<FacialLandmarkDetector>()),
    //       state_tracker_(std::make_unique<StateTracker>()),
    //       head_pose_detector_(std::make_unique<HeadPoseDetector>())
    // {}
    DrowsinessDetectionSystem::DrowsinessDetectionSystem(const Config &config)
    {
        this->config_ = config;
        this->detector_ = std::make_unique<FacialLandmarkDetector>();
        this->state_tracker_ = std::make_unique<StateTracker>();
        config.enable_head_pose_detection ? this->head_pose_detector_ = std::make_unique<HeadPoseDetector>() : this->head_pose_detector_ = nullptr;
    }

    bool DrowsinessDetectionSystem::initialize()
    {
        bool detector_ok = detector_->initialize(config_.model_path);
        if (!config_.enable_head_pose_detection)
            return detector_ok;

        bool head_pose_ok = head_pose_detector_->initialize(640, 480);

        head_pose_detector_->setThresholds(
            config_.head_pose_yaw_left_threshold,
            config_.head_pose_yaw_right_threshold,
            config_.head_pose_pitch_up_threshold,
            config_.head_pose_pitch_down_threshold);

        return detector_ok && head_pose_ok;
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
        int processed_frames = 0;
        while (cap.read(frame))
        {
            if (frame.empty())
                break;

            // Skip frames for performance if configured
            if (++frame_count % config_.frame_skip != 0)
            {
                std::cout << "Skipping frame " << frame_count << std::endl;
                continue;
            }

            processFrame(frame);
            processed_frames++;
            if (cv::waitKey(Constants::WAIT_KEY_MS) == Constants::ESC_KEY)
            {
                break;
            }
        }
        std::cout << "Total Processed Frames: " << processed_frames << std::endl;
        cleanup();
        return EXIT_SUCCESS;
    }

    void DrowsinessDetectionSystem::processFrame(cv::Mat &frame)
    {
        cv::Rect face_rect;
        std::vector<cv::Point2f> left_eye, right_eye, mouth;
        dlib::full_object_detection all_landmarks;

        // Detect face and landmarks (including full landmarks for head pose)
        bool face_detected = detector_->detectFaceAndAllLandmarks(frame, face_rect,
                                                                  left_eye, right_eye, mouth,
                                                                  all_landmarks);

        if (!face_detected)
        {
            drawNoFaceDetected(frame);
            cv::imshow("Drowsiness Detection System", frame);
            config_.enable_head_pose_detection ? Logger::log(DriverState::NO_FACE_DETECTED, "No face detected", 0.0, 0.0, 0.0, frame) : Logger::log(DriverState::NO_FACE_DETECTED, "No face detected", 0.0, 0.0, frame);
            return;
        }

        // Calculate EAR and MAR
        double left_ear = CVUtils::calculateEAR(left_eye);
        double right_ear = CVUtils::calculateEAR(right_eye);
        double avg_ear = (left_ear + right_ear) / 2.0;
        double mar = CVUtils::calculateMAR(mouth);

        DriverState current_state;
        HeadPose head_pose;
        if (config_.enable_head_pose_detection && head_pose_detector_->isInitialized())
        {
            head_pose = head_pose_detector_->estimatePose(all_landmarks, frame.cols, frame.rows);
            current_state = state_tracker_->updateState(avg_ear, mar, head_pose, config_);
        }
        current_state = state_tracker_->updateState(avg_ear, mar, config_);

        if (config_.enable_head_pose_detection && head_pose_detector_->isInitialized())
            drawVisualization(frame, face_rect, current_state, avg_ear, mar, head_pose);
        else
            drawVisualization(frame, face_rect, current_state, avg_ear, mar);

        // Draw head pose visualization
        if (head_pose.is_valid && config_.show_head_direction_vector && head_pose_detector_->isInitialized() && config_.enable_head_pose_detection)
        {
            drawHeadPoseVisualization(frame, head_pose, all_landmarks);
        }
        cv::imshow("Drowsiness Detection System", frame);

        // Log with head pose data
        if (current_state != DriverState::ALERT)
        {
            std::string message = generateStateMessage(current_state);
            config_.enable_head_pose_detection ? Logger::log(current_state, message, avg_ear, mar, head_pose.yaw, frame) : Logger::log(current_state, message, avg_ear, mar, frame);
        }
    }

    void DrowsinessDetectionSystem::drawNoFaceDetected(cv::Mat &frame)
    {
        cv::putText(frame, "No Face Detected", cv::Point(50, 50),
                    cv::FONT_HERSHEY_SIMPLEX, 1.2, cv::Scalar(0, 0, 255), 2);
    }

    void DrowsinessDetectionSystem::drawVisualization(cv::Mat &frame, const cv::Rect &face_rect, DriverState state,
                                                      double ear, double mar, const HeadPose &head_pose)
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
            // Existing metrics
            cv::putText(frame, "EAR: " + CVUtils::formatDouble(ear),
                        cv::Point(50, 90), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 2);
            cv::putText(frame, "MAR: " + CVUtils::formatDouble(mar),
                        cv::Point(50, 120), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 2);

            // Eyes closed duration
            double eyes_closed_time = state_tracker_->getEyesClosedDuration();
            cv::putText(frame, "Eyes Closed: " + CVUtils::formatDouble(eyes_closed_time, 1) + "s",
                        cv::Point(50, 150), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 0), 2);
        }

        // NEW: Head pose information
        if (config_.show_head_pose_info && head_pose.is_valid)
        {
            int y_offset = config_.show_debug_info ? 180 : 90;

            // Head direction
            std::string direction_text = "Head: " + HeadPoseDetector::headDirectionToString(head_pose.direction);
            cv::Scalar direction_color = (head_pose.direction == HeadDirection::FORWARD) ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 165, 255);
            cv::putText(frame, direction_text, cv::Point(50, y_offset),
                        cv::FONT_HERSHEY_SIMPLEX, 0.8, direction_color, 2);

            if (config_.show_debug_info)
            {
                // Detailed pose angles
                cv::putText(frame, "Pitch: " + CVUtils::formatDouble(head_pose.pitch, 1) + "°",
                            cv::Point(50, y_offset + 30), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(200, 200, 200), 1);
                cv::putText(frame, "Yaw: " + CVUtils::formatDouble(head_pose.yaw, 1) + "°",
                            cv::Point(50, y_offset + 50), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(200, 200, 200), 1);
                cv::putText(frame, "Roll: " + CVUtils::formatDouble(head_pose.roll, 1) + "°",
                            cv::Point(50, y_offset + 70), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(200, 200, 200), 1);

                // Distraction duration
                double distraction_time = state_tracker_->getDistractionDuration();
                if (distraction_time > 0.0)
                {
                    cv::putText(frame, "Distracted: " + CVUtils::formatDouble(distraction_time, 1) + "s",
                                cv::Point(50, y_offset + 100), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 165, 0), 2);
                }
            }
        }

        // Existing threshold display code...
        if (config_.show_debug_info)
        {
            cv::putText(frame, "EAR Thresh: " + CVUtils::formatDouble(config_.ear_threshold),
                        cv::Point(50, frame.rows - 100), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(200, 200, 200), 1);
            cv::putText(frame, "MAR Thresh: " + CVUtils::formatDouble(config_.mar_threshold),
                        cv::Point(50, frame.rows - 80), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(200, 200, 200), 1);

            cv::putText(frame, "Pitch Thresh: " + CVUtils::formatDouble(config_.head_pose_pitch_up_threshold) + "/" + CVUtils::formatDouble(config_.head_pose_pitch_down_threshold),
                        cv::Point(50, frame.rows - 60), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(200, 200, 200), 1);

            cv::putText(frame, "Yaw Thresh: " + CVUtils::formatDouble(config_.head_pose_yaw_left_threshold) + "/" + CVUtils::formatDouble(config_.head_pose_yaw_right_threshold),
                        cv::Point(50, frame.rows - 40), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(200, 200, 200), 1);
        }
    }

    //@ without head pose visualization
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

    void DrowsinessDetectionSystem::drawHeadPoseVisualization(cv::Mat &frame, const HeadPose &head_pose,
                                                              const dlib::full_object_detection &landmarks)
    {
        if (!head_pose.is_valid)
            return;

        // Get nose tip point (landmark 30)
        cv::Point2f nose_tip(landmarks.part(30).x(), landmarks.part(30).y());

        // Calculate direction vector endpoint based on yaw and pitch
        float vector_length = 100.0f;
        cv::Point2f direction_end(
            nose_tip.x + head_pose.yaw * vector_length * 0.1f,
            nose_tip.y - head_pose.pitch * vector_length * 0.1f);

        // Draw direction vector
        cv::Scalar vector_color = (head_pose.direction == HeadDirection::FORWARD) ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 165, 255);
        cv::arrowedLine(frame, nose_tip, direction_end, vector_color, 3);

        // Draw nose tip point
        cv::circle(frame, nose_tip, 5, cv::Scalar(255, 0, 0), -1);
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
        case DriverState::DISTRACTED:
            return "Driver is looking away from the road";
        case DriverState::DROWSY_DISTRACTED:
            return "Driver is drowsy and distracted - CRITICAL RISK";
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
