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

        auto start = std::chrono::high_resolution_clock::now();
        int frame_count = 0;
        int processed_frames = 0;
        int sum_frames_processed = 0;

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

            // auto end = std::chrono::high_resolution_clock::now();
            // double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / 1000.0;

            // // Every 5 seconds  prints the average processed FPS. Good way to benchmark how fast the system really runs.
            // if (elapsed >= 5)
            // {
            //     double fps = processed_frames / elapsed;
            //     std::cout << "Processed FPS: " << fps << std::endl;
            //     sum_frames_processed += processed_frames;
            //     processed_frames = 0;
            //     start = end;
            // }

            processFrame(frame);
            // process_frame_old_way(frame);
            processed_frames++;
            if (cv::waitKey(Constants::WAIT_KEY_MS) == Constants::ESC_KEY)
            {
                break;
            }
        }
        std::cout << "Total Processed Frames: " << sum_frames_processed + processed_frames << std::endl;
        cleanup();
        return EXIT_SUCCESS;
    }

    void DrowsinessDetectionSystem::process_frame_old_way(cv::Mat &frame)
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

    void DrowsinessDetectionSystem::processFrame(cv::Mat &frame)
    {
        cv::Mat processed_frame = frame;
        cv::Point2f roi_offset(0, 0);
        double scale_factor = 1.0;
        bool using_optimizations = false;

        // 1. Resize for performance
        if (config_.enable_resize)
        {
            cv::resize(processed_frame, processed_frame,
                       cv::Size(), config_.resize_scale, config_.resize_scale);
            scale_factor = 1.0 / config_.resize_scale;
            using_optimizations = true;
        }

        // 2. Extract ROI
        cv::Mat roi_frame = processed_frame;
        if (config_.enable_roi && have_previous_face_location)
        {
            cv::Rect expanded_roi = expandRect(last_face_rect_, config_.roi_expansion_factor, processed_frame.size());

            // Validate ROI before using it
            if (expanded_roi.width > 0 && expanded_roi.height > 0 &&
                expanded_roi.x >= 0 && expanded_roi.y >= 0 &&
                expanded_roi.x + expanded_roi.width <= processed_frame.cols &&
                expanded_roi.y + expanded_roi.height <= processed_frame.rows)
            {
                roi_frame = processed_frame(expanded_roi);
                roi_offset = cv::Point2f(expanded_roi.x, expanded_roi.y);
                using_optimizations = true;
            }
            else
            {
                // ROI is invalid, reset tracking and use full frame
                std::cout << "Invalid ROI detected, resetting tracking" << std::endl;
                have_previous_face_location = false;
                roi_frame = processed_frame;
            }
        }

        cv::Rect face_rect;
        std::vector<cv::Point2f> left_eye, right_eye, mouth;

        // 3. Detect on optimized frame
        bool face_detected = detector_->detectFaceAndLandmarks(roi_frame, face_rect, left_eye, right_eye, mouth);

        if (!face_detected)
        {
            // Always show a valid frame when no face is detected
            cv::Mat display_frame = (using_optimizations && !config_.show_full_frame) ? roi_frame : frame;
            drawNoFaceDetected(display_frame);
            cv::imshow("Drowsiness Detection System", display_frame);

            Logger::log(DriverState::NO_FACE_DETECTED, "No face detected", 0.0, 0.0);
            have_previous_face_location = false;
            return;
        }

        // 4. Calculate metrics (on ROI coordinates - no transformation needed)
        double left_ear = CVUtils::calculateEAR(left_eye);
        double right_ear = CVUtils::calculateEAR(right_eye);
        double avg_ear = (left_ear + right_ear) / 2.0;
        double mar = CVUtils::calculateMAR(mouth);

        // Update state
        DriverState current_state = state_tracker_->updateState(avg_ear, mar, config_);

        // 5. Smart display logic based on optimizations used
        if (config_.show_full_frame || !using_optimizations)
        {
            // Show full frame mode
            if (using_optimizations)
            {
                // Need to transform coordinates back
                cv::Rect display_face_rect = face_rect;
                std::vector<cv::Point2f> display_left_eye = left_eye;
                std::vector<cv::Point2f> display_right_eye = right_eye;
                std::vector<cv::Point2f> display_mouth = mouth;

                transformCoordinatesBack(display_face_rect, display_left_eye, display_right_eye,
                                         display_mouth, scale_factor, roi_offset);

                drawVisualization(frame, display_face_rect, current_state, avg_ear, mar);
                updateFaceTracking(display_face_rect, scale_factor);
            }
            else
            {
                // No optimizations, coordinates are already in frame space
                drawVisualization(frame, face_rect, current_state, avg_ear, mar);
                last_face_rect_ = face_rect;
                have_previous_face_location = true;
            }

            cv::imshow("Drowsiness Detection System", frame);

            // Log with original frame
            if (current_state != DriverState::ALERT)
            {
                std::string message = generateStateMessage(current_state);
                Logger::log(current_state, message, avg_ear, mar, frame);
            }
        }
        else
        {
            // Show ROI frame mode (only when optimizations are active)
            drawVisualization(roi_frame, face_rect, current_state, avg_ear, mar);
            cv::imshow("Drowsiness Detection System", roi_frame);

            // Update face tracking directly
            last_face_rect_ = face_rect;
            have_previous_face_location = true;

            // Log with ROI frame
            if (current_state != DriverState::ALERT)
            {
                std::string message = generateStateMessage(current_state);
                Logger::log(current_state, message, avg_ear, mar, roi_frame);
            }
        }

        // Optional debug view
        if (config_.show_debug_view && using_optimizations)
        {
            showDebugView(frame, roi_frame, face_rect, current_state, avg_ear, mar);
        }
    }

    // Enhanced expandRect with better validation
    cv::Rect DrowsinessDetectionSystem::expandRect(const cv::Rect &rect, double factor, const cv::Size &frame_size)
    {
        // Validate input rect first
        if (rect.width <= 0 || rect.height <= 0 || factor <= 0)
        {
            return cv::Rect(0, 0, frame_size.width, frame_size.height); // Return full frame as fallback
        }

        cv::Point2f center(rect.x + rect.width / 2.0f, rect.y + rect.height / 2.0f);
        int new_width = static_cast<int>(rect.width * factor);
        int new_height = static_cast<int>(rect.height * factor);

        cv::Rect expanded(
            static_cast<int>(center.x - new_width / 2.0f),
            static_cast<int>(center.y - new_height / 2.0f),
            new_width,
            new_height);

        // Clamp to frame boundaries with extra validation
        expanded.x = std::max(0, expanded.x);
        expanded.y = std::max(0, expanded.y);
        expanded.width = std::min(expanded.width, frame_size.width - expanded.x);
        expanded.height = std::min(expanded.height, frame_size.height - expanded.y);

        // Final validation
        if (expanded.width <= 0 || expanded.height <= 0)
        {
            return cv::Rect(0, 0, frame_size.width, frame_size.height); // Fallback to full frame
        }

        return expanded;
    }

    // Helper functions to add to your class:

    void DrowsinessDetectionSystem::showDebugView(const cv::Mat &original_frame,
                                                  const cv::Mat &roi_frame,
                                                  const cv::Rect &face_rect,
                                                  DriverState state,
                                                  double avg_ear,
                                                  double mar)
    {
        // Create side-by-side comparison
        cv::Mat debug_display;

        // Resize ROI to reasonable size for comparison
        cv::Mat roi_resized;
        int target_height = std::min(300, original_frame.rows);
        double roi_scale = static_cast<double>(target_height) / roi_frame.rows;
        cv::resize(roi_frame, roi_resized, cv::Size(), roi_scale, roi_scale);

        // Resize original frame to match height
        cv::Mat orig_resized;
        double orig_scale = static_cast<double>(target_height) / original_frame.rows;
        cv::resize(original_frame, orig_resized, cv::Size(), orig_scale, orig_scale);

        // Concatenate horizontally
        cv::hconcat(orig_resized, roi_resized, debug_display);

        // Add labels
        cv::putText(debug_display, "Original", cv::Point(10, 30),
                    cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);
        cv::putText(debug_display, "ROI Processing",
                    cv::Point(orig_resized.cols + 10, 30),
                    cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);

        // Add performance info
        std::string perf_info = "EAR: " + std::to_string(avg_ear).substr(0, 4) +
                                " MAR: " + std::to_string(mar).substr(0, 4);
        cv::putText(debug_display, perf_info, cv::Point(10, debug_display.rows - 20),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 1);

        cv::imshow("Debug View", debug_display);
    }


    void DrowsinessDetectionSystem::transformCoordinatesBack(cv::Rect &face_rect,
                                                             std::vector<cv::Point2f> &left_eye,
                                                             std::vector<cv::Point2f> &right_eye,
                                                             std::vector<cv::Point2f> &mouth,
                                                             double scale_factor,
                                                             cv::Point2f roi_offset)
    {
        // Transform face rectangle
        face_rect.x = static_cast<int>((face_rect.x + roi_offset.x) * scale_factor);
        face_rect.y = static_cast<int>((face_rect.y + roi_offset.y) * scale_factor);
        face_rect.width = static_cast<int>(face_rect.width * scale_factor);
        face_rect.height = static_cast<int>(face_rect.height * scale_factor);

        // Transform eye points
        for (auto &point : left_eye)
        {
            point.x = (point.x + roi_offset.x) * scale_factor;
            point.y = (point.y + roi_offset.y) * scale_factor;
        }

        for (auto &point : right_eye)
        {
            point.x = (point.x + roi_offset.x) * scale_factor;
            point.y = (point.y + roi_offset.y) * scale_factor;
        }

        // Transform mouth points
        for (auto &point : mouth)
        {
            point.x = (point.x + roi_offset.x) * scale_factor;
            point.y = (point.y + roi_offset.y) * scale_factor;
        }
    }

    void DrowsinessDetectionSystem::updateFaceTracking(const cv::Rect &face_rect, double scale_factor)
    {
        // Store face location for next frame ROI (scaled down for processing frame coordinates)
        last_face_rect_.x = static_cast<int>(face_rect.x / scale_factor);
        last_face_rect_.y = static_cast<int>(face_rect.y / scale_factor);
        last_face_rect_.width = static_cast<int>(face_rect.width / scale_factor);
        last_face_rect_.height = static_cast<int>(face_rect.height / scale_factor);

        have_previous_face_location = true;
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
