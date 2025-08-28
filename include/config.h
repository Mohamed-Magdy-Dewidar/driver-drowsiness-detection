#ifndef CONFIG_H
#define CONFIG_H

#include <string>
#include <opencv2/opencv.hpp>

namespace DrowsinessDetector
{
    struct Config
    {
        // Detection thresholds
        double ear_threshold = 0.25;
        double mar_threshold = 0.7;
        double drowsy_time_seconds = 2.0;

        // image processing settings
        double resize_scale = 0.5; // 50% reduction = 4x faster processing
        // 0.75 = moderate speedup, better accuracy
        // 0.5 = significant speedup, good accuracy
        // 0.25 = maximum speed, reduced accuracy
        bool enable_resize = false;
        bool enable_roi = false;
        double roi_expansion_factor = 1.5;

        // Display options
        bool show_full_frame = true;  // Default to ROI view for better performance
        bool show_debug_view = false; // Side-by-side comparison for development

        // logging options
        bool enable_console_logging = false;
        bool enable_file_logging_json = true;
        bool save_snapshots = true;
        bool enable_file_logging = true;

        // Paths
        std::string snapshot_path = "snapshots/";
        std::string log_path = "logs/";
        std::string log_filename = "drowsiness_log.jsonl";
        std::string model_path = "models/shape_predictor_68_face_landmarks.dat";
        std::string video_path = "Videos/Sleepy_while_driving.mp4";
        // std::string video_path = "Videos/SS_Sleepy While driving.mp4";

        // Performance settings
        int frame_skip = 1; // Process every N frames


        // Zero Mq Configurations
        std::string zmq_endpoint = "tcp://*:5555";
        // Configuration
        bool enable_publishing_ = true;

        

        // Display settings 
        bool show_debug_info = true;
        cv::Scalar alert_color = cv::Scalar(0, 255, 0);
        cv::Scalar warning_color = cv::Scalar(0, 165, 255);
        cv::Scalar danger_color = cv::Scalar(0, 0, 255);
    };
}

#endif // CONFIG_H