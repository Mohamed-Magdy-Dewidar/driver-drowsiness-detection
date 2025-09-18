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

        // NEW: Head pose detection thresholds
        // double head_pose_yaw_left_threshold = -15.0;   // degrees
        // double head_pose_yaw_right_threshold = 15.0;   // degrees
        // double head_pose_pitch_up_threshold = 15.0;    // degrees
        // double head_pose_pitch_down_threshold = -15.0; // degrees
        // double distraction_time_seconds = 3.0;         // seconds looking away before alert



        // double head_pose_yaw_left_threshold = -35.0;   // degrees
        // double head_pose_yaw_right_threshold = 35.0;   // degrees
        // double head_pose_pitch_up_threshold = 205.0;   // Based on your 175.9° + buffer
        // double head_pose_pitch_down_threshold = 145.0; // Based on your 175.9° - buffer
        // double distraction_time_seconds = 2.0;         // Shorter since we're more accurate



        double head_pose_yaw_left_threshold = -35.0;     // -3.4 - 22° = LEFT
        double head_pose_yaw_right_threshold = 30.0;     // -3.4 + 23° = RIGHT
        double head_pose_pitch_up_threshold = 205;       // -179.7 + 20° = UP
        double head_pose_pitch_down_threshold = -200.0;  // -179.7 - 20° = DOWN
        double distraction_time_seconds = 2.0;

        // Display options and enabling for head pose
        bool show_head_pose_info = true;
        bool show_head_direction_vector = true;
        bool enable_head_pose_detection = false;

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
        // std::string video_path = "Videos/SS_Sleepy While driving.mp4";

        std::string video_path = "Videos/Sleepy_while_driving.mp4";
        // std::string video_path = "Videos/Veo3_1.mp4";
        // std::string video_path = "Videos/Veo3_2.mp4";
        // std::string video_path = "Videos/Veo3_3.mp4";
        // std::string video_path = "Videos/Veo3_4.mp4";
        // std::string video_path = "Videos/Veo3_5.mp4";
        // std::string video_path = "Videos/Veo3_6.mp4";

        // Performance settings
        int frame_skip = 1; // Process every N frames

        // Zero Mq Configurations
        std::string zmq_endpoint = "tcp://*:5555";
        bool enable_publishing_ = false;

        // Display settings
        bool show_debug_info = true;
        cv::Scalar alert_color = cv::Scalar(0, 255, 0);
        cv::Scalar warning_color = cv::Scalar(0, 165, 255);
        cv::Scalar danger_color = cv::Scalar(0, 0, 255);
    };
}

#endif // CONFIG_H