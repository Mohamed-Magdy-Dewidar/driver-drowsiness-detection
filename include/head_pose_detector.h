#ifndef HEAD_POSE_DETECTOR_H
#define HEAD_POSE_DETECTOR_H

#include <opencv2/opencv.hpp>
#include <dlib/opencv.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing.h>
#include <vector>

namespace DrowsinessDetector
{
    enum class HeadDirection
    {
        FORWARD,
        LOOKING_LEFT,
        LOOKING_RIGHT,
        LOOKING_UP,
        LOOKING_DOWN,
        UNKNOWN
    };

    struct HeadPose
    {
        double pitch;  // X-axis rotation (up/down)
        double yaw;    // Y-axis rotation (left/right) 
        double roll;   // Z-axis rotation (tilt)
        HeadDirection direction;
        bool is_valid;

        HeadPose() : pitch(0.0), yaw(0.0), roll(0.0), direction(HeadDirection::UNKNOWN), is_valid(false) {}
    };

    class HeadPoseDetector
    {
    private:
        // 3D model points for facial landmarks (in mm, relative to nose tip)
        std::vector<cv::Point3f> model_points_;
        
        // Camera matrix and distortion coefficients
        cv::Mat camera_matrix_;
        cv::Mat dist_coeffs_;
        
        bool is_initialized_;
        
        // Thresholds for head direction classification
        struct PoseThresholds
        {
            double yaw_left_threshold = -15.0;    // degrees
            double yaw_right_threshold = 15.0;    // degrees  
            double pitch_up_threshold = 15.0;     // degrees
            double pitch_down_threshold = -15.0;  // degrees
        } thresholds_;

        void initializeModelPoints();
        void initializeCameraMatrix(int img_width, int img_height);
        HeadDirection classifyHeadDirection(double pitch, double yaw) const;
        std::vector<cv::Point2f> extractHeadPosePoints(const dlib::full_object_detection& landmarks) const;

    public:
        HeadPoseDetector();
        ~HeadPoseDetector() = default;

        bool initialize(int img_width, int img_height);
        HeadPose estimatePose(const dlib::full_object_detection& landmarks, int img_width, int img_height);
        
        // Utility methods
        static std::string headDirectionToString(HeadDirection direction);
        void setThresholds(double yaw_left, double yaw_right, double pitch_up, double pitch_down);
        
        bool isInitialized() const { return is_initialized_; }
    };
}

#endif // HEAD_POSE_DETECTOR_H