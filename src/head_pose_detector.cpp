#include "../include/head_pose_detector.h"
#include "../include/constants.h"
#include "../include/config.h"
#include <iostream>

namespace DrowsinessDetector
{
    HeadPoseDetector::HeadPoseDetector() : is_initialized_(false)
    {
        dist_coeffs_ = cv::Mat::zeros(4, 1, CV_64FC1); // Assume no lens distortion
    }

    bool HeadPoseDetector::initialize(int img_width, int img_height)
    {
        try
        {
            initializeModelPoints();
            initializeCameraMatrix(img_width, img_height);
            is_initialized_ = true;
            return true;
        }
        catch (const std::exception &e)
        {
            std::cerr << "Failed to initialize HeadPoseDetector: " << e.what() << std::endl;
            return false;
        }
    }

    void HeadPoseDetector::initializeModelPoints()
    {
        // 3D model points corresponding to key facial landmarks
        // These are approximate 3D coordinates in mm, relative to nose tip (0,0,0)
        model_points_.clear();
        model_points_.reserve(6);
        model_points_.push_back(cv::Point3f(0.0f, 0.0f, 0.0f));          // Nose tip (landmark 30)
        model_points_.push_back(cv::Point3f(0.0f, -330.0f, -65.0f));     // Chin (landmark 8)
        model_points_.push_back(cv::Point3f(-225.0f, 170.0f, -135.0f));  // Left eye left corner (landmark 36)
        model_points_.push_back(cv::Point3f(225.0f, 170.0f, -135.0f));   // Right eye right corner (landmark 45)
        model_points_.push_back(cv::Point3f(-150.0f, -150.0f, -125.0f)); // Left mouth corner (landmark 48)
        model_points_.push_back(cv::Point3f(150.0f, -150.0f, -125.0f));  // Right mouth corner (landmark 54)
    }

    void HeadPoseDetector::initializeCameraMatrix(int img_width, int img_height)
    {
        double focal_length = static_cast<double>(img_width);
        cv::Point2d center(img_width / 2.0, img_height / 2.0);

        camera_matrix_ = (cv::Mat_<double>(3, 3) << focal_length, 0, center.x,
                          0, focal_length, center.y,
                          0, 0, 1);
    }

    HeadPose HeadPoseDetector::estimatePose(const dlib::full_object_detection &landmarks,
                                            int img_width, int img_height)
    {
        HeadPose pose;

        if (!is_initialized_ || landmarks.num_parts() != Constants::FACE_LANDMARK_COUNT)
        {
            return pose; // Returns invalid pose
        }

        try
        {
            // Extract 2D points corresponding to our 3D model
            std::vector<cv::Point2f> image_points = extractHeadPosePoints(landmarks);

            if (image_points.size() != model_points_.size())
            {
                return pose;
            }

            // Update camera matrix if image size changed
            if (camera_matrix_.at<double>(0, 2) != img_width / 2.0)
            {
                initializeCameraMatrix(img_width, img_height);
            }

            // Solve PnP to get rotation and translation vectors
            cv::Mat rotation_vector, translation_vector;
            bool success = cv::solvePnP(model_points_, image_points, camera_matrix_,
                                        dist_coeffs_, rotation_vector, translation_vector);

            if (!success)
            {
                return pose;
            }

            // Convert rotation vector to rotation matrix
            cv::Mat rotation_matrix;
            cv::Rodrigues(rotation_vector, rotation_matrix);

            // Decompose rotation matrix to get Euler angles
            cv::Mat mtxR, mtxQ, Qx, Qy, Qz;
            cv::Vec3d angles = cv::RQDecomp3x3(rotation_matrix, mtxR, mtxQ, Qx, Qy, Qz);

            // constexpr double RAD2DEG = 180.0 / CV_PI;
            constexpr double RAD2DEG = 1;
            pose.pitch = angles[0] * RAD2DEG;
            pose.yaw = angles[1] * RAD2DEG;
            pose.roll = angles[2] * RAD2DEG;

            // Classify head direction
            pose.direction = classifyHeadDirection(pose.pitch, pose.yaw);
            pose.is_valid = true;

            return pose;
        }
        catch (const std::exception &e)
        {
            std::cerr << "Error in head pose estimation: " << e.what() << std::endl;
            return pose; // Returns invalid pose
        }
    }

    std::vector<cv::Point2f> HeadPoseDetector::extractHeadPosePoints(const dlib::full_object_detection &landmarks) const
    {
        std::vector<cv::Point2f> points;

        // Extract points corresponding to our 3D model
        // Nose tip (landmark 30)
        points.emplace_back(landmarks.part(30).x(), landmarks.part(30).y());

        // Chin (landmark 8)
        points.emplace_back(landmarks.part(8).x(), landmarks.part(8).y());

        // Left eye left corner (landmark 36)
        points.emplace_back(landmarks.part(36).x(), landmarks.part(36).y());

        // Right eye right corner (landmark 45)
        points.emplace_back(landmarks.part(45).x(), landmarks.part(45).y());

        // Left mouth corner (landmark 48)
        points.emplace_back(landmarks.part(48).x(), landmarks.part(48).y());

        // Right mouth corner (landmark 54)
        points.emplace_back(landmarks.part(54).x(), landmarks.part(54).y());

        return points;
    }

    HeadDirection HeadPoseDetector::classifyHeadDirection(double pitch, double yaw) const
    {
        // Check yaw first (left/right)
        if (yaw < thresholds_.yaw_left_threshold)
        {
            return HeadDirection::LOOKING_LEFT;
        }
        else if (yaw > thresholds_.yaw_right_threshold)
        {
            return HeadDirection::LOOKING_RIGHT;
        }

        // Then check pitch (up/down)
        if (pitch < thresholds_.pitch_down_threshold)
        {
            return HeadDirection::LOOKING_DOWN;
        }
        else if (pitch > thresholds_.pitch_up_threshold)
        {
            return HeadDirection::LOOKING_UP;
        }

        return HeadDirection::FORWARD;
    }

    std::string HeadPoseDetector::headDirectionToString(HeadDirection direction)
    {
        switch (direction)
        {
        case HeadDirection::FORWARD:
            return "Forward";
        case HeadDirection::LOOKING_LEFT:
            return "Looking Left";
        case HeadDirection::LOOKING_RIGHT:
            return "Looking Right";
        case HeadDirection::LOOKING_UP:
            return "Looking Up";
        case HeadDirection::LOOKING_DOWN:
            return "Looking Down";
        case HeadDirection::UNKNOWN:
        default:
            return "Unknown";
        }
    }

    void HeadPoseDetector::setThresholds(double yaw_left, double yaw_right, double pitch_up, double pitch_down)

    {
        thresholds_.yaw_left_threshold = yaw_left;
        thresholds_.yaw_right_threshold = yaw_right;
        thresholds_.pitch_up_threshold = pitch_up;
        thresholds_.pitch_down_threshold = pitch_down;
    }
}
