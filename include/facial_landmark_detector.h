#ifndef FACIAL_LANDMARK_DETECTOR_H
#define FACIAL_LANDMARK_DETECTOR_H

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <dlib/opencv.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing.h>

namespace DrowsinessDetector
{
    class FacialLandmarkDetector
    {
    private:
        dlib::frontal_face_detector face_detector_;
        dlib::shape_predictor landmark_predictor_;
        bool is_initialized_ = false;

    public:
        bool initialize(const std::string &model_path);
        bool detectFaceAndLandmarks(const cv::Mat &frame, cv::Rect &face_rect,
                                    std::vector<cv::Point2f> &left_eye,
                                    std::vector<cv::Point2f> &right_eye,
                                    std::vector<cv::Point2f> &mouth);

    private:
        void extractEyePoints(const dlib::full_object_detection &landmarks, int start, int end,
                              std::vector<cv::Point2f> &eye_points);
        void extractMouthPoints(const dlib::full_object_detection &landmarks,
                                std::vector<cv::Point2f> &mouth_points);
    };
}

#endif // FACIAL_LANDMARK_DETECTOR_H