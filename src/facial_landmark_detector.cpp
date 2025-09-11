#include "../include/facial_landmark_detector.h"
#include "../include/constants.h"
#include <iostream>
#include <algorithm>

namespace DrowsinessDetector
{
    bool FacialLandmarkDetector::initialize(const std::string &model_path)
    {
        try
        {
            face_detector_ = dlib::get_frontal_face_detector();
            dlib::deserialize(model_path) >> landmark_predictor_;
            is_initialized_ = true;
            return true;
        }
        catch (const std::exception &e)
        {
            std::cerr << "Failed to initialize face detector: " << e.what() << std::endl;
            return false;
        }
    }

    bool FacialLandmarkDetector::detectFaceAndLandmarks(const cv::Mat &frame, cv::Rect &face_rect,
                                                        std::vector<cv::Point2f> &left_eye,
                                                        std::vector<cv::Point2f> &right_eye,
                                                        std::vector<cv::Point2f> &mouth)
    {
        if (!is_initialized_ || frame.empty())
            return false;

        try
        {
            dlib::cv_image<dlib::bgr_pixel> dlib_img(frame);
            std::vector<dlib::rectangle> faces = face_detector_(dlib_img);

            if (faces.empty())
                return false;

            // Use the largest face (most confident detection)
            dlib::rectangle face = *std::max_element(faces.begin(), faces.end(),
                                                     [](const dlib::rectangle &a, const dlib::rectangle &b)
                                                     {
                                                         return a.area() < b.area();
                                                     });

            dlib::full_object_detection landmarks = landmark_predictor_(dlib_img, face);
            if (landmarks.num_parts() != Constants::FACE_LANDMARK_COUNT)
                return false;

            // Convert to OpenCV format and extract features
            face_rect = cv::Rect(face.left(), face.top(), face.width(), face.height()) &
                        cv::Rect(0, 0, frame.cols, frame.rows);

            extractEyePoints(landmarks, LandmarkIndices::LEFT_EYE_START, LandmarkIndices::LEFT_EYE_END, left_eye);
            extractEyePoints(landmarks, LandmarkIndices::RIGHT_EYE_START, LandmarkIndices::RIGHT_EYE_END, right_eye);
            extractMouthPoints(landmarks, mouth);
            return true;
        }
        catch (const std::exception &e)
        {
            std::cerr << "Error in face detection: " << e.what() << std::endl;
            return false;
        }
    }

    void FacialLandmarkDetector::extractEyePoints(const dlib::full_object_detection &landmarks, int start, int end,
                                                  std::vector<cv::Point2f> &eye_points)
    {
        eye_points.clear();
        for (int i = start; i <= end; ++i)
        {
            eye_points.emplace_back(landmarks.part(i).x(), landmarks.part(i).y());
        }
    }

    void FacialLandmarkDetector::extractMouthPoints(const dlib::full_object_detection &landmarks,
                                                    std::vector<cv::Point2f> &mouth_points)
    {
        mouth_points.clear();
        for (int i = LandmarkIndices::MOUTH_START; i <= LandmarkIndices::MOUTH_END; ++i)
        {
            mouth_points.emplace_back(landmarks.part(i).x(), landmarks.part(i).y());
        }
    }
}