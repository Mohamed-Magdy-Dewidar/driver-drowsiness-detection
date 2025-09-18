#ifndef DROWSINESS_DETECTION_SYSTEM_H
#define DROWSINESS_DETECTION_SYSTEM_H

#include <memory>
#include <opencv2/opencv.hpp>
#include "config.h"
#include "driver_state.h"
#include "facial_landmark_detector.h"
#include "head_pose_detector.h"

namespace DrowsinessDetector
{
    class DrowsinessDetectionSystem
    {
    private:
        Config config_;
        std::unique_ptr<FacialLandmarkDetector> detector_;
        std::unique_ptr<HeadPoseDetector> head_pose_detector_;
        std::unique_ptr<StateTracker> state_tracker_;
        bool have_previous_face_location = false;
        cv::Rect last_face_rect_;

    public:
        explicit DrowsinessDetectionSystem(const Config &config);
        bool initialize();
        int run();

    private:
        void processFrame(cv::Mat &frame);
        void drawNoFaceDetected(cv::Mat &frame);

        // without head pose visualization
        // void drawVisualization(cv::Mat &frame, const cv::Rect &face_rect, DriverState state,
        //                        double ear, double mar);

        void drawVisualization(cv::Mat &frame, const cv::Rect &face_rect, DriverState state,
                               double ear, double mar, const HeadPose &head_pose);

        void drawVisualization(cv::Mat &frame, const cv::Rect &face_rect, DriverState state,
                                                          double ear, double mar);

            void drawHeadPoseVisualization(cv::Mat &frame, const HeadPose &head_pose,
                                           const dlib::full_object_detection &landmarks);

        std::string generateStateMessage(DriverState state);
        void cleanup();
    };
}

#endif // DROWSINESS_DETECTION_SYSTEM_H