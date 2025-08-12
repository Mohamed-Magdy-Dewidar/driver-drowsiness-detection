#ifndef DROWSINESS_DETECTION_SYSTEM_H
#define DROWSINESS_DETECTION_SYSTEM_H

#include <memory>
#include <opencv2/opencv.hpp>
#include "config.h"
#include "driver_state.h"
#include "facial_landmark_detector.h"

namespace DrowsinessDetector
{
    class DrowsinessDetectionSystem
    {
    private:
        Config config_;
        std::unique_ptr<FacialLandmarkDetector> detector_;
        std::unique_ptr<StateTracker> state_tracker_;

    public:
        explicit DrowsinessDetectionSystem(const Config &config);
        bool initialize();
        int run();

    private:
        void processFrame(cv::Mat &frame);
        void drawNoFaceDetected(cv::Mat &frame);
        void drawVisualization(cv::Mat &frame, const cv::Rect &face_rect, DriverState state,
                               double ear, double mar);
        std::string generateStateMessage(DriverState state);
        void cleanup();
    };
}

#endif // DROWSINESS_DETECTION_SYSTEM_H