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
        bool have_previous_face_location = false;
        cv::Rect last_face_rect_;

    public:
        explicit DrowsinessDetectionSystem(const Config &config);
        bool initialize();
        int run();

    private:
        void processFrame(cv::Mat &frame);
        void process_frame_old_way(cv::Mat &frame);
         

        void showDebugView(const cv::Mat &original_frame, const cv::Mat &roi_frame,
                           const cv::Rect &face_rect, DriverState state,
                           double avg_ear, double mar);
        cv::Rect expandRect(const cv::Rect &rect, double factor, const cv::Size &frame_size);
        void transformCoordinatesBack(cv::Rect &face_rect,
                                      std::vector<cv::Point2f> &left_eye,
                                      std::vector<cv::Point2f> &right_eye,
                                      std::vector<cv::Point2f> &mouth,
                                      double scale_factor,
                                      cv::Point2f roi_offset);
        void updateFaceTracking(const cv::Rect &face_rect, double scale_factor);
        void drawNoFaceDetected(cv::Mat &frame);
        void drawVisualization(cv::Mat &frame, const cv::Rect &face_rect, DriverState state,
                               double ear, double mar);
        std::string generateStateMessage(DriverState state);
        void cleanup();
    };
}

#endif // DROWSINESS_DETECTION_SYSTEM_H