#ifndef CV_UTILS_H
#define CV_UTILS_H

#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include "driver_state.h"
#include "config.h"

namespace DrowsinessDetector
{
    namespace CVUtils
    {
        double calculateEAR(const std::vector<cv::Point2f> &eye_points);
        double calculateMAR(const std::vector<cv::Point2f> &mouth_points);
        cv::Scalar getStateColor(DriverState state, const Config &config);
        std::string formatDouble(double value, int precision = 3);
    }
}

#endif // CV_UTILS_H