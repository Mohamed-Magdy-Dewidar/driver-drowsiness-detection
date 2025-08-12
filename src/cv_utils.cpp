#include "../include/cv_utils.h"
#include "../include/constants.h"
#include <sstream>
#include <iomanip>

namespace DrowsinessDetector
{
    namespace CVUtils
    {
        double calculateEAR(const std::vector<cv::Point2f> &eye_points)
        {
            if (eye_points.size() != 6)
                return 0.0;

            double vertical1 = cv::norm(eye_points[1] - eye_points[5]);
            double vertical2 = cv::norm(eye_points[2] - eye_points[4]);
            double horizontal = cv::norm(eye_points[0] - eye_points[3]);

            return (horizontal < Constants::EPSILON) ? 0.0 : (vertical1 + vertical2) / (2.0 * horizontal);
        }

        double calculateMAR(const std::vector<cv::Point2f> &mouth_points)
        {
            if (mouth_points.size() != 8)
                return 0.0;

            double vertical1 = cv::norm(mouth_points[3] - mouth_points[7]);
            double vertical2 = cv::norm(mouth_points[2] - mouth_points[6]);
            double vertical3 = cv::norm(mouth_points[1] - mouth_points[5]);
            double horizontal = cv::norm(mouth_points[0] - mouth_points[4]);

            return (horizontal < Constants::EPSILON) ? 0.0 : (vertical1 + vertical2 + vertical3) / (2.0 * horizontal);
        }

        cv::Scalar getStateColor(DriverState state, const Config &config)
        {
            switch (state)
            {
            case DriverState::ALERT:
                return config.alert_color;
            case DriverState::YAWNING:
                return config.warning_color;
            case DriverState::DROWSY:
            case DriverState::DROWSY_YAWNING:
                return config.danger_color;
            default:
                return cv::Scalar(128, 128, 128); // Gray for unknown states
            }
        }

        std::string formatDouble(double value, int precision)
        {
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(precision) << value;
            return oss.str();
        }
    }
}