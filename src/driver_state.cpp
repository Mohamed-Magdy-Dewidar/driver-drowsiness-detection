#include "../include/driver_state.h"

namespace DrowsinessDetector
{
    DriverState StateTracker::updateState(double ear, double mar, const HeadPose &head_pose, const Config &config)
    {
        bool is_drowsy = checkDrowsiness(ear, config);
        bool is_yawning = checkYawning(mar, config);
        DriverState current_state;
        if (config.enable_head_pose_detection)
        {
            bool is_distracted = checkDistraction(head_pose, config);
            current_state = getCurrentDriverState(is_drowsy, is_yawning, is_distracted);
        }
        else
        {
            current_state = getCurrentDriverState(is_drowsy, is_yawning);
        }
        last_state_ = current_state;
        return current_state;
    }

    DriverState StateTracker::updateState(double ear, double mar, const Config &config)
    {
        bool is_drowsy = checkDrowsiness(ear, config);
        bool is_yawning = checkYawning(mar, config);
        DriverState current_state = getCurrentDriverState(is_drowsy, is_yawning);
        last_state_ = current_state;
        return current_state;
    }

    DriverState StateTracker::getCurrentDriverState(bool is_drowsy, bool is_yawning) const
    {
        if (is_drowsy && is_yawning)
        {
            return DriverState::DROWSY_YAWNING;
        }
        else if (is_drowsy)
        {
            return DriverState::DROWSY;
        }
        else if (is_yawning)
        {
            return DriverState::YAWNING;
        }
        else
        {
            return DriverState::ALERT;
        }
    }

    DriverState StateTracker::getCurrentDriverState(bool is_drowsy, bool is_yawning, bool is_distracted) const
    {
        // Priority: Drowsy + Distracted is most dangerous
        if (is_drowsy && is_distracted)
        {
            return DriverState::DROWSY_DISTRACTED;
        }
        else if (is_drowsy && is_yawning)
        {
            return DriverState::DROWSY_YAWNING;
        }
        else if (is_drowsy)
        {
            return DriverState::DROWSY;
        }
        else if (is_distracted)
        {
            return DriverState::DISTRACTED;
        }
        else if (is_yawning)
        {
            return DriverState::YAWNING;
        }
        else
        {
            return DriverState::ALERT;
        }
    }

    double StateTracker::getDistractionDuration() const
    {
        if (!distraction_timer_active_)
            return 0.0;
        auto elapsed = std::chrono::steady_clock::now() - distraction_start_;
        return std::chrono::duration<double>(elapsed).count();
    }

    double StateTracker::getEyesClosedDuration() const
    {
        if (!eyes_closed_timer_active_)
            return 0.0;
        auto elapsed = std::chrono::steady_clock::now() - eyes_closed_start_;
        return std::chrono::duration<double>(elapsed).count();
    }

    bool StateTracker::checkDrowsiness(double ear, const Config &config)
    {
        if (ear < config.ear_threshold)
        {
            if (!eyes_closed_timer_active_)
            {
                eyes_closed_start_ = std::chrono::steady_clock::now();
                eyes_closed_timer_active_ = true;
                return false;
            }

            auto elapsed = std::chrono::steady_clock::now() - eyes_closed_start_;
            return std::chrono::duration<double>(elapsed).count() >= config.drowsy_time_seconds;
        }
        else
        {
            eyes_closed_timer_active_ = false;
            return false;
        }
    }

    bool StateTracker::checkYawning(double mar, const Config &config)
    {
        return mar > config.mar_threshold;
    }

    bool StateTracker::checkDistraction(const HeadPose &head_pose, const Config &config)
    {
        if (!head_pose.is_valid)
        {
            distraction_timer_active_ = false;
            return false;
        }

        // Consider distraction if looking away from forward
        bool is_looking_away = (head_pose.direction != HeadDirection::FORWARD);

        if (is_looking_away)
        {
            if (!distraction_timer_active_)
            {
                distraction_start_ = std::chrono::steady_clock::now();
                distraction_timer_active_ = true;
                return false;
            }

            auto elapsed = std::chrono::steady_clock::now() - distraction_start_;
            return std::chrono::duration<double>(elapsed).count() >= config.distraction_time_seconds;
        }
        else
        {
            distraction_timer_active_ = false;
            return false;
        }
    }
}