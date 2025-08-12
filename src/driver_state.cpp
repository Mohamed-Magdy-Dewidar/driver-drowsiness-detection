#include "../include/driver_state.h"

namespace DrowsinessDetector
{
    DriverState StateTracker::updateState(double ear, double mar, const Config &config)
    {
        bool is_drowsy = checkDrowsiness(ear, config);
        bool is_yawning = checkYawning(mar, config);

        DriverState current_state;
        if (is_drowsy && is_yawning)
        {
            current_state = DriverState::DROWSY_YAWNING;
        }
        else if (is_drowsy)
        {
            current_state = DriverState::DROWSY;
        }
        else if (is_yawning)
        {
            current_state = DriverState::YAWNING;
        }
        else
        {
            current_state = DriverState::ALERT;
        }

        last_state_ = current_state;
        return current_state;
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
}