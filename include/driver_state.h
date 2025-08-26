#ifndef DRIVER_STATE_H
#define DRIVER_STATE_H

#include <chrono>
#include "config.h"

namespace DrowsinessDetector
{
    enum class DriverState
    {
        ALERT,
        DROWSY,
        YAWNING,
        DROWSY_YAWNING,
        NO_FACE_DETECTED,
    };

    class StateTracker
    {
    private:
        std::chrono::steady_clock::time_point eyes_closed_start_;
        std::chrono::steady_clock::time_point yawning_start_;
        bool eyes_closed_timer_active_ = false;
        bool yawning_timer_active_ = false;
        DriverState last_state_ = DriverState::ALERT;

    public:
        DriverState updateState(double ear, double mar, const Config &config);
        double getEyesClosedDuration() const;

    private:
        bool checkDrowsiness(double ear, const Config &config);
        bool checkYawning(double mar, const Config &config);
        DriverState getCurrentDriverState(bool is_drowsy , bool is_yawning) const;
    };
}

#endif // DRIVER_STATE_H