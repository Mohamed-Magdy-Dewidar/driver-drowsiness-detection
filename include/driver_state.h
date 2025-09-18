#ifndef DRIVER_STATE_H
#define DRIVER_STATE_H

#include <chrono>
#include "config.h"
#include "head_pose_detector.h"

namespace DrowsinessDetector
{
    enum class DriverState
    {
        ALERT,
        DROWSY,
        YAWNING,
        DROWSY_YAWNING,
        DISTRACTED,        // NEW: Looking away from road
        DROWSY_DISTRACTED, // NEW: Drowsy + looking away
        NO_FACE_DETECTED
    };

    class StateTracker
    {
    private:
        std::chrono::steady_clock::time_point eyes_closed_start_;
        std::chrono::steady_clock::time_point distraction_start_;
        bool eyes_closed_timer_active_ = false;
        bool distraction_timer_active_ = false;
        DriverState last_state_ = DriverState::ALERT;

        bool checkDrowsiness(double ear, const Config &config);
        bool checkYawning(double mar, const Config &config);
        bool checkDistraction(const HeadPose &head_pose, const Config &config);
        DriverState getCurrentDriverState(bool is_drowsy, bool is_yawning, bool is_distracted) const;
        DriverState getCurrentDriverState(bool is_drowsy, bool is_yawning) const;

    public:
        StateTracker() = default;
        ~StateTracker() = default;

        // Updated to include head pose
        DriverState updateState(double ear, double mar, const HeadPose &head_pose, const Config &config);
        DriverState updateState(double ear, double mar, const Config &config);

        DriverState getLastState() const { return last_state_; }
        double getEyesClosedDuration() const;
        double getDistractionDuration() const;
    };
}

#endif // DRIVER_STATE_H