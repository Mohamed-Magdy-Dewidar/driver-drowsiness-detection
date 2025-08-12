#ifndef CONSTANTS_H
#define CONSTANTS_H

namespace DrowsinessDetector
{
    namespace Constants
    {
        constexpr int ESC_KEY = 27;
        constexpr int WAIT_KEY_MS = 3;
        constexpr double EPSILON = 1e-6;
        constexpr int MAX_LOG_ENTRIES = 1000;
        constexpr int FACE_LANDMARK_COUNT = 68;
    }

    namespace LandmarkIndices
    {
        constexpr int LEFT_EYE_START = 42, LEFT_EYE_END = 47;
        constexpr int RIGHT_EYE_START = 36, RIGHT_EYE_END = 41;
        constexpr int MOUTH_START = 60, MOUTH_END = 67;
    }
}

#endif // CONSTANTS_H