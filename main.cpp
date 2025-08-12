// Enhanced Drowsiness Detection System
// Main entry point - includes the refactored header

#include "drowsiness_detector.h"

// ========================= Main Function =========================
int main(int argc, char *argv[])
{
    using namespace DrowsinessDetector;

    try
    {
        Config config;
        // Initialize the singleton logger
        Logger::getInstance().setupConfig(config);

        DrowsinessDetectionSystem system(config);

        if (!system.initialize())
        {
            std::cerr << "Failed to initialize detection system" << std::endl;
            return -1;
        }

        return system.run();
    }
    catch (const std::exception &e)
    {
        std::cerr << "System error: " << e.what() << std::endl;
        return -1;
    }
    catch (...)
    {
        std::cerr << "Unknown system error occurred" << std::endl;
        return -1;
    }
}
