#include "include/drowsiness_detection_system.h"
#include "include/logger.h"
#include "include/config.h"
#include <iostream>

int main()
{
    try
    {
        // Create configuration
        DrowsinessDetector::Config config;

        // Initialize logger with configuration
        DrowsinessDetector::Logger::getInstance().setupConfig(config);

        // Create and initialize the detection system
        DrowsinessDetector::DrowsinessDetectionSystem system(config);

        if (!system.initialize())
        {
            std::cerr << "Failed to initialize drowsiness detection system" << std::endl;
            return -1;
        }

        // Run the system
        return system.run();
    }
    catch (const std::exception &e)
    {
        std::cerr << "Fatal error: " << e.what() << std::endl;
        return -1;
    }
    catch (...)
    {
        std::cerr << "Unknown fatal error occurred" << std::endl;
        return -1;
    }
}