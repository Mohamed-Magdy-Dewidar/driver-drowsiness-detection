#include "include/drowsiness_detection_system.h"
#include "include/logger.h"
#include "include/config.h"
#include <iostream>

static int alloc_count = 0;
static uint64_t total_allocated_size = 0;

static int freed_alloc_count = 0;
static uint64_t total_freed_size = 0;

void *operator new(size_t size)
{
    alloc_count++;
    total_allocated_size += size;
    return malloc(size);
}

void operator delete(void *ptr) noexcept
{
    if (ptr)
    {
        freed_alloc_count++;

#ifdef _WIN32
        total_freed_size += _msize(ptr);
#elif defined(__linux__)
        total_freed_size += malloc_usable_size(ptr);
#elif defined(__APPLE__)
        total_freed_size += malloc_size(ptr);
#else
        // Fallback - can't track size accurately
        // total_freed_size += 0; // or estimate
#endif

        free(ptr);
    }
}

int main()
{
    try
    {
        // Create configuration
        DrowsinessDetector::Config config;

        // Initialize logger with configuration
        DrowsinessDetector::Logger::getInstance().setupConfig(config);
        // Use:

        // DrowsinessDetector::Logger::getInstance().setupConfig(config, true, "tcp://*:5555");

        // Create and initialize the detection system
        DrowsinessDetector::DrowsinessDetectionSystem system(config);

        if (!system.initialize())
        {
            std::cerr << "Failed to initialize drowsiness detection system" << std::endl;
            return -1;
        }

        // Run the system
        bool result = system.run();
        std::cout << "Number Of Memory Allocations: " << alloc_count << "\n";
        std::cout << "Total allocated size: " << total_allocated_size << " bytes" << std::endl;
        std::cout << "Number Of Memory Deallocations: " << freed_alloc_count << "\n";
        std::cout << "Total freed size: " << total_freed_size << " bytes" << std::endl;
        std::cout << "Net memory allocated (should be 0): " << (total_allocated_size - total_freed_size) << " bytes" << std::endl;
        // Add this to your main() for production metrics
        double allocation_efficiency = (double)total_freed_size / total_allocated_size * 100.0;
        double avg_allocation_size = (double)total_allocated_size / alloc_count;

        std::cout << "Memory efficiency: " << allocation_efficiency << "%" << std::endl;
        std::cout << "Average allocation size: " << avg_allocation_size << " bytes" << std::endl;
        std::cout << "Allocations per frame: " << alloc_count / 823.0 << std::endl;
        return result;
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