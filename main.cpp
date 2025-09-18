#include "include/drowsiness_detection_system.h"
#include "include/logger.h"
#include "include/config.h"
#include <iostream>

#include <unordered_map>
#include <mutex>

struct MemoryTracker
{
    static int alloc_count;
    static uint64_t total_allocated_size;
    static int freed_alloc_count;
    static uint64_t total_freed_size;
    static void DisplayMemoryUsage()
    {
        // std::cout << "Number Of Memory Allocations: " << alloc_count << "\n";
        // std::cout << "Total allocated size: " << total_allocated_size << " bytes" << std::endl;
        // std::cout << "Number Of Memory Deallocations: " << freed_alloc_count << "\n";
        // std::cout << "Total freed size: " << total_freed_size << " bytes" << std::endl;
        // std::cout << "Net memory allocated : " << (total_allocated_size - total_freed_size) << " bytes" << std::endl;
        // std::cout << "Allocations per frame: " << alloc_count / 823.0 << std::endl;
        double allocation_efficiency = (double)total_freed_size / total_allocated_size * 100.0;
        double avg_allocation_size = (double)total_allocated_size / alloc_count;
        std::cout << "Memory efficiency: " << allocation_efficiency << "%" << std::endl;
        // std::cout << "Average allocation size: " << avg_allocation_size << " bytes" << std::endl;
    }
};

int MemoryTracker::alloc_count = 0;
uint64_t MemoryTracker::total_allocated_size = 0;
int MemoryTracker::freed_alloc_count = 0;
uint64_t MemoryTracker::total_freed_size = 0;

// Memory tracking overrides
void *operator new(size_t size)
{
    MemoryTracker::alloc_count++;
    MemoryTracker::total_allocated_size += size;
    return malloc(size);
}

void operator delete(void *ptr, size_t size) noexcept
{
    if (ptr)
    {
        MemoryTracker::freed_alloc_count++;
        MemoryTracker::total_freed_size += size;
        free(ptr);
    }
}

void operator delete(void *ptr) noexcept
{
    if (ptr)
    {
        MemoryTracker::freed_alloc_count++;

#ifdef _WIN32
        MemoryTracker::total_freed_size += _msize(ptr);
#elif defined(__linux__)
        MemoryTracker::total_freed_size += malloc_usable_size(ptr);
#elif defined(__APPLE__)
        MemoryTracker::total_freed_size += malloc_size(ptr);
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

        // Create and initialize the detection system
        DrowsinessDetector::DrowsinessDetectionSystem system(config);

        if (!system.initialize())
        {
            std::cerr << "Failed to initialize drowsiness detection system" << std::endl;
            return -1;
        }

        // Run the system
        bool result = system.run();
        MemoryTracker::DisplayMemoryUsage();
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