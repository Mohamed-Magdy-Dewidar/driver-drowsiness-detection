#ifndef LOGGER_H
#define LOGGER_H

#include <string>
#include <chrono>
#include <queue>
#include <mutex>
#include <thread>
#include <atomic>
#include <memory>
#include <fstream>
#include <opencv2/opencv.hpp>
#include "driver_state.h"
#include "config.h"

namespace DrowsinessDetector
{
    struct LogEntry
    {
        std::chrono::system_clock::time_point timestamp;
        DriverState state;
        std::string message;
        double ear_value;
        double mar_value;
        std::string image_filename;

        LogEntry(DriverState s, const std::string &msg, double ear, double mar, const std::string &img = "");
    };

    class Logger
    {
    private:
        // Static members for singleton
        static std::unique_ptr<Logger> instance_;
        static std::once_flag once_flag_;
        static std::mutex instance_mutex_;

        // Instance members
        std::queue<LogEntry> log_queue_;
        std::mutex queue_mutex_;
        std::thread worker_thread_;
        std::atomic<bool> should_stop_{false};
        Config config_;
        bool is_initialized_{false};

        // Private constructor - prevents direct instantiation
        Logger() = default;

        // Delete copy constructor and assignment operator
        Logger(const Logger &) = delete;
        Logger &operator=(const Logger &) = delete;

        // Delete move constructor and assignment operator
        Logger(Logger &&) = delete;
        Logger &operator=(Logger &&) = delete;

    public:
        static Logger &getInstance();

        // Setup configuration - must be called before using the logger
        void setupConfig(const Config &config);

        // Static logging method - easy to call from anywhere
        static void log(DriverState state, const std::string &message, double ear, double mar,
                        const cv::Mat &frame = cv::Mat());

        // Static shutdown method
        static void shutdown();

        // Destructor
        ~Logger();

        static std::string stateToString(DriverState state);

    private:
        void shutdownImpl();
        void logImpl(DriverState state, const std::string &message, double ear, double mar,
                     const cv::Mat &frame = cv::Mat());
        void setupDirectories();
        std::string GetCurrentTimeStamp();
        std::string saveSnapshot(const cv::Mat &frame);
        void printToConsole(const LogEntry &entry);
        void processLogQueue();
        void writeToFile(std::ofstream &file, const LogEntry &entry);
    };
}

#endif // LOGGER_H