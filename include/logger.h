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
#include "message_publisher.h"

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

        std::unique_ptr<MessagePublisher> message_publisher_;
        

        // Statistics
        size_t total_events_logged_;
        size_t images_saved_;

        // Instance members
        std::queue<LogEntry> log_queue_;
        std::mutex queue_mutex_;
        std::thread worker_thread_;
        std::atomic<bool> should_stop_{false};
        Config config_;
        bool is_initialized_{false};

        // Private constructor - prevents direct instantiation
        // Logger() = default;
        Logger() : message_publisher_(nullptr), total_events_logged_(0),
                   images_saved_(0) {}

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

        static void log(DriverState state, const std::string &message,
                        double ear, double mar, const cv::Mat &frame);

        // Get publishing statistics
        void getStats(size_t &events_logged, size_t &images_saved,
                      size_t &messages_sent, size_t &messages_failed) const;

        // Static shutdown method
        static void shutdown();

        // Destructor
        ~Logger();

        static std::string stateToString(DriverState state);

    private:
        void shutdownImpl();
        void logImpl(DriverState state, const std::string &message, double ear, double mar,
                     const cv::Mat &frame);

        void setupDirectories();
        std::string GetCurrentTimeStamp();
        std::string saveSnapshot(const cv::Mat &frame);
        void printToConsole(const LogEntry &entry);
        void processLogQueue();
        std::string LogEntryToJsonString(const LogEntry &entry);
        void writeToFile(std::ofstream &file, const LogEntry &entry);
        void publishMessage(const std::string &json_entry);
        std::string formatLogTimestamp(const std::chrono::system_clock::time_point &tp);
    };
}

#endif // LOGGER_H