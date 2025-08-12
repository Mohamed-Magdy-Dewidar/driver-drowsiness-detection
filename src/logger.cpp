#include "../include/logger.h"
#include "../include/constants.h"
#include <iostream>
#include <iomanip>
#include <sstream>
#include <filesystem>

namespace DrowsinessDetector
{
    // Static member definitions
    std::unique_ptr<Logger> Logger::instance_ = nullptr;
    std::once_flag Logger::once_flag_;
    std::mutex Logger::instance_mutex_;

    // LogEntry constructor
    LogEntry::LogEntry(DriverState s, const std::string &msg, double ear, double mar, const std::string &img)
        : timestamp(std::chrono::system_clock::now()), state(s), message(msg),
          ear_value(ear), mar_value(mar), image_filename(img) {}

    Logger &Logger::getInstance()
    {
        std::call_once(once_flag_, []()
                       { instance_ = std::unique_ptr<Logger>(new Logger()); });
        std::cout << "Logging Singleton instance at: " << instance_.get() << std::endl;
        return *instance_;
    }

    void Logger::setupConfig(const Config &config)
    {
        std::lock_guard<std::mutex> lock(instance_mutex_);

        if (is_initialized_)
        {
            std::cerr << "Warning: Logger already initialized. Config changes ignored." << std::endl;
            return;
        }

        config_ = config;
        setupDirectories();

        if (config_.enable_file_logging)
        {
            worker_thread_ = std::thread(&Logger::processLogQueue, this);
        }

        is_initialized_ = true;
        std::cout << "Logger initialized successfully" << std::endl;
    }

    void Logger::log(DriverState state, const std::string &message, double ear, double mar,
                     const cv::Mat &frame)
    {
        Logger &logger = getInstance();
        std::cout << "Logging Singleton instance at: " << logger.instance_.get() << std::endl;

        if (!logger.is_initialized_)
        {
            std::cerr << "Error: Logger not initialized. Call setupConfig() first." << std::endl;
            return;
        }

        logger.logImpl(state, message, ear, mar, frame);
    }

    void Logger::shutdown()
    {
        std::lock_guard<std::mutex> lock(instance_mutex_);
        if (instance_ && instance_->is_initialized_)
        {
            instance_->shutdownImpl();
        }
    }

    Logger::~Logger()
    {
        if (is_initialized_)
        {
            std::cout << "[Logging Destructor] Singleton destroyed at " << this << std::endl;
            shutdownImpl();
        }
    }

    std::string Logger::stateToString(DriverState state)
    {
        switch (state)
        {
        case DriverState::ALERT:
            return "ALERT";
        case DriverState::DROWSY:
            return "DROWSY";
        case DriverState::YAWNING:
            return "YAWNING";
        case DriverState::DROWSY_YAWNING:
            return "DROWSY_YAWNING";
        case DriverState::NO_FACE_DETECTED:
            return "NO_FACE";
        default:
            return "UNKNOWN";
        }
    }

    void Logger::shutdownImpl()
    {
        should_stop_ = true;
        if (worker_thread_.joinable())
        {
            worker_thread_.join();
        }
        is_initialized_ = false;
        std::cout << "Logger shutdown complete" << std::endl;
    }

    void Logger::logImpl(DriverState state, const std::string &message, double ear, double mar,
                         const cv::Mat &frame)
    {
        std::string image_filename;
        if (config_.save_snapshots && !frame.empty() &&
            (state == DriverState::DROWSY || state == DriverState::DROWSY_YAWNING))
        {
            image_filename = saveSnapshot(frame);
        }

        LogEntry entry(state, message, ear, mar, image_filename);

        // Console output (immediate optional)
        // printToConsole(entry);

        // File logging (async)
        if (config_.enable_file_logging)
        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            log_queue_.push(entry);

            // Prevent queue from growing too large
            while (log_queue_.size() > Constants::MAX_LOG_ENTRIES)
            {
                log_queue_.pop();
            }
        }
    }

    void Logger::setupDirectories()
    {
        if (config_.save_snapshots && !std::filesystem::exists(config_.snapshot_path))
        {
            std::filesystem::create_directories(config_.snapshot_path);
        }
        if (config_.enable_file_logging && !std::filesystem::exists(config_.log_path))
        {
            std::filesystem::create_directories(config_.log_path);
        }
    }

    std::string Logger::GetCurrentTimeStamp()
    {
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;
        std::stringstream ss;

        // Correct and descriptive format
        ss << std::put_time(std::localtime(&time_t), "%b%d_%Y_%Hh%Mm%Ss");
        ss << "_" << std::setw(3) << std::setfill('0') << ms.count();
        return ss.str();
    }

    std::string Logger::saveSnapshot(const cv::Mat &frame)
    {
        std::string current_time_stamp = GetCurrentTimeStamp();
        std::string filename = config_.snapshot_path + "drowsy_detected_" + current_time_stamp + ".jpg";
        if (cv::imwrite(filename, frame))
        {
            return filename;
        }
        return "";
    }

    void Logger::printToConsole(const LogEntry &entry)
    {
        auto time_t = std::chrono::system_clock::to_time_t(entry.timestamp);
        std::cout << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S")
                  << " | " << stateToString(entry.state)
                  << " | EAR: " << std::fixed << std::setprecision(3) << entry.ear_value
                  << " | MAR: " << std::fixed << std::setprecision(3) << entry.mar_value
                  << " | " << entry.message << std::endl;
    }

    void Logger::processLogQueue()
    {
        std::ofstream log_file(config_.log_path + config_.log_filename, std::ios::app);

        while (!should_stop_ || !log_queue_.empty())
        {
            std::queue<LogEntry> temp_queue;

            {
                std::lock_guard<std::mutex> lock(queue_mutex_);
                temp_queue.swap(log_queue_);
            }

            while (!temp_queue.empty())
            {
                const auto &entry = temp_queue.front();
                writeToFile(log_file, entry);
                temp_queue.pop();
            }

            log_file.flush();
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }

    void Logger::writeToFile(std::ofstream &file, const LogEntry &entry)
    {
        if (!file.is_open())
            return;

        auto time_t = std::chrono::system_clock::to_time_t(entry.timestamp);
        file << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S")
             << " | State: " << stateToString(entry.state)
             << " | EAR: " << entry.ear_value
             << " | MAR: " << entry.mar_value
             << " | Message: " << entry.message;

        if (!entry.image_filename.empty())
        {
            file << " | Image: " << entry.image_filename;
        }
        file << "\n";
    }
}