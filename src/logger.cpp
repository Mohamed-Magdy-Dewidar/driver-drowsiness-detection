#include "../include/logger.h"
#include "../include/constants.h"
#include <iostream>
#include <iomanip>
#include <sstream>
#include <filesystem>
#include <nlohmann/json.hpp>

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
        return *instance_;
    }

    void Logger::setupConfig(const Config &config)
    {
        std::lock_guard<std::mutex> lock(instance_mutex_);

        if (is_initialized_)
        {
            std::cerr << "Warning: Logger already initialized. Config changes ignored." << "\n";
            return;
        }

        config_ = config;
        setupDirectories();

        if (config_.enable_file_logging || config_.enable_publishing_)
        {
            worker_thread_ = std::thread(&Logger::processLogQueue , this);
        }
        if (config_.enable_publishing_)
        {
            message_publisher_ = std::make_unique<MessagePublisher>();
            if (!message_publisher_->initialize(config_.zmq_endpoint))
            {
                std::cerr << "Logger: Failed to initialize ZeroMQ publisher, continuing without publishing" << "\n";
                message_publisher_.reset();
            }
            else
            {
                std::cout << "Logger: ZeroMQ publishing enabled on " << config_.zmq_endpoint << "\n";
            }
        }

        is_initialized_ = true;
        std::cout << "Logger initialized successfully" << "\n";
    }

    void Logger::log(DriverState state, const std::string &message, double ear, double mar,
                     const cv::Mat &frame)
    {
        Logger &logger = getInstance();
        if (!logger.is_initialized_)
        {
            std::cerr << "Error: Logger not initialized. Call setupConfig() first." << "\n";
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
            std::cout << "[Logging Destructor] Singleton destroyed at " << this << "\n";
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

        if (message_publisher_)
        {
            message_publisher_->shutdown();
        }

        std::cout << "Logger shutdown complete" << "\n";
    }

    void Logger::logImpl(DriverState state, const std::string &message, double ear, double mar,
                         const cv::Mat &frame)
    {
        std::string image_filename;
        if (config_.save_snapshots && !frame.empty() && (state != DriverState::ALERT))
        {
            image_filename = saveSnapshot(frame);
        }
        if (!image_filename.empty())
            images_saved_++;

        LogEntry entry(state, message, ear, mar, image_filename);

        if (config_.enable_console_logging)
            printToConsole(entry);

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

        std::string filename = config_.snapshot_path + "drowsy_detected_" + GetCurrentTimeStamp() + ".jpg";
        if (cv::imwrite(filename, frame))
        {
            return filename;
        }
        // Log the error to a file or console
        std::cerr << "Logger: Error saving image " << filename << std::endl;
        return "";
    }

    void Logger::printToConsole(const LogEntry &entry)
    {       
        std::cout << this->formatLogTimestamp(entry.timestamp)
                  << " | " << stateToString(entry.state)
                  << " | EAR: " << std::fixed << std::setprecision(3) << entry.ear_value
                  << " | MAR: " << std::fixed << std::setprecision(3) << entry.mar_value
                  << " | " << entry.message << "\n";
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
                total_events_logged_++;
                if (config_.enable_publishing_ && message_publisher_)
                    publishMessage(LogEntryToJsonString(entry));
                temp_queue.pop();
            }

            log_file.flush();
            std::this_thread::sleep_for(std::chrono::milliseconds(400));
        }
    }

    void Logger::publishMessage(const std::string &json_entry)
    {
        if (message_publisher_ && message_publisher_->isReady())
        {
            message_publisher_->publishMessage(json_entry);
        }
    }

    void Logger::getStats(size_t &events_logged, size_t &images_saved,
                          size_t &messages_sent, size_t &messages_failed) const
    {
        events_logged = total_events_logged_;
        images_saved = images_saved_;

        if (message_publisher_)
        {
            message_publisher_->getStats(messages_sent, messages_failed);
        }
        else
        {
            messages_sent = 0;
            messages_failed = 0;
        }
    }

    std::string Logger::formatLogTimestamp(const std::chrono::system_clock::time_point &tp)
    {
        // Convert timestamp to time_t
        std::time_t time_t = std::chrono::system_clock::to_time_t(tp);
        std::stringstream ss;
        ss << std::put_time(std::localtime(&time_t), "%b%d_%Y_%Hh%Mm%Ss");
        return ss.str();
    }

    // in header file std::string LogEntryToJsonString(const LogEntry &entry);
    std::string Logger::LogEntryToJsonString(const LogEntry &entry) 
    {
        nlohmann::json log_json;
        log_json["timestamp"] = this->formatLogTimestamp(entry.timestamp);
        log_json["state"] = stateToString(entry.state);
        log_json["ear"] = entry.ear_value;
        log_json["mar"] = entry.mar_value;
        log_json["message"] = entry.message;
        if (!entry.image_filename.empty())
            log_json["image"] = entry.image_filename;

        return log_json.dump();
    }

    void Logger::writeToFile(std::ofstream &file, const LogEntry &entry)
    {
        if (!file.is_open())
            return;
        std::string log_entry_time_stamp_str = this->formatLogTimestamp(entry.timestamp);
        if (config_.enable_file_logging_json)
        {
            file << LogEntryToJsonString(entry) << "\n";
        }
        else
        {
            // Plain text log entry
            file << log_entry_time_stamp_str
                 << " | State: " << stateToString(entry.state)
                 << " | EAR: " << entry.ear_value
                 << " | MAR: " << entry.mar_value
                 << " | Message: " << entry.message;

            if (!entry.image_filename.empty())
                file << " | Image: " << entry.image_filename;

            file << "\n";
        }
    }

}