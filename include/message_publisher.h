#pragma once

#include <zmq.hpp>
#include <string>
#include <memory>
#include <mutex>

namespace DrowsinessDetector
{
    /**
     * @brief Handles ZeroMQ message publishing with thread safety and error handling
     * 
     * Single responsibility: ZeroMQ communication only
     * - Initializes ZeroMQ context and publisher socket
     * - Publishes JSON messages non-blocking
     * - Handles connection failures gracefully
     */
    class MessagePublisher
    {
    private:
        std::unique_ptr<zmq::context_t> context_;
        std::unique_ptr<zmq::socket_t> publisher_;
        std::string endpoint_;
        bool is_initialized_;
        mutable std::mutex publisher_mutex_;
        
        // Statistics for monitoring
        size_t messages_sent_;
        size_t failed_sends_;
        
    public:
        MessagePublisher();
        ~MessagePublisher();
        
        /**
         * @brief Initialize ZeroMQ publisher
         * @param endpoint ZeroMQ endpoint (e.g., "tcp://*:5555" or "ipc:///tmp/drowsiness_events")
         * @return true if successful, false otherwise
         */
        bool initialize(const std::string& endpoint);
        
        /**
         * @brief Publish a JSON message (non-blocking)
         * @param json_message JSON string to publish
         * @return true if message was queued successfully, false otherwise
         */
        bool publishMessage(const std::string& json_message);
        
        /**
         * @brief Check if publisher is initialized and ready
         * @return true if ready to publish messages
         */
        bool isReady() const;
        
        /**
         * @brief Get publishing statistics
         * @param sent Reference to store messages sent count
         * @param failed Reference to store failed sends count
         */
        void getStats(size_t& sent, size_t& failed) const;
        
        /**
         * @brief Shutdown the publisher gracefully
         */
        void shutdown();
        
        // Delete copy constructor and assignment operator
        MessagePublisher(const MessagePublisher&) = delete;
        MessagePublisher& operator=(const MessagePublisher&) = delete;
    };
}