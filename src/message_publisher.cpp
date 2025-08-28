#include "../include/message_publisher.h"
#include <iostream>
#include <chrono>

namespace DrowsinessDetector
{
    MessagePublisher::MessagePublisher()
        : context_(nullptr),
          publisher_(nullptr),
          endpoint_(""),
          is_initialized_(false),
          messages_sent_(0),
          failed_sends_(0)
    {
    }

    MessagePublisher::~MessagePublisher()
    {
        shutdown();
    }

    bool MessagePublisher::initialize(const std::string& endpoint)
    {
        std::lock_guard<std::mutex> lock(publisher_mutex_);
        
        try
        {
            // Shutdown existing connection if any
            if (is_initialized_)
            {
                shutdown();
            }
            
            endpoint_ = endpoint;
            
            // Create ZeroMQ context
            context_ = std::make_unique<zmq::context_t>(1);
            
            // Create publisher socket
            publisher_ = std::make_unique<zmq::socket_t>(*context_, ZMQ_PUB);
            
            // Set socket options for better performance
            int linger = 1000; // 1 second linger time
            publisher_->setsockopt(ZMQ_LINGER, &linger, sizeof(linger));
            
            int hwm = 1000; // High water mark - queue up to 1000 messages
            publisher_->setsockopt(ZMQ_SNDHWM, &hwm, sizeof(hwm));
            
            // Bind to endpoint
            publisher_->bind(endpoint_);
            
            is_initialized_ = true;
            messages_sent_ = 0;
            failed_sends_ = 0;
            
            std::cout << "MessagePublisher: Initialized successfully on " << endpoint_ << std::endl;
            return true;
        }
        catch (const zmq::error_t& e)
        {
            std::cerr << "MessagePublisher: ZeroMQ error during initialization: " << e.what() << std::endl;
            is_initialized_ = false;
            return false;
        }
        catch (const std::exception& e)
        {
            std::cerr << "MessagePublisher: Error during initialization: " << e.what() << std::endl;
            is_initialized_ = false;
            return false;
        }
    }

    bool MessagePublisher::publishMessage(const std::string& json_message)
    {
        std::lock_guard<std::mutex> lock(publisher_mutex_);
        
        if (!is_initialized_ || !publisher_)
        {
            failed_sends_++;
            return false;
        }
        
        try
        {
            // Create ZeroMQ message
            zmq::message_t message(json_message.size());
            memcpy(message.data(), json_message.c_str(), json_message.size());
            
            // Send message (non-blocking)
            zmq::send_result_t result = publisher_->send(message, zmq::send_flags::dontwait);
            
            if (result.has_value())
            {
                messages_sent_++;
                return true;
            }
            else
            {
                // Send would block - queue is full
                failed_sends_++;
                std::cerr << "MessagePublisher: Send would block - message queue full" << std::endl;
                return false;
            }
        }
        catch (const zmq::error_t& e)
        {
            failed_sends_++;
            std::cerr << "MessagePublisher: ZeroMQ error during send: " << e.what() << std::endl;
            return false;
        }
        catch (const std::exception& e)
        {
            failed_sends_++;
            std::cerr << "MessagePublisher: Error during send: " << e.what() << std::endl;
            return false;
        }
    }

    bool MessagePublisher::isReady() const
    {
        std::lock_guard<std::mutex> lock(publisher_mutex_);
        return is_initialized_ && publisher_ != nullptr;
    }

    void MessagePublisher::getStats(size_t& sent, size_t& failed) const
    {
        std::lock_guard<std::mutex> lock(publisher_mutex_);
        sent = messages_sent_;
        failed = failed_sends_;
    }

    void MessagePublisher::shutdown()
    {
        if (is_initialized_)
        {
            try
            {
                if (publisher_)
                {
                    publisher_->close();
                    publisher_.reset();
                }
                
                if (context_)
                {
                    context_->close();
                    context_.reset();
                }
                
                is_initialized_ = false;
                
                std::cout << "MessagePublisher: Shutdown complete. Stats - Sent: " 
                         << messages_sent_ << ", Failed: " << failed_sends_ << std::endl;
            }
            catch (const std::exception& e)
            {
                std::cerr << "MessagePublisher: Error during shutdown: " << e.what() << std::endl;
            }
        }
    }
}