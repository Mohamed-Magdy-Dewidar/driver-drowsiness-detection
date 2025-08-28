#!/usr/bin/env python3
"""
Drowsiness Detection Cloud Service

This service:
1. Subscribes to ZeroMQ events from C++ detection system
2. Immediately uploads snapshot images to AWS S3
3. Batches log events and periodically uploads as .jsonl files
4. Organizes files in S3 by date: logs/YYYY/MM/DD/ and snapshots/YYYY/MM/DD/

Requirements:
- pyzmq: pip install pyzmq
- boto3: pip install boto3
- python-dotenv: pip install python-dotenv (optional, for environment variables)
"""

import json
import time
import threading
import queue
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging
import signal
import mimetypes

import zmq
import boto3
from botocore.exceptions import NoCredentialsError, ClientError
from dotenv import load_dotenv

load_dotenv()


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('DrowsinessCloudService')


class S3Manager:
    """Handles all S3 operations with error handling and retry logic"""
    
    def __init__(self, bucket_name: str, region: str = 'us-east-1'):
        self.bucket_name = bucket_name
        self.region = region
        self.s3_client = None
        self._initialize_s3()
        
    def _initialize_s3(self):
        """Initialize S3 client with error handling"""
        try:
            self.s3_client = boto3.client('s3', region_name=self.region)
            # Test connection
            self.s3_client.head_bucket(Bucket=self.bucket_name)
            logger.info(f"S3 connection established to bucket: {self.bucket_name}")
        except NoCredentialsError:
            logger.error("AWS credentials not found. Please configure AWS CLI or set environment variables.")
            sys.exit(1)
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == '404':
                logger.error(f"S3 bucket '{self.bucket_name}' not found.")
            elif error_code == '403':
                logger.error(f"Access denied to S3 bucket '{self.bucket_name}'.")
            else:
                logger.error(f"S3 error: {e}")
            sys.exit(1)
    
    def upload_image(self, local_path: str, s3_key: str, retries: int = 3) -> bool:
        """Upload image file to S3 with retry logic"""
        if not os.path.exists(local_path):
            logger.error(f"Local file not found: {local_path}")
            return False
        
        for attempt in range(retries):
            try:
                self.s3_client.upload_file(
                    local_path, 
                    self.bucket_name, 
                    s3_key,
                    ExtraArgs={'ContentType': 'image/jpeg'}
                )
                logger.info(f"Successfully uploaded image: {s3_key}")
                return True
            except ClientError as e:
                logger.warning(f"Upload attempt {attempt + 1} failed: {e}")
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                    
        logger.error(f"Failed to upload image after {retries} attempts: {s3_key}")
        return False
    
    def upload_jsonl(self, jsonl_content: str, s3_key: str, retries: int = 3) -> bool:
        """Upload JSONL content to S3 with retry logic"""
        for attempt in range(retries):
            try:
                self.s3_client.put_object(
                    Bucket=self.bucket_name,
                    Key=s3_key,
                    Body=jsonl_content.encode('utf-8'),
                    ContentType='application/x-jsonlines'
                )
                logger.info(f"Successfully uploaded JSONL: {s3_key}")
                return True
            except ClientError as e:
                logger.warning(f"JSONL upload attempt {attempt + 1} failed: {e}")
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                    
        logger.error(f"Failed to upload JSONL after {retries} attempts: {s3_key}")
        return False


class EventBuffer:
    """Thread-safe buffer for collecting events before batch upload"""
    
    def __init__(self, flush_interval: int = 60):
        self.events: List[Dict[str, Any]] = []
        self.flush_interval = flush_interval
        self.last_flush = time.time()
        self.lock = threading.Lock()
        
    def add_event(self, event: Dict[str, Any]) -> None:
        """Add event to buffer"""
        with self.lock:
            self.events.append(event)
    
    def should_flush(self) -> bool:
        """Check if buffer should be flushed"""
        return (time.time() - self.last_flush) >= self.flush_interval
    
    def get_and_clear_events(self) -> List[Dict[str, Any]]:
        """Get all events and clear buffer"""
        with self.lock:
            events = self.events.copy()
            self.events.clear()
            self.last_flush = time.time()
            return events
    
    def get_count(self) -> int:
        """Get current event count"""
        with self.lock:
            return len(self.events)


class DrowsinessCloudService:
    """Main service class that coordinates ZeroMQ subscription and S3 uploads"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.s3_manager = S3Manager(config['s3_bucket'], config.get('aws_region', 'us-east-1'))
        self.event_buffer = EventBuffer(config.get('batch_flush_interval', 60))
        self.running = False
        
        # Statistics
        self.stats = {
            'events_received': 0,
            'images_uploaded': 0,
            'images_failed': 0,
            'batches_uploaded': 0,
            'batches_failed': 0,
            'start_time': time.time()
        }
        
        # ZeroMQ setup
        self.context = zmq.Context()
        self.subscriber = self.context.socket(zmq.SUB)
        self.subscriber.connect(config['zmq_endpoint'])
        self.subscriber.setsockopt(zmq.SUBSCRIBE, b"")  # Subscribe to all messages
        
        # Set receive timeout for graceful shutdown
        self.subscriber.setsockopt(zmq.RCVTIMEO, 1000)  # 1 second timeout
        
        logger.info(f"Connected to ZeroMQ at {config['zmq_endpoint']}")
    
    def _get_s3_paths(self, timestamp_str: str, filename: str) -> tuple:
        """Generate S3 paths based on timestamp and filename"""
        try:
            # Parse timestamp - expecting format like "Aug27_2025_06h47m10s_276"
            dt_part = timestamp_str.split('_')[1:3]  # ['2025', '06h47m10s']
            year = dt_part[0]
            
            # Extract month and day from filename if available
            if 'Aug27_2025' in timestamp_str:
                # Parse the date part
                date_part = timestamp_str.split('_')[0:2]  # ['Aug27', '2025']
                month_name = date_part[0][:3]  # 'Aug'
                day = date_part[0][3:]  # '27'
                
                # Convert month name to number
                month_map = {'Jan': '01', 'Feb': '02', 'Mar': '03', 'Apr': '04',
                           'May': '05', 'Jun': '06', 'Jul': '07', 'Aug': '08',
                           'Sep': '09', 'Oct': '10', 'Nov': '11', 'Dec': '12'}
                month = month_map.get(month_name, '01')
            else:
                # Fallback to current date
                now = datetime.now()
                month = f"{now.month:02d}"
                day = f"{now.day:02d}"
                year = str(now.year)
            
            # Generate paths
            date_path = f"{year}/{month}/{day}"
            snapshot_key = f"snapshots/{date_path}/{os.path.basename(filename)}"
            
            return snapshot_key, date_path
            
        except Exception as e:
            logger.warning(f"Error parsing timestamp '{timestamp_str}': {e}")
            # Fallback to current date
            now = datetime.now()
            date_path = f"{now.year}/{now.month:02d}/{now.day:02d}"
            snapshot_key = f"snapshots/{date_path}/{os.path.basename(filename)}"
            return snapshot_key, date_path
    
    def _process_event(self, event_data: Dict[str, Any]) -> None:
        """Process a single event: upload image and buffer log entry"""
        try:
            timestamp = event_data.get('timestamp', '')
            image_path = event_data.get('image', '')
            
            # Upload image immediately if path exists
            if image_path and image_path != 'failed_to_save' and os.path.exists(image_path):
                snapshot_key, date_path = self._get_s3_paths(timestamp, image_path)
                
                if self.s3_manager.upload_image(image_path, snapshot_key):
                    self.stats['images_uploaded'] += 1
                    # Update event data with S3 path
                    event_data['s3_image_path'] = snapshot_key
                else:
                    self.stats['images_failed'] += 1
                    event_data['s3_image_path'] = 'upload_failed'
            else:
                logger.warning(f"Image file not found or invalid: {image_path}")
                event_data['s3_image_path'] = 'file_not_found'
            
            # Add to event buffer for batch processing
            self.event_buffer.add_event(event_data)
            self.stats['events_received'] += 1
            
        except Exception as e:
            logger.error(f"Error processing event: {e}")
            logger.debug(f"Event data: {event_data}")
    
    def _flush_event_buffer(self) -> None:
        """Flush accumulated events to S3 as JSONL file"""
        events = self.event_buffer.get_and_clear_events()
        
        if not events:
            return
        
        try:
            # Use the first event's timestamp to determine the date path
            first_event = events[0]
            timestamp = first_event.get('timestamp', '')
            _, date_path = self._get_s3_paths(timestamp, '')
            
            # Create JSONL content
            jsonl_lines = [json.dumps(event) for event in events]
            jsonl_content = '\n'.join(jsonl_lines) + '\n'
            
            # Generate S3 key for logs
            current_time = datetime.now().strftime("%H%M%S")
            logs_key = f"logs/{date_path}/events_{current_time}.jsonl"
            
            if self.s3_manager.upload_jsonl(jsonl_content, logs_key):
                self.stats['batches_uploaded'] += 1
                logger.info(f"Uploaded batch of {len(events)} events to {logs_key}")
            else:
                self.stats['batches_failed'] += 1
                logger.error(f"Failed to upload batch of {len(events)} events")
                
        except Exception as e:
            logger.error(f"Error flushing event buffer: {e}")
            self.stats['batches_failed'] += 1
    
    def _print_stats(self) -> None:
        """Print service statistics"""
        uptime = time.time() - self.stats['start_time']
        buffer_count = self.event_buffer.get_count()
        
        logger.info(f"=== Service Statistics (Uptime: {uptime:.0f}s) ===")
        logger.info(f"Events received: {self.stats['events_received']}")
        logger.info(f"Images uploaded: {self.stats['images_uploaded']}")
        logger.info(f"Images failed: {self.stats['images_failed']}")
        logger.info(f"Batches uploaded: {self.stats['batches_uploaded']}")
        logger.info(f"Batches failed: {self.stats['batches_failed']}")
        logger.info(f"Events in buffer: {buffer_count}")
        logger.info("=" * 50)
    
    def run(self) -> None:
        """Main service loop"""
        self.running = True
        logger.info("Drowsiness Cloud Service started")
        
        last_stats_time = time.time()
        stats_interval = self.config.get('stats_interval', 300)  # 5 minutes
        
        while self.running:
            try:
                # Try to receive message with timeout
                try:
                    message = self.subscriber.recv_string(zmq.NOBLOCK)
                    
                    # Parse JSON message
                    try:
                        event_data = json.loads(message)
                        self._process_event(event_data)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Invalid JSON received: {e}")
                        continue
                        
                except zmq.Again:
                    # No message received (timeout), continue
                    pass
                
                # Check if we should flush the buffer
                if self.event_buffer.should_flush():
                    self._flush_event_buffer()
                
                # Print stats periodically
                if time.time() - last_stats_time > stats_interval:
                    self._print_stats()
                    last_stats_time = time.time()
                
                # Small sleep to prevent busy waiting
                time.sleep(0.01)
                
            except KeyboardInterrupt:
                logger.info("Received interrupt signal, shutting down...")
                break
            except Exception as e:
                logger.error(f"Unexpected error in main loop: {e}")
                time.sleep(1)  # Wait before retrying
    
    def shutdown(self) -> None:
        """Graceful shutdown"""
        logger.info("Shutting down service...")
        self.running = False
        
        # Flush any remaining events
        if self.event_buffer.get_count() > 0:
            logger.info("Flushing remaining events...")
            self._flush_event_buffer()
        
        # Close ZeroMQ connections
        self.subscriber.close()
        self.context.term()
        
        # Print final stats
        self._print_stats()
        logger.info("Service shutdown complete")


def load_config() -> Dict[str, Any]:
    """Load configuration from environment variables or use defaults"""
    config = {
        # ZeroMQ settings
        'zmq_endpoint': os.getenv('ZMQ_ENDPOINT', 'tcp://localhost:5555'),
        
        # AWS S3 settings
        's3_bucket': os.getenv('S3_BUCKET', 'drowsiness-detection-data'),
        'aws_region': os.getenv('AWS_REGION', 'us-east-1'),
        
        # Service settings
        'batch_flush_interval': int(os.getenv('BATCH_FLUSH_INTERVAL', '60')),
        'stats_interval': int(os.getenv('STATS_INTERVAL', '300')),
    }
    
    # Validate required settings
    if not config['s3_bucket']:
        logger.error("S3_BUCKET environment variable is required")
        sys.exit(1)
    
    return config


def signal_handler(signum, frame):
    """Handle shutdown signals"""
    logger.info(f"Received signal {signum}")
    global service
    if service:
        service.shutdown()
    sys.exit(0)


def main():
    """Main entry point"""
    global service
    
    # Set up signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Load configuration
        config = load_config()
        logger.info(f"Configuration loaded: {config}")
        
        # Create and run service
        service = DrowsinessCloudService(config)
        service.run()
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    service = None
    main()