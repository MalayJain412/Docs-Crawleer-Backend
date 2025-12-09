#!/usr/bin/env python3
"""Test script to verify task status formatting improvements."""

import time
import json


def simulate_task_states():
    """Simulate different task states to test the formatting."""
    
    # Test case 1: Started task with all fields
    started_task = {
        "task_id": "crawl_1731955616000",
        "type": "crawl",
        "status": "started",
        "message": "Initializing crawl for https://example.com",
        "progress": 0,
        "domain": "example-com",
        "start_time": time.time(),
        "url": "https://example.com",
        "end_time": None,
        "error": None
    }
    
    # Test case 2: Crawling task with progress
    crawling_task = {
        "task_id": "crawl_1731955616000",
        "type": "crawl",
        "status": "crawling",
        "message": "Processing pages... (5 completed)",
        "progress": 45,
        "domain": "example-com",
        "start_time": time.time() - 30,
        "url": "https://example.com",
        "end_time": None,
        "error": None
    }
    
    # Test case 3: Completed task
    completed_task = {
        "task_id": "crawl_1731955616000",
        "type": "crawl",
        "status": "completed",
        "message": "Successfully crawled 25 pages",
        "progress": 100,
        "domain": "example-com",
        "start_time": time.time() - 120,
        "url": "https://example.com",
        "end_time": time.time(),
        "error": None,
        "results": {
            "total_pages": 25,
            "successful_pages": 25,
            "failed_pages": 0,
            "documents_saved": 25
        }
    }
    
    # Test case 4: Failed task
    failed_task = {
        "task_id": "crawl_1731955616001",
        "type": "crawl",
        "status": "failed",
        "message": "Crawling failed: Connection timeout",
        "progress": 0,
        "domain": "bad-example-com",
        "start_time": time.time() - 60,
        "url": "https://bad-example.com",
        "end_time": time.time() - 30,
        "error": "Connection timeout after 30 seconds"
    }
    
    # Test case 5: Legacy task without task_id or message (the problematic case)
    legacy_task = {
        "type": "embed",
        "status": "processing",
        "domain": "old-domain",
        "start_time": time.time() - 45
    }
    
    return [started_task, crawling_task, completed_task, failed_task, legacy_task]


def test_javascript_formatting_logic():
    """Test the JavaScript formatting logic equivalent in Python."""
    
    def format_time(timestamp):
        if not timestamp or timestamp == 0:
            return 'N/A'
        # Handle both Unix timestamp and ISO string formats
        if isinstance(timestamp, (int, float)):
            date = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(timestamp))
        else:
            date = str(timestamp)
        return date

    def calculate_duration(start, end):
        if not start or not end:
            return 'N/A'
        duration = end - start
        return f"{int(duration)}s" if duration > 0 else 'N/A'

    def format_task_status(status):
        return {
            'id': status.get('task_id') or status.get('id') or 'Unknown ID',
            'status': status.get('status') or 'unknown',
            'progress': status.get('progress') or 0,
            'message': status.get('message') or status.get('result') or f"{status.get('status', 'Processing')}...",
            'startTime': format_time(status.get('start_time')),
            'endTime': format_time(status.get('end_time')),
            'duration': status.get('duration') or calculate_duration(status.get('start_time'), status.get('end_time')),
            'type': status.get('type') or 'task'
        }
    
    tasks = simulate_task_states()
    
    print("=== Task Status Formatting Test ===\n")
    
    for i, task in enumerate(tasks, 1):
        print(f"Test Case {i}: {task.get('status', 'unknown').upper()} Task")
        print(f"Original: {json.dumps(task, indent=2)}")
        
        formatted = format_task_status(task)
        print(f"Formatted: {json.dumps(formatted, indent=2)}")
        
        # Check for the original issues
        issues_found = []
        if formatted['id'] == 'Unknown ID':
            issues_found.append("❌ Missing task ID")
        else:
            print("✅ Task ID present")
            
        if formatted['message'] in ['No message', 'Processing...']:
            issues_found.append("❌ Generic/missing message")
        else:
            print("✅ Meaningful message present")
            
        if issues_found:
            print("Issues found:", ", ".join(issues_found))
        else:
            print("✅ All checks passed")
            
        print("-" * 50)


if __name__ == "__main__":
    test_javascript_formatting_logic()