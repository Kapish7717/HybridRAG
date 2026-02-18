from collections import defaultdict
from datetime import datetime, timedelta
from fastapi import HTTPException

# Track requests per user
request_counts = defaultdict(list)

def check_rate_limit(username: str, max_requests: int = 20, window_minutes: int = 1):
    """
    Limit users to max_requests per window_minutes
    Default: 20 requests per minute
    """
    now = datetime.now()
    window_start = now - timedelta(minutes=window_minutes)
    
    # Clean old requests
    request_counts[username] = [
        req_time for req_time in request_counts[username]
        if req_time > window_start
    ]
    
    # Check limit
    if len(request_counts[username]) >= max_requests:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Max {max_requests} requests per {window_minutes} minute(s). Please wait."
        )
    
    # Record this request
    request_counts[username].append(now)