#!/usr/bin/env python3
"""
User Filter Functions for User Behavior Analysis Tools
STEP 1 functions that take original database as input and return filtered data dictionaries
"""

from typing import Dict, List
import sys
import os

# Add the parent directory to sys.path so we can import from core
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from core.analyzer import UserBehaviorAnalyzer

def analyze_day1_returning_users(db_path: str) -> Dict:
    """
    FILTER FUNCTION: Analyze users who returned on day 2 from original database
    Returns filtered data dictionary that can be used by analysis functions
    
    Args:
        db_path: Path to the SQLite database file
        
    Returns:
        Dictionary of user data for users who returned on day 2
    """
    analyzer = UserBehaviorAnalyzer(db_path)
    standardized_result = analyzer.get_day1_actions_of_returning_users()
    
    # Extract the actual user data for use in analysis functions
    return standardized_result['results']

def analyze_users_without_day1_return(db_path: str) -> Dict:
    """
    FILTER FUNCTION: Analyze users who didn't return on day 2 from original database
    Returns filtered data dictionary that can be used by analysis functions
    
    Args:
        db_path: Path to the SQLite database file
        
    Returns:
        Dictionary of user data for users who didn't return on day 2
    """
    analyzer = UserBehaviorAnalyzer(db_path)
    standardized_result = analyzer.get_users_without_day1_return()
    
    # Extract the actual user data for use in analysis functions
    return standardized_result['results']

def filter_by_location(db_path: str, country: str = None, timezone: str = None) -> Dict:
    """
    FILTER FUNCTION: Filter users by location from original database
    Returns filtered data dictionary that can be used by analysis functions
    
    Args:
        db_path: Path to the SQLite database file
        country: Country name to filter by (e.g., "United States")
        timezone: Timezone to filter by (e.g., "America/New_York")
        
    Returns:
        Dictionary of user data for users matching location criteria
    """
    analyzer = UserBehaviorAnalyzer(db_path)
    standardized_result = analyzer.filter_users_by_locationORtimezone(country=country, timezone=timezone)
    
    # Extract the actual user data for use in analysis functions
    # Convert list of records to dictionary format for consistency
    filtered_data = {}
    for record in standardized_result['results']:
        device_id = record['device_id']
        filtered_data[device_id] = record
    
    return filtered_data

def get_users_by_device_ids(db_path: str, device_ids: List[str]) -> Dict:
    """
    FILTER FUNCTION: Get users by device IDs from original database
    Returns filtered data dictionary that can be used by analysis functions
    
    Args:
        db_path: Path to the SQLite database file
        device_ids: List of device IDs to retrieve information for
        
    Returns:
        Dictionary of user data for specified device IDs
    """
    analyzer = UserBehaviorAnalyzer(db_path)
    standardized_result = analyzer.get_user_info_by_device_list(device_ids)
    
    # Extract the actual user data for use in analysis functions
    return standardized_result['results'] 