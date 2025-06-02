#!/usr/bin/env python3
"""
User Behavior Analysis Tools Package
Two-step architecture for analyzing user behavior patterns from device event data
"""

# Import the core analyzer class
from .core.analyzer import UserBehaviorAnalyzer

# Import Step 1 Filter Functions
from .filters.user_filters import (
    analyze_day1_returning_users,
    analyze_users_without_day1_return,
    filter_by_location,
    get_users_by_device_ids
)

# Import Step 2 Analysis Functions
from .analysis import (
    get_event_statistics,
    plot_event_per_user,
    find_frequent_events,
    filter_devices_by_events,
    plot_event_counts,
    get_events_by_popularity,
    get_users_with_search_terms_percentage
)

# Import Utility Functions
from .utils.export_utils import (
    export_results,
    export_filtered_data,
    export_analysis_results,
    extract_device_ids,
    extract_user_data
)

# Import main function
from .main import main

__version__ = "1.0.0"
__author__ = "User Behavior Analysis Team"

__all__ = [
    # Core class
    'UserBehaviorAnalyzer',
    
    # Step 1: Filter Functions
    'analyze_day1_returning_users',
    'analyze_users_without_day1_return',
    'filter_by_location',
    'get_users_by_device_ids',
    
    # Step 2: Analysis Functions
    'get_event_statistics',
    'plot_event_per_user',
    'find_frequent_events',
    'filter_devices_by_events',
    'plot_event_counts',
    'get_events_by_popularity',
    'get_users_with_search_terms_percentage',
    
    # Utility Functions
    'export_results',
    'export_filtered_data',
    'export_analysis_results',
    'extract_device_ids',
    'extract_user_data',
    
    # Main function
    'main'
] 