#!/usr/bin/env python3
"""
User Behavior Analysis Tools Library - Backward Compatibility Module

This file maintains backward compatibility with existing code by importing all functions
from the new modular structure.

🏗️  NEW MODULAR STRUCTURE:
- Tools/core/analyzer.py - UserBehaviorAnalyzer class
- Tools/filters/user_filters.py - Step 1 filter functions  
- Tools/analysis/event_analysis.py - Step 2 analysis functions
- Tools/utils/export_utils.py - Export and utility functions
- Tools/main.py - Main entry point

All original functionality is preserved and available through the same imports:
>>> from Tools.Check_Data import *
"""

import sys
import os

# Add current directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Import required modules that were available in the original file
import pandas as pd
import sqlite3
import json
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Core analyzer class
from core.analyzer import UserBehaviorAnalyzer

# Step 1: Filter Functions
from filters.user_filters import (
    analyze_day1_returning_users,
    analyze_users_without_day1_return,
    filter_by_location,
    get_users_by_device_ids
)

# Step 2: Analysis Functions
from analysis.event_analysis import (
    get_event_statistics,
    get_events_by_popularity,
    plot_event_per_user,
    find_frequent_events,
    filter_devices_by_events,
    plot_event_counts,
    get_users_with_search_terms_percentage
)

# Utility and Export Functions
from utils.export_utils import (
    export_results,
    export_filtered_data,
    export_analysis_results,
    extract_device_ids,
    extract_user_data
)

# Main function
from main import main

# Explicitly add all functions to globals() to ensure they're available
globals().update({
    'UserBehaviorAnalyzer': UserBehaviorAnalyzer,
    'analyze_day1_returning_users': analyze_day1_returning_users,
    'analyze_users_without_day1_return': analyze_users_without_day1_return,
    'filter_by_location': filter_by_location,
    'get_users_by_device_ids': get_users_by_device_ids,
    'get_event_statistics': get_event_statistics,
    'get_events_by_popularity': get_events_by_popularity,
    'plot_event_per_user': plot_event_per_user,
    'find_frequent_events': find_frequent_events,
    'filter_devices_by_events': filter_devices_by_events,
    'plot_event_counts': plot_event_counts,
    'export_results': export_results,
    'export_filtered_data': export_filtered_data,
    'export_analysis_results': export_analysis_results,
    'extract_device_ids': extract_device_ids,
    'extract_user_data': extract_user_data,
    'main': main
})

# Version and metadata
__version__ = "2.0.0"
__status__ = "Modular"

# Export all public functions for backwards compatibility
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
    'get_events_by_popularity',
    'plot_event_per_user',
    'find_frequent_events',
    'filter_devices_by_events',
    'plot_event_counts',
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

# Print reorganization info when imported (but not when run as main)
if __name__ != "__main__":
    print("📦 User Behavior Analysis Tools - Modular Structure")
    print("✅ Backward compatibility maintained - all functions available")
    print("🔧 Code separated into: core/, filters/, analysis/, utils/")

if __name__ == "__main__":
    main() 

