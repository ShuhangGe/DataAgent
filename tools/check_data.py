#!/usr/bin/env python3
"""
User Behavior Analysis Tools Library
Provides functions for analyzing user behavior patterns from device event data
"""

import pandas as pd
import sqlite3
import json
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import os
import argparse

class UserBehaviorAnalyzer:
    """Tools library for analyzing user behavior from device event dictionaries"""
    
    def __init__(self, db_path: str):
        """Initialize the analyzer with database path
        
        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = db_path
        self.df = None
        self._load_data()
    
    def _load_data(self):
        """Load data from database"""
        if not os.path.exists(self.db_path):
            raise FileNotFoundError(f"Database file not found: {self.db_path}")
        
        try:
            conn = sqlite3.connect(self.db_path)
            self.df = pd.read_sql_query("SELECT * FROM device_event_dictionaries", conn)
            conn.close()
            print(f"✅ Loaded {len(self.df)} device records from database")
        except Exception as e:
            raise Exception(f"Error loading data: {e}")
    
    def get_day1_actions_of_returning_users(self) -> Dict[str, List[Dict]]:
        """
        Identify what actions were taken on the first day by users who returned 
        to open the app the next day (day 1 return - logged in on two consecutive days)
        
        Returns:
            Dictionary mapping device_id to their day 1 actions for users who returned day 2
        """
        print("🔍 Analyzing day 1 actions of users who returned on day 2...")
        
        returning_users_day1_actions = {}
        
        for _, row in self.df.iterrows():
            try:
                device_id = row['device_id']
                event_pairs = json.loads(row['event_time_pairs'])
                
                if len(event_pairs) < 2:
                    continue  # Need at least 2 events to check consecutive days
                
                # Group events by date
                events_by_date = {}
                for event in event_pairs:
                    event_date = datetime.fromisoformat(event['timestamp'].replace('Z', '+00:00')).date()
                    if event_date not in events_by_date:
                        events_by_date[event_date] = []
                    events_by_date[event_date].append(event)
                
                # Sort dates
                sorted_dates = sorted(events_by_date.keys())
                
                if len(sorted_dates) < 2:
                    continue  # Need events on at least 2 different days
                
                # Check for consecutive days starting from first day
                first_day = sorted_dates[0]
                second_day = first_day + timedelta(days=1)
                
                # Check if user had activity on consecutive days
                if second_day in events_by_date:
                    # This user returned the next day - get their day 1 actions
                    day1_actions = events_by_date[first_day]
                    
                    # Create event sequence list (just event names in order)
                    event_sequence = [action['event'] for action in day1_actions]
                    
                    returning_users_day1_actions[device_id] = {
                        'first_day': first_day.isoformat(),
                        'second_day': second_day.isoformat(),
                        'day1_actions': day1_actions,
                        'day1_action_count': len(day1_actions),
                        'day1_unique_events': list(set([action['event'] for action in day1_actions])),
                        'event_sequence': event_sequence,  # New: just the sequence of events
                        'timezone': row['timezone'],
                        'country': row['country']
                    }
                
            except (json.JSONDecodeError, ValueError, KeyError) as e:
                print(f"⚠️  Error processing device {row.get('device_id', 'unknown')}: {e}")
                continue
        
        print(f"✅ Found {len(returning_users_day1_actions)} users who returned on day 2")
        return returning_users_day1_actions
    
    def get_users_without_day1_return(self) -> Dict[str, Dict]:
        """
        Identify users who don't have day 1 return (didn't come back the next day)
        
        Returns:
            Dictionary mapping device_id to their information for users who didn't return day 2
        """
        print("🔍 Analyzing users who didn't return on day 2...")
        
        non_returning_users = {}
        
        for _, row in self.df.iterrows():
            try:
                device_id = row['device_id']
                event_pairs = json.loads(row['event_time_pairs'])
                
                if len(event_pairs) == 0:
                    continue  # Skip users with no events
                
                # Group events by date
                events_by_date = {}
                for event in event_pairs:
                    event_date = datetime.fromisoformat(event['timestamp'].replace('Z', '+00:00')).date()
                    if event_date not in events_by_date:
                        events_by_date[event_date] = []
                    events_by_date[event_date].append(event)
                
                # Sort dates
                sorted_dates = sorted(events_by_date.keys())
                first_day = sorted_dates[0]
                second_day = first_day + timedelta(days=1)
                
                # Check if user did NOT have activity on the next day
                if second_day not in events_by_date:
                    # This user did NOT return the next day
                    day1_actions = events_by_date[first_day]
                    
                    # Check if they had any activity after day 1 (but not day 2)
                    later_activity = any(date > second_day for date in sorted_dates)
                    
                    # Create event sequence list (just event names in order)
                    event_sequence = [action['event'] for action in day1_actions]
                    
                    non_returning_users[device_id] = {
                        'first_day': first_day.isoformat(),
                        'expected_return_day': second_day.isoformat(),
                        'day1_actions': day1_actions,
                        'day1_action_count': len(day1_actions),
                        'day1_unique_events': list(set([action['event'] for action in day1_actions])),
                        'event_sequence': event_sequence,  # New: just the sequence of events
                        'total_active_days': len(sorted_dates),
                        'last_activity_day': sorted_dates[-1].isoformat(),
                        'had_later_activity': later_activity,
                        'days_between_first_last': (sorted_dates[-1] - first_day).days,
                        'timezone': row['timezone'],
                        'country': row['country'],
                        'total_events': row['total_events']
                    }
                
            except (json.JSONDecodeError, ValueError, KeyError) as e:
                print(f"⚠️  Error processing device {row.get('device_id', 'unknown')}: {e}")
                continue
        
        print(f"✅ Found {len(non_returning_users)} users who didn't return on day 2")
        return non_returning_users
    
    def filter_users_by_location(self, country: Optional[str] = None, timezone: Optional[str] = None) -> pd.DataFrame:
        """
        Filter users based on location (country and/or timezone)
        
        Args:
            country: Country name to filter by (e.g., "United States")
            timezone: Timezone to filter by (e.g., "America/New_York")
            
        Returns:
            Filtered DataFrame of users matching location criteria
        """
        print(f"🌍 Filtering users by location - Country: {country}, Timezone: {timezone}")
        
        filtered_df = self.df.copy()
        
        if country:
            filtered_df = filtered_df[filtered_df['country'] == country]
            print(f"   After country filter: {len(filtered_df)} users")
        
        if timezone:
            filtered_df = filtered_df[filtered_df['timezone'] == timezone]
            print(f"   After timezone filter: {len(filtered_df)} users")
        
        print(f"✅ Final filtered result: {len(filtered_df)} users")
        return filtered_df
    
    def get_user_info_by_device_list(self, device_ids: List[str], include_event_details: bool = True) -> Dict[str, Dict]:
        """
        Get all user information for a given list of device_ids
        
        Args:
            device_ids: List of device IDs to retrieve information for
            include_event_details: Whether to include parsed event_time_pairs details
            
        Returns:
            Dictionary mapping device_id to complete user information
        """
        print(f"📋 Getting user information for {len(device_ids)} devices...")
        
        user_info = {}
        found_devices = []
        missing_devices = []
        
        for device_id in device_ids:
            device_row = self.df[self.df['device_id'] == device_id]
            
            if device_row.empty:
                missing_devices.append(device_id)
                continue
            
            found_devices.append(device_id)
            row = device_row.iloc[0]
            
            # Basic user information
            user_data = {
                'device_id': device_id,
                'event': row['event'],
                'timestamp': row['timestamp'],
                'uuid': row['uuid'],
                'distinct_id': row['distinct_id'],
                'country': row['country'],
                'timezone': row['timezone'],
                'newDevice': row['newDevice'],
                'total_events': row['total_events'],
                'first_event_time': row['first_event_time'],
                'last_event_time': row['last_event_time'],
                'time_span_hours': row['time_span_hours']
            }
            
            if include_event_details:
                try:
                    # Parse event_time_pairs and event_types
                    event_pairs = json.loads(row['event_time_pairs'])
                    event_types = json.loads(row['event_types'])
                    
                    user_data.update({
                        'event_time_pairs': event_pairs,
                        'event_types': event_types,
                        'unique_event_count': len(event_types),
                        'events_by_hour': self._analyze_events_by_hour(event_pairs),
                        'events_by_day': self._analyze_events_by_day(event_pairs)
                    })
                    
                except json.JSONDecodeError as e:
                    print(f"⚠️  Error parsing JSON for device {device_id}: {e}")
                    user_data['event_time_pairs'] = []
                    user_data['event_types'] = []
            
            user_info[device_id] = user_data
        
        print(f"✅ Found information for {len(found_devices)} devices")
        if missing_devices:
            print(f"⚠️  Missing devices: {missing_devices}")
        
        return user_info
    
    def _analyze_events_by_hour(self, event_pairs: List[Dict]) -> Dict[int, int]:
        """Analyze event distribution by hour of day"""
        hour_counts = {}
        for event in event_pairs:
            hour = event.get('hour', 0)
            hour_counts[hour] = hour_counts.get(hour, 0) + 1
        return hour_counts
    
    def _analyze_events_by_day(self, event_pairs: List[Dict]) -> Dict[str, int]:
        """Analyze event distribution by day of week"""
        day_counts = {}
        for event in event_pairs:
            day = event.get('day_of_week', 'Unknown')
            day_counts[day] = day_counts.get(day, 0) + 1
        return day_counts
    
    def filter_by_event_count_range(self, num_start: int, num_end: int) -> Dict[str, int]:
        """
        Filter devices by event count range (original FilterNumpairs functionality)
        
        Args:
            num_start: Start index for sorted devices
            num_end: End index for sorted devices
            
        Returns:
            Dictionary mapping device_id to event count for devices in range
        """
        print(f"🔢 Filtering devices by event count range: {num_start} to {num_end}")
        
        counter = {}
        for _, row in self.df.iterrows():
            try:
                event_pairs = json.loads(row['event_time_pairs'])
                counter[row['device_id']] = len(event_pairs)
            except json.JSONDecodeError:
                counter[row['device_id']] = 0
        
        # Sort by event count and get range
        sorted_devices = sorted(counter.items(), key=lambda item: item[1])
        range_devices = dict(sorted_devices[num_start:num_end])
        
        print(f"✅ Found {len(range_devices)} devices in range")
        return range_devices
    
    def get_event_user_counts(self) -> Dict[str, Dict]:
        """
        Get the number of users for each event type and time pairs
        
        Returns:
            Dictionary with event statistics including user counts and time distribution
        """
        print("📊 Analyzing event user counts and time patterns...")
        
        event_stats = {}
        
        for _, row in self.df.iterrows():
            try:
                device_id = row['device_id']
                event_pairs = json.loads(row['event_time_pairs'])
                
                for event_pair in event_pairs:
                    event_name = event_pair['event']
                    timestamp = event_pair['timestamp']
                    hour = event_pair.get('hour', 0)
                    day_of_week = event_pair.get('day_of_week', 'Unknown')
                    
                    # Initialize event stats if not exists
                    if event_name not in event_stats:
                        event_stats[event_name] = {
                            'total_occurrences': 0,
                            'unique_users': set(),
                            'hours_distribution': {},
                            'days_distribution': {},
                            'user_event_counts': {}
                        }
                    
                    # Update statistics
                    event_stats[event_name]['total_occurrences'] += 1
                    event_stats[event_name]['unique_users'].add(device_id)
                    
                    # Hour distribution
                    if hour not in event_stats[event_name]['hours_distribution']:
                        event_stats[event_name]['hours_distribution'][hour] = 0
                    event_stats[event_name]['hours_distribution'][hour] += 1
                    
                    # Day distribution
                    if day_of_week not in event_stats[event_name]['days_distribution']:
                        event_stats[event_name]['days_distribution'][day_of_week] = 0
                    event_stats[event_name]['days_distribution'][day_of_week] += 1
                    
                    # User event counts
                    if device_id not in event_stats[event_name]['user_event_counts']:
                        event_stats[event_name]['user_event_counts'][device_id] = 0
                    event_stats[event_name]['user_event_counts'][device_id] += 1
                    
            except (json.JSONDecodeError, KeyError) as e:
                print(f"⚠️  Error processing device {row.get('device_id', 'unknown')}: {e}")
                continue
        
        # Convert sets to counts and clean up data
        for event_name in event_stats:
            event_stats[event_name]['unique_user_count'] = len(event_stats[event_name]['unique_users'])
            event_stats[event_name]['unique_users'] = list(event_stats[event_name]['unique_users'])  # Convert set to list
            event_stats[event_name]['avg_events_per_user'] = (
                event_stats[event_name]['total_occurrences'] / 
                event_stats[event_name]['unique_user_count'] if event_stats[event_name]['unique_user_count'] > 0 else 0
            )
        
        print(f"✅ Analyzed {len(event_stats)} different event types")
        return event_stats
    
    def get_events_sorted_by_user_count(self, ascending: bool = False) -> List[Tuple[str, Dict]]:
        """
        Get events sorted by number of unique users
        
        Args:
            ascending: If True, sort from least to most users. If False, sort from most to least users.
            
        Returns:
            List of tuples (event_name, event_stats) sorted by unique user count
        """
        print(f"📈 Sorting events by user count (ascending={ascending})...")
        
        event_stats = self.get_event_user_counts()
        
        # Sort by unique user count
        sorted_events = sorted(
            event_stats.items(), 
            key=lambda x: x[1]['unique_user_count'], 
            reverse=not ascending
        )
        
        print(f"✅ Sorted {len(sorted_events)} events by user count")
        return sorted_events
    
    def filter_devices_by_event_count(self, min_events: int = 1, max_events: int = None) -> pd.DataFrame:
        """
        Filter devices by total event count
        
        Args:
            min_events: Minimum number of events (inclusive)
            max_events: Maximum number of events (inclusive), None for no upper limit
            
        Returns:
            Filtered DataFrame of devices
        """
        print(f"🎯 Filtering devices by event count: min={min_events}, max={max_events}")
        
        filtered_df = self.df[self.df['total_events'] >= min_events].copy()
        
        if max_events is not None:
            filtered_df = filtered_df[filtered_df['total_events'] <= max_events]
        
        filtered_df = filtered_df.sort_values('total_events', ascending=False)
        
        print(f"✅ Found {len(filtered_df)} devices matching event count criteria")
        return filtered_df
    
    def filter_devices_by_specific_event(self, event_name: str, min_occurrences: int = 1) -> List[str]:
        """
        Filter devices that performed a specific event a minimum number of times
        
        Args:
            event_name: Name of the event to filter by
            min_occurrences: Minimum number of times the event must occur
            
        Returns:
            List of device_ids that meet the criteria
        """
        print(f"🔍 Filtering devices by specific event: '{event_name}' (min {min_occurrences} times)")
        
        matching_devices = []
        
        for _, row in self.df.iterrows():
            try:
                device_id = row['device_id']
                event_pairs = json.loads(row['event_time_pairs'])
                
                # Count occurrences of the specific event
                event_count = sum(1 for event_pair in event_pairs if event_pair['event'] == event_name)
                
                if event_count >= min_occurrences:
                    matching_devices.append(device_id)
                    
            except (json.JSONDecodeError, KeyError) as e:
                print(f"⚠️  Error processing device {row.get('device_id', 'unknown')}: {e}")
                continue
        
        print(f"✅ Found {len(matching_devices)} devices with '{event_name}' >= {min_occurrences} times")
        return matching_devices
    
    def export_analysis_results(self, data: Dict, filename: str, format_type: str = 'json'):
        """
        Export analysis results to file with custom formatting for event sequences
        
        Args:
            data: Data to export
            filename: Output filename
            format_type: 'json' or 'csv'
        """
        if format_type == 'json':
            # Custom JSON formatting to keep event_sequence on single lines
            def format_json_with_compact_arrays(obj, indent=0):
                """Custom JSON formatter that keeps arrays on single lines"""
                if isinstance(obj, dict):
                    items = []
                    for key, value in obj.items():
                        if key == 'event_sequence' and isinstance(value, list):
                            # Format event_sequence on a single line
                            formatted_value = json.dumps(value, ensure_ascii=False)
                        elif isinstance(value, (dict, list)) and key != 'event_sequence':
                            formatted_value = format_json_with_compact_arrays(value, indent + 2)
                        else:
                            formatted_value = json.dumps(value, ensure_ascii=False, default=str)
                        
                        items.append(f'{"  " * (indent + 1)}"{key}": {formatted_value}')
                    
                    if items:
                        return "{\n" + ",\n".join(items) + f"\n{'  ' * indent}}}"
                    else:
                        return "{}"
                        
                elif isinstance(obj, list):
                    if not obj:
                        return "[]"
                    # For non-event_sequence lists, format normally
                    items = [format_json_with_compact_arrays(item, indent + 2) for item in obj]
                    return "[\n" + ",\n".join(f"{'  ' * (indent + 1)}{item}" for item in items) + f"\n{'  ' * indent}]"
                else:
                    return json.dumps(obj, ensure_ascii=False, default=str)
            
            formatted_json = format_json_with_compact_arrays(data)
            
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(formatted_json)
                
        elif format_type == 'csv' and isinstance(data, dict):
            # Convert dict to DataFrame for CSV export
            df = pd.DataFrame.from_dict(data, orient='index')
            df.to_csv(filename, encoding='utf-8')
        
        print(f"✅ Exported results to {filename}")

# Convenience functions for direct usage
def analyze_day1_returning_users(db_path: str) -> Dict[str, List[Dict]]:
    """Convenience function to analyze day 1 actions of returning users"""
    analyzer = UserBehaviorAnalyzer(db_path)
    return analyzer.get_day1_actions_of_returning_users()

def analyze_users_without_day1_return(db_path: str) -> Dict[str, Dict]:
    """Convenience function to analyze users who didn't return on day 2"""
    analyzer = UserBehaviorAnalyzer(db_path)
    return analyzer.get_users_without_day1_return()

def filter_by_location(db_path: str, country: str = None, timezone: str = None) -> pd.DataFrame:
    """Convenience function to filter users by location"""
    analyzer = UserBehaviorAnalyzer(db_path)
    return analyzer.filter_users_by_location(country, timezone)

def get_users_by_device_ids(db_path: str, device_ids: List[str]) -> Dict[str, Dict]:
    """Convenience function to get user info by device ID list"""
    analyzer = UserBehaviorAnalyzer(db_path)
    return analyzer.get_user_info_by_device_list(device_ids)

def get_event_statistics(db_path: str) -> Dict[str, Dict]:
    """Convenience function to get event user counts and statistics"""
    analyzer = UserBehaviorAnalyzer(db_path)
    return analyzer.get_event_user_counts()

def get_events_by_popularity(db_path: str, ascending: bool = False) -> List[Tuple[str, Dict]]:
    """Convenience function to get events sorted by user count"""
    analyzer = UserBehaviorAnalyzer(db_path)
    return analyzer.get_events_sorted_by_user_count(ascending)

def filter_devices_by_events(db_path: str, min_events: int = 1, max_events: int = None) -> pd.DataFrame:
    """Convenience function to filter devices by event count"""
    analyzer = UserBehaviorAnalyzer(db_path)
    return analyzer.filter_devices_by_event_count(min_events, max_events)

# Main function for testing
def main(args=None):
    """Main function for testing the tools library
    
    Args:
        args: Parsed command line arguments object with db_path and test_function attributes
    """
    # If no args provided, create default args
    if args is None:
        class DefaultArgs:
            def __init__(self):
                self.db_path = "/Users/shuhangge/Desktop/my_projects/Sekai/DataAgent/DataProcess/event_analysis_fake.db"
                self.test_function = ['all']
        args = DefaultArgs()
    
    db_path = args.db_path
    test_functions = args.test_function if isinstance(args.test_function, list) else [args.test_function]
    
    # If 'all' is specified, include all test functions
    if 'all' in test_functions:
        test_functions = ['returning', 'non_returning', 'location', 'device_info', 'event_stats', 'event_filter']
    
    print(f"📁 Using database: {db_path}")
    print(f"🧪 Running tests: {', '.join(test_functions)}")
    
    try:
        # Initialize analyzer
        analyzer = UserBehaviorAnalyzer(db_path)
        
        print("\n" + "="*60)
        print("🧪 TESTING USER BEHAVIOR ANALYSIS TOOLS")
        print("="*60)
        
        # Initialize variables for cross-function usage
        returning_users = None
        non_returning_users = None
        event_stats = None
        sorted_events = None
        
        # Test 1: Day 1 actions of returning users
        if 'returning' in test_functions:
            print("\n1️⃣ Testing: Day 1 actions of returning users")
            returning_users = analyzer.get_day1_actions_of_returning_users()
            
            if returning_users:
                # Show sample results
                sample_device = list(returning_users.keys())[0]
                sample_data = returning_users[sample_device]
                print(f"\nSample returning user: {sample_device}")
                print(f"   First day: {sample_data['first_day']}")
                print(f"   Day 1 actions: {sample_data['day1_unique_events']}")
                print(f"   Action count: {sample_data['day1_action_count']}")
                print(f"   Event sequence: {sample_data['event_sequence']}")
            
            # Export results
            analyzer.export_analysis_results(returning_users, "./returning_users_analysis.json")
        
        # Test 1b: Users without day 1 return
        if 'non_returning' in test_functions:
            print("\n1️⃣b Testing: Users without day 1 return")
            non_returning_users = analyzer.get_users_without_day1_return()
            
            if non_returning_users:
                # Show sample results
                sample_device = list(non_returning_users.keys())[0]
                sample_data = non_returning_users[sample_device]
                print(f"\nSample non-returning user: {sample_device}")
                print(f"   First day: {sample_data['first_day']}")
                print(f"   Expected return day: {sample_data['expected_return_day']}")
                print(f"   Day 1 actions: {sample_data['day1_unique_events']}")
                print(f"   Event sequence: {sample_data['event_sequence']}")
                print(f"   Had later activity: {sample_data['had_later_activity']}")
                print(f"   Total active days: {sample_data['total_active_days']}")
            
            # Export results
            analyzer.export_analysis_results(non_returning_users, "./non_returning_users_analysis.json")
        
        # Test 2: Filter by location
        if 'location' in test_functions:
            print("\n2️⃣ Testing: Filter by location")
            us_users = analyzer.filter_users_by_location(country="United States")
            print(f"Sample US users: {us_users['device_id'].head().tolist()}")
        
        # Test 3: Get user info by device list
        if 'device_info' in test_functions:
            print("\n3️⃣ Testing: Get user info by device list")
            # Get some sample devices for testing
            sample_devices = analyzer.df['device_id'].head(3).tolist()
            user_info = analyzer.get_user_info_by_device_list(sample_devices)
            
            for device_id, info in user_info.items():
                print(f"\nDevice: {device_id}")
                print(f"   Country: {info['country']}")
                print(f"   Total events: {info['total_events']}")
                print(f"   Unique events: {info.get('unique_event_count', 'N/A')}")
        
        # Test 4: Event statistics and sorting
        if 'event_stats' in test_functions:
            print("\n4️⃣ Testing: Event statistics and user counts")
            
            # Get event statistics
            event_stats = analyzer.get_event_user_counts()
            print(f"Found {len(event_stats)} different event types")
            
            # Show top 5 events by user count
            sorted_events = analyzer.get_events_sorted_by_user_count(ascending=False)
            print(f"\nTop 5 events by user count:")
            for i, (event_name, stats) in enumerate(sorted_events[:5], 1):
                print(f"   {i}. {event_name}: {stats['unique_user_count']} users, {stats['total_occurrences']} total occurrences")
                print(f"      Avg per user: {stats['avg_events_per_user']:.1f}")
            
            # Export event statistics
            analyzer.export_analysis_results(event_stats, "./event_statistics.json")
        
        # Test 5: Filter devices by event count
        if 'event_filter' in test_functions:
            print("\n5️⃣ Testing: Filter devices by event count")
            
            # Get event stats if not already loaded
            if event_stats is None:
                event_stats = analyzer.get_event_user_counts()
                sorted_events = analyzer.get_events_sorted_by_user_count(ascending=False)
            
            # Filter highly active devices (10+ events)
            high_activity = analyzer.filter_devices_by_event_count(min_events=10)
            print(f"Highly active devices (10+ events): {len(high_activity)}")
            if len(high_activity) > 0:
                print(f"Sample: {high_activity['device_id'].head(3).tolist()}")
            
            # Filter moderate activity devices (2-9 events)
            moderate_activity = analyzer.filter_devices_by_event_count(min_events=2, max_events=9)
            print(f"Moderate activity devices (2-9 events): {len(moderate_activity)}")
            
            # Test specific event filtering
            if len(event_stats) > 0:
                # Get the most common event
                most_common_event = sorted_events[0][0] if sorted_events else None
                if most_common_event:
                    specific_event_users = analyzer.filter_devices_by_specific_event(most_common_event, min_occurrences=2)
                    print(f"Devices with '{most_common_event}' >= 2 times: {len(specific_event_users)}")
        
        # Cross-analysis: Combine returning users with event filtering
        if 'returning' in test_functions and 'event_filter' in test_functions:
            print("\n🔗 Cross-Analysis: Returning users + Event filtering")
            if returning_users and event_stats:
                returning_device_ids = list(returning_users.keys())
                
                # Analyze event patterns of returning users
                returning_event_counts = {}
                for device_id in returning_device_ids:
                    device_row = analyzer.df[analyzer.df['device_id'] == device_id]
                    if not device_row.empty:
                        total_events = device_row.iloc[0]['total_events']
                        returning_event_counts[device_id] = total_events
                
                if returning_event_counts:
                    avg_events_returning = sum(returning_event_counts.values()) / len(returning_event_counts)
                    print(f"   Returning users average events: {avg_events_returning:.1f}")
                    
                    # High activity returning users
                    high_activity_returning = [device_id for device_id, count in returning_event_counts.items() if count >= 10]
                    print(f"   High activity returning users (10+ events): {len(high_activity_returning)}")
        
        # Cross-analysis: Non-returning users with event filtering
        if 'non_returning' in test_functions and 'event_filter' in test_functions:
            print("\n🔗 Cross-Analysis: Non-returning users + Event filtering")
            if non_returning_users:
                non_returning_device_ids = list(non_returning_users.keys())
                
                # Analyze event patterns of non-returning users
                non_returning_event_counts = {}
                for device_id in non_returning_device_ids:
                    device_row = analyzer.df[analyzer.df['device_id'] == device_id]
                    if not device_row.empty:
                        total_events = device_row.iloc[0]['total_events']
                        non_returning_event_counts[device_id] = total_events
                
                if non_returning_event_counts:
                    avg_events_non_returning = sum(non_returning_event_counts.values()) / len(non_returning_event_counts)
                    print(f"   Non-returning users average events: {avg_events_non_returning:.1f}")
                    
                    # Compare with returning users if available
                    if 'returning' in test_functions and returning_users:
                        returning_device_ids = list(returning_users.keys())
                        returning_event_counts = {}
                        for device_id in returning_device_ids:
                            device_row = analyzer.df[analyzer.df['device_id'] == device_id]
                            if not device_row.empty:
                                total_events = device_row.iloc[0]['total_events']
                                returning_event_counts[device_id] = total_events
                        
                        if returning_event_counts:
                            avg_events_returning = sum(returning_event_counts.values()) / len(returning_event_counts)
                            print(f"   📊 Comparison:")
                            print(f"      Returning users avg events: {avg_events_returning:.1f}")
                            print(f"      Non-returning users avg events: {avg_events_non_returning:.1f}")
                            print(f"      Difference: {avg_events_returning - avg_events_non_returning:.1f}")
        
        print(f"\n✅ All selected tests completed successfully!")
        print(f"📋 Tests run: {', '.join(test_functions)}")
        
    except Exception as e:
        print(f"❌ Error in testing: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='User Behavior Analysis Tools')
    parser.add_argument('--db_path', '-d', 
                        default="/Users/shuhangge/Desktop/my_projects/Sekai/DataAgent/DataProcess/event_analysis_fake.db",
                        help='Path to the SQLite database file')
    parser.add_argument('--test_function', '-t', 
                        nargs='+',  # Allow multiple values
                        choices=['all', 'returning', 'non_returning', 'location', 'device_info', 'event_stats', 'event_filter'],
                        default=['all'],
                        help='Which functions to test (can specify multiple, default: all). Example: -t returning event_filter')
    
    args = parser.parse_args()
    main(args)