#!/usr/bin/env python3
"""
Core User Behavior Analyzer Class
Contains the main UserBehaviorAnalyzer class with database operations and analysis methods
"""

import pandas as pd
import sqlite3
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple

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
    
    def get_day1_actions_of_returning_users(self, export_path: str = None, export_format: str = 'json') -> Dict:
        """
        Identify what actions were taken on the first day and second day by users who returned 
        to open the app the next day (day 1 return - logged in on two consecutive days)
        
        Args:
            export_path: Optional path to export results
            export_format: Format for export ('json' or 'csv')
        
        Returns:
            Standardized dictionary with analysis results including both day 1 and day 2 events
        """
        print("🔍 Analyzing day 1 and day 2 actions of users who returned on day 2...")
        
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
                    # This user returned the next day - get their day 1 and day 2 actions
                    day0_actions = events_by_date[first_day]
                    day1_actions = events_by_date[second_day]
                    
                    # Create event sequence lists
                    day0_event_sequence = [action['event'] for action in day0_actions]
                    day1_event_sequence = [action['event'] for action in day1_actions]
                    combined_event_sequence = day0_event_sequence + day1_event_sequence
                    
                    returning_users_day1_actions[device_id] = {
                        'first_day': first_day.isoformat(),
                        'second_day': second_day.isoformat(),
                        'day0_actions': day1_actions,
                        'day0_action_count': len(day1_actions),
                        'day0_unique_events': list(set([action['event'] for action in day1_actions])),
                        'day0_event_sequence': day1_event_sequence,  # Existing: just day 1 sequence for backward compatibility
                        'day1_actions': day1_actions,
                        'day1_action_count': len(day1_actions),
                        'day1_unique_events': list(set([action['event'] for action in day1_actions])),
                        'day1_event_sequence': day1_event_sequence,
                        'combined_event_sequence': combined_event_sequence,  # New: full 2-day sequence
                        'total_two_day_events': len(day1_actions) + len(day1_actions),
                        'timezone': row['timezone'],
                        'country': row['country']
                    }
                
            except (json.JSONDecodeError, ValueError, KeyError) as e:
                print(f"⚠️  Error processing device {row.get('device_id', 'unknown')}: {e}")
                continue
        
        print(f"✅ Found {len(returning_users_day1_actions)} users who returned on day 2")
        
        # Create summary for standardized output
        summary = {
            'total_returning_users': len(returning_users_day1_actions),
            'analysis_parameters': {
                'filter_criteria': 'Users who returned on day 2',
                'data_included': ['day0_actions', 'day1_actions', 'event_sequences', 'combined_sequences', 'timing_info', 'event_counts']
            }
        }
        
        # Create standardized output
        standardized_output = self._create_standardized_output(
            'returning_users_analysis', 
            returning_users_day1_actions, 
            summary
        )
        
        # Optional export
        if export_path:
            self.export_analysis_results(standardized_output, export_path, export_format)
        
        return standardized_output
    
    def get_users_without_day1_return(self, export_path: str = None, export_format: str = 'json') -> Dict:
        """
        Identify users who don't have day 1 return (didn't come back the next day)
        
        Args:
            export_path: Optional path to export results
            export_format: Format for export ('json' or 'csv')
        
        Returns:
            Standardized dictionary with analysis results
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
                    day0_actions = events_by_date[first_day]
                    
                    # Check if they had any activity after day 1 (but not day 2)
                    later_activity = any(date > second_day for date in sorted_dates)
                    
                    # Create event sequence list (just event names in order)
                    event_sequence = [action['event'] for action in day0_actions]
                    
                    non_returning_users[device_id] = {
                        'first_day': first_day.isoformat(),
                        'expected_return_day': second_day.isoformat(),
                        'day0_actions': day0_actions,
                        'day0_action_count': len(day0_actions),
                        'day0_unique_events': list(set([action['event'] for action in day0_actions])),
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
        
        # Create summary for standardized output
        summary = {
            'total_non_returning_users': len(non_returning_users),
            'analysis_parameters': {
                'filter_criteria': 'Users who did not return on day 2',
                'data_included': ['day0_actions', 'event_sequence', 'later_activity_info']
            }
        }
        
        # Create standardized output
        standardized_output = self._create_standardized_output(
            'non_returning_users_analysis', 
            non_returning_users, 
            summary
        )
        
        # Optional export
        if export_path:
            self.export_analysis_results(standardized_output, export_path, export_format)
        
        return standardized_output
    
    def filter_users_by_locationORtimezone(self, country: Optional[str] = None, timezone: Optional[str] = None, export_path: str = None, export_format: str = 'json') -> Dict:
        """
        Filter users based on location (country and/or timezone)
        
        Args:
            country: Country name to filter by (e.g., "United States")
            timezone: Timezone to filter by (e.g., "America/New_York")
            export_path: Optional path to export results
            export_format: Format for export ('json' or 'csv')
            
        Returns:
            Standardized dictionary with analysis results
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
        
        # Convert DataFrame to dict for standardized output
        results = filtered_df.to_dict('records')
        
        # Create summary for standardized output
        summary = {
            'total_users_found': len(results),
            'filter_criteria': {
                'country': country,
                'timezone': timezone
            },
            'analysis_parameters': {
                'data_included': ['device_info', 'location_data', 'event_summary']
            }
        }
        
        # Create standardized output
        standardized_output = self._create_standardized_output(
            'location_analysis', 
            results, 
            summary
        )
        
        # Optional export
        if export_path:
            self.export_analysis_results(standardized_output, export_path, export_format)
        
        return standardized_output
    
    def get_user_info_by_device_list(self, device_ids: List[str], include_event_details: bool = True, export_path: str = None, export_format: str = 'json') -> Dict:
        """
        Get all user information for a given list of device_ids
        
        Args:
            device_ids: List of device IDs to retrieve information for
            include_event_details: Whether to include parsed event_time_pairs details
            export_path: Optional path to export results
            export_format: Format for export ('json' or 'csv')
            
        Returns:
            Standardized dictionary with analysis results
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
        
        # Create summary for standardized output
        summary = {
            'requested_devices': len(device_ids),
            'found_devices': len(user_info),
            'missing_devices': len(missing_devices),
            'missing_device_list': missing_devices,
            'analysis_parameters': {
                'include_event_details': include_event_details,
                'data_included': ['device_info', 'event_summary', 'time_analysis'] if include_event_details else ['device_info']
            }
        }
        
        # Create standardized output
        standardized_output = self._create_standardized_output(
            'device_info_analysis', 
            user_info, 
            summary
        )
        
        # Optional export
        if export_path:
            self.export_analysis_results(standardized_output, export_path, export_format)
        
        return standardized_output
    
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
    
    def export_analysis_results(self, data: Dict, filename: str, format_type: str = 'json'):
        """
        Export analysis results to file with custom formatting for event sequences
        
        Args:
            data: Data to export
            filename: Output filename
            format_type: 'json' or 'csv'
        """
        if format_type == 'json':
            # Custom JSON formatting to keep event sequence arrays on single lines
            def format_json_with_compact_arrays(obj, indent=0):
                """Custom JSON formatter that keeps event sequence arrays on single lines"""
                if isinstance(obj, dict):
                    items = []
                    for key, value in obj.items():
                        if key in ['event_sequence', 'day1_event_sequence', 'combined_event_sequence'] and isinstance(value, list):
                            # Format all event sequence arrays on a single line
                            formatted_value = json.dumps(value, ensure_ascii=False)
                        elif isinstance(value, (dict, list)) and key not in ['event_sequence', 'day1_event_sequence', 'combined_event_sequence']:
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
            import pandas as pd
            df = pd.DataFrame.from_dict(data, orient='index')
            df.to_csv(filename, encoding='utf-8')
        
        print(f"✅ Exported results to {filename}")
    
    def _create_standardized_output(self, analysis_type: str, results: Dict, summary: Dict = None) -> Dict:
        """
        Create standardized output format for all analysis functions
        
        Args:
            analysis_type: Type of analysis (e.g., 'returning_users', 'event_sequences')
            results: Main analysis results
            summary: Optional summary statistics
            
        Returns:
            Standardized output dictionary
        """
        from datetime import datetime
        
        output = {
            'analysis_info': {
                'analysis_type': analysis_type,
                'timestamp': datetime.now().isoformat(),
                'total_records': len(results) if isinstance(results, dict) else len(results) if isinstance(results, list) else 0,
                'database_path': self.db_path
            },
            'summary': summary or {},
            'results': results
        }
        
        return output 

if __name__=='__main__':
    data_path = '/Users/shuhangge/Desktop/my_projects/Sekai/DataAgent/DataProcess/event_analysis_fake.db'
    analyzer = UserBehaviorAnalyzer(data_path)
    result = analyzer.get_day1_actions_of_returning_users()
