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
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

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
                    day1_actions = events_by_date[first_day]
                    day2_actions = events_by_date[second_day]
                    
                    # Create event sequence lists
                    day1_event_sequence = [action['event'] for action in day1_actions]
                    day2_event_sequence = [action['event'] for action in day2_actions]
                    combined_event_sequence = day1_event_sequence + day2_event_sequence
                    
                    returning_users_day1_actions[device_id] = {
                        'first_day': first_day.isoformat(),
                        'second_day': second_day.isoformat(),
                        'day1_actions': day1_actions,
                        'day1_action_count': len(day1_actions),
                        'day1_unique_events': list(set([action['event'] for action in day1_actions])),
                        'day1_event_sequence': day1_event_sequence,  # Existing: just day 1 sequence for backward compatibility
                        'day2_actions': day2_actions,
                        'day2_action_count': len(day2_actions),
                        'day2_unique_events': list(set([action['event'] for action in day2_actions])),
                        'day2_event_sequence': day2_event_sequence,
                        'combined_event_sequence': combined_event_sequence,  # New: full 2-day sequence
                        'total_two_day_events': len(day1_actions) + len(day2_actions),
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
                'data_included': ['day1_actions', 'day2_actions', 'event_sequences', 'combined_sequences', 'timing_info', 'event_counts']
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
        
        # Create summary for standardized output
        summary = {
            'total_non_returning_users': len(non_returning_users),
            'analysis_parameters': {
                'filter_criteria': 'Users who did not return on day 2',
                'data_included': ['day1_actions', 'event_sequence', 'later_activity_info']
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
    
    def get_event_user_counts(self, export_path: str = None, export_format: str = 'json') -> Dict:
        """
        Get the number of users for each event type and time pairs
        
        Args:
            export_path: Optional path to export results
            export_format: Format for export ('json' or 'csv')
        
        Returns:
            Standardized dictionary with analysis results
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
        
        # Calculate summary statistics
        total_events = sum(stats['total_occurrences'] for stats in event_stats.values())
        total_unique_users = len(set().union(*[stats['unique_users'] for stats in event_stats.values()]))
        
        summary = {
            'total_event_types': len(event_stats),
            'total_events_analyzed': total_events,
            'total_unique_users': total_unique_users,
            'analysis_parameters': {
                'data_included': ['event_counts', 'user_counts', 'time_distribution', 'usage_statistics']
            }
        }
        
        # Create standardized output
        standardized_output = self._create_standardized_output(
            'event_statistics_analysis', 
            event_stats, 
            summary
        )
        
        # Optional export
        if export_path:
            self.export_analysis_results(standardized_output, export_path, export_format)
        
        return standardized_output
    
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
    
    def filter_devices_by_event_count(self, min_events: int = 1, max_events: int = None, export_path: str = None, export_format: str = 'json') -> Dict:
        """
        Filter devices by total event count
        
        Args:
            min_events: Minimum number of events (inclusive)
            max_events: Maximum number of events (inclusive), None for no upper limit
            export_path: Optional path to export results
            export_format: Format for export ('json' or 'csv')
            
        Returns:
            Standardized dictionary with analysis results
        """
        print(f"🎯 Filtering devices by event count: min={min_events}, max={max_events}")
        
        filtered_df = self.df[self.df['total_events'] >= min_events].copy()
        
        if max_events is not None:
            filtered_df = filtered_df[filtered_df['total_events'] <= max_events]
        
        filtered_df = filtered_df.sort_values('total_events', ascending=False)
        
        print(f"✅ Found {len(filtered_df)} devices matching event count criteria")
        
        # Convert DataFrame to dict for standardized output
        results = filtered_df.to_dict('records')
        
        # Create summary for standardized output
        summary = {
            'total_devices_found': len(results),
            'filter_criteria': {
                'min_events': min_events,
                'max_events': max_events
            },
            'analysis_parameters': {
                'data_included': ['device_info', 'event_counts', 'activity_metrics']
            }
        }
        
        # Create standardized output
        standardized_output = self._create_standardized_output(
            'event_filter_analysis', 
            results, 
            summary
        )
        
        # Optional export
        if export_path:
            self.export_analysis_results(standardized_output, export_path, export_format)
        
        return standardized_output
    
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
    
    def find_common_event_sequences(self, min_sequence_length: int = 2, max_sequence_length: int = 5, min_users: int = 2, top_n: int = 10, export_path: str = None, export_format: str = 'json') -> Dict:
        """
        Find common event sequences (patterns) that multiple users follow
        
        Args:
            min_sequence_length: Minimum length of sequences to analyze (default: 2)
            max_sequence_length: Maximum length of sequences to analyze (default: 5)
            min_users: Minimum number of users that must have the sequence (default: 2)
            top_n: Number of top sequences to return (default: 10)
            export_path: Optional path to export results
            export_format: Format for export ('json' or 'csv')
            
        Returns:
            Standardized dictionary with analysis results
        """
        print(f"🔍 Analyzing common event sequences (length {min_sequence_length}-{max_sequence_length}, min {min_users} users)...")
        
        # Collect all user sequences
        user_sequences = {}
        sequence_patterns = {}
        total_users_analyzed = 0
        
        for _, row in self.df.iterrows():
            try:
                device_id = row['device_id']
                event_pairs = json.loads(row['event_time_pairs'])
                total_users_analyzed += 1
                
                if len(event_pairs) < min_sequence_length:
                    continue  # Skip users with too few events
                
                # Extract event sequence for this user
                user_event_sequence = [event_pair['event'] for event_pair in event_pairs]
                user_sequences[device_id] = user_event_sequence
                
                # Generate all possible subsequences of different lengths
                for seq_length in range(min_sequence_length, min(max_sequence_length + 1, len(user_event_sequence) + 1)):
                    for start_idx in range(len(user_event_sequence) - seq_length + 1):
                        sequence = tuple(user_event_sequence[start_idx:start_idx + seq_length])
                        
                        if sequence not in sequence_patterns:
                            sequence_patterns[sequence] = {
                                'users': set(),
                                'total_occurrences': 0,
                                'sequence_length': seq_length,
                                'first_event': sequence[0],
                                'last_event': sequence[-1],
                                'positions': [],  # Track where in user journey this sequence appears
                                'user_examples': []
                            }
                        
                        sequence_patterns[sequence]['users'].add(device_id)
                        sequence_patterns[sequence]['total_occurrences'] += 1
                        sequence_patterns[sequence]['positions'].append({
                            'user': device_id,
                            'start_position': start_idx,
                            'end_position': start_idx + seq_length - 1,
                            'total_user_events': len(user_event_sequence)
                        })
                        
            except (json.JSONDecodeError, KeyError) as e:
                print(f"⚠️  Error processing device {row.get('device_id', 'unknown')}: {e}")
                continue
        
        # Filter sequences by minimum user count and calculate statistics
        common_sequences = {}
        for sequence, data in sequence_patterns.items():
            user_count = len(data['users'])
            if user_count >= min_users:
                # Calculate additional statistics
                positions = data['positions']
                avg_position = sum(p['start_position'] for p in positions) / len(positions)
                
                # Analyze position in user journey
                early_journey = sum(1 for p in positions if p['start_position'] <= 2)
                mid_journey = sum(1 for p in positions if 2 < p['start_position'] < p['total_user_events'] - 2)
                late_journey = sum(1 for p in positions if p['start_position'] >= p['total_user_events'] - 2)
                
                # Get example users
                example_users = list(data['users'])[:5]  # First 5 users as examples
                
                common_sequences[sequence] = {
                    'sequence': list(sequence),
                    'sequence_string': ' → '.join(sequence),
                    'user_count': user_count,
                    'total_occurrences': data['total_occurrences'],
                    'sequence_length': data['sequence_length'],
                    'first_event': data['first_event'],
                    'last_event': data['last_event'],
                    'avg_position_in_journey': round(avg_position, 1),
                    'position_analysis': {
                        'early_journey': early_journey,
                        'mid_journey': mid_journey,
                        'late_journey': late_journey
                    },
                    'percentage_of_users': round((user_count / total_users_analyzed) * 100, 2),
                    'avg_occurrences_per_user': round(data['total_occurrences'] / user_count, 2),
                    'example_users': example_users,
                    'users_list': list(data['users'])
                }
        
        # Sort by user count (most common sequences first)
        sorted_sequences = sorted(common_sequences.items(), key=lambda x: x[1]['user_count'], reverse=True)
        top_sequences = dict(sorted_sequences[:top_n])
        
        # Analyze sequence characteristics
        sequence_lengths = {}
        for seq, data in common_sequences.items():
            length = data['sequence_length']
            if length not in sequence_lengths:
                sequence_lengths[length] = {'count': 0, 'avg_users': 0, 'sequences': []}
            sequence_lengths[length]['count'] += 1
            sequence_lengths[length]['sequences'].append(data['user_count'])
        
        # Calculate averages for each length
        for length, stats in sequence_lengths.items():
            stats['avg_users'] = round(sum(stats['sequences']) / len(stats['sequences']), 1)
            del stats['sequences']  # Remove raw data
        
        # Create analysis summary
        analysis_summary = {
            'total_users_analyzed': total_users_analyzed,
            'total_unique_sequences': len(common_sequences),
            'sequences_with_min_users': len(common_sequences),
            'top_sequences_count': len(top_sequences),
            'sequence_length_distribution': sequence_lengths,
            'analysis_parameters': {
                'min_sequence_length': min_sequence_length,
                'max_sequence_length': max_sequence_length,
                'min_users': min_users
            }
        }
        
        result = {
            'summary': analysis_summary,
            'top_sequences': top_sequences,
            'all_sequences_ranking': [
                {
                    'sequence': data['sequence_string'],
                    'user_count': data['user_count'],
                    'percentage_of_users': data['percentage_of_users'],
                    'sequence_length': data['sequence_length']
                } for seq, data in sorted_sequences
            ]
        }
        
        if sorted_sequences:
            top_sequence = sorted_sequences[0][1]
            print(f"✅ Found {len(common_sequences)} common sequences")
            print(f"🏆 Most common: '{top_sequence['sequence_string']}' ({top_sequence['user_count']} users, {top_sequence['percentage_of_users']}%)")
        else:
            print(f"✅ No common sequences found with minimum {min_users} users")
        
        # Create standardized output
        standardized_output = self._create_standardized_output(
            'sequence_analysis', 
            top_sequences, 
            analysis_summary
        )
        
        # Optional export
        if export_path:
            self.export_analysis_results(standardized_output, export_path, export_format)
        
        return standardized_output
    @classmethod
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
                        if key in ['event_sequence', 'day2_event_sequence', 'combined_event_sequence'] and isinstance(value, list):
                            # Format all event sequence arrays on a single line
                            formatted_value = json.dumps(value, ensure_ascii=False)
                        elif isinstance(value, (dict, list)) and key not in ['event_sequence', 'day2_event_sequence', 'combined_event_sequence']:
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
    
    def plot_event_per_user(self, device_ids: List[str] = None, export_path: str = None, export_format: str = 'json', 
                               show_plots: bool = True, save_plots: bool = True, 
                               plot_dir: str = './plots') -> Dict:
        """
        Create visualizations for the distribution of event numbers across users
        
        Args:
            device_ids: Optional list of device IDs to analyze (if None, analyze all users)
            export_path: Optional path to export analysis results
            export_format: Format for export ('json' or 'csv')
            show_plots: Whether to display plots interactively
            save_plots: Whether to save plots to files
            plot_dir: Directory to save plots
            
        Returns:
            Standardized dictionary with analysis results and plot information
        """
        if device_ids:
            print(f"📊 Analyzing event number distribution for {len(device_ids)} specific users...")
        else:
            print("📊 Analyzing event number distribution for all users...")
        
        # Extract event counts for specified users or all users
        event_counts = []
        device_event_mapping = {}
        
        # Filter dataframe if device_ids specified
        if device_ids:
            filtered_df = self.df[self.df['device_id'].isin(device_ids)]
            if filtered_df.empty:
                print("❌ No matching devices found in database")
                return self._create_standardized_output('event_distribution_analysis', {}, {})
        else:
            filtered_df = self.df
        
        for _, row in filtered_df.iterrows():
            try:
                device_id = row['device_id']
                total_events = row['total_events']
                event_counts.append(total_events)
                device_event_mapping[device_id] = total_events
            except Exception as e:
                print(f"⚠️  Error processing device {row.get('device_id', 'unknown')}: {e}")
                continue
        
        if not event_counts:
            print("❌ No event data found for analysis")
            return self._create_standardized_output('event_distribution_analysis', {}, {})
        
        # Convert to numpy array for easier analysis
        event_counts = np.array(event_counts)
        
        # Calculate statistics
        stats = {
            'total_users': len(event_counts),
            'mean_events': float(np.mean(event_counts)),
            'median_events': float(np.median(event_counts)),
            'std_events': float(np.std(event_counts)),
            'min_events': int(np.min(event_counts)),
            'max_events': int(np.max(event_counts)),
            'q25_events': float(np.percentile(event_counts, 25)),
            'q75_events': float(np.percentile(event_counts, 75)),
            'iqr_events': float(np.percentile(event_counts, 75) - np.percentile(event_counts, 25))
        }
        
        # Create distribution bins for analysis
        bins = [1, 2, 3, 5, 10, 20, 50, 100, float('inf')]
        bin_labels = ['1', '2', '3-4', '5-9', '10-19', '20-49', '50-99', '100+']
        
        distribution_counts = {}
        for i, (start, end) in enumerate(zip(bins[:-1], bins[1:])):
            if end == float('inf'):
                count = np.sum(event_counts >= start)
                label = bin_labels[i]
            else:
                count = np.sum((event_counts >= start) & (event_counts < end))
                label = bin_labels[i]
            distribution_counts[label] = int(count)
        
        # Create plots if requested
        plot_files = []
        if save_plots or show_plots:
            # Create plot directory if it doesn't exist
            if save_plots:
                os.makedirs(plot_dir, exist_ok=True)
            
            # Set style for better-looking plots
            plt.style.use('default')
            sns.set_palette("husl")
            
            # Add suffix to plot names if filtering by device_ids
            plot_suffix = f"_filtered_{len(device_ids)}_users" if device_ids else ""
            
            # Handle outliers for balanced visualization
            # Calculate percentiles to identify outliers
            p95 = np.percentile(event_counts, 95)
            p99 = np.percentile(event_counts, 99)
            
            # Create main dataset excluding extreme outliers for better visualization
            main_data = event_counts[event_counts <= p95]
            outliers_count = len(event_counts) - len(main_data)
            
            # Determine optimal number of bins
            n_bins = min(50, max(10, int(np.sqrt(len(main_data)))))
            
            # Create dual-panel figure
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            
            # LEFT PANEL: Main Range (excluding top 5% outliers)
            ax1.hist(main_data, bins=n_bins, alpha=0.7, color='skyblue', edgecolor='black')
            
            # Add mean and median lines
            ax1.axvline(stats['mean_events'], color='red', linestyle='--', linewidth=2, label=f'Mean: {stats["mean_events"]:.1f}')
            ax1.axvline(stats['median_events'], color='green', linestyle='--', linewidth=2, label=f'Median: {stats["median_events"]:.1f}')
            
            ax1.set_xlabel('Number of Events per User')
            ax1.set_ylabel('Number of Users')
            ax1.set_title(f'Event Distribution - Main Range ({stats["total_users"]} filtered users)')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            
            # Add outlier information box
            if outliers_count > 0:
                outlier_text = f'Excluded {outliers_count} users with >{p95:.0f} events\n(Top 5% outliers: {np.min(event_counts[event_counts > p95]):.0f}-{np.max(event_counts[event_counts > p95]):.0f} events)'
                ax1.text(0.98, 0.85, outlier_text, transform=ax1.transAxes, 
                        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8),
                        verticalalignment='top', horizontalalignment='right', fontsize=9)
            
            # RIGHT PANEL: Full Range with Log Scale
            # Use log-spaced bins for better visualization
            if np.max(event_counts) > 0:
                log_bins = np.logspace(0, np.log10(max(event_counts.max(), 1)), 30)
                ax2.hist(event_counts, bins=log_bins, alpha=0.7, color='lightcoral', edgecolor='black')
            
            ax2.set_xlabel('Number of Events per User (Log Scale)')
            ax2.set_ylabel('Number of Users')
            ax2.set_title(f'Event Distribution - Full Range (Log Scale) ({stats["total_users"]} filtered users)')
            ax2.set_xscale('log')
            ax2.grid(True, alpha=0.3)
            
            # Add percentile markers
            percentiles = [50, 75, 90, 95, 99]
            for p in percentiles:
                if p <= 99:  # Only show percentiles that make sense
                    value = np.percentile(event_counts, p)
                    if value > 0:  # Only show if value is positive for log scale
                        ax2.axvline(value, color='red', linestyle=':', alpha=0.7)
                        ax2.text(value, ax2.get_ylim()[1] * 0.9, f'P{p}: {value:.0f}', 
                                rotation=90, ha='right', va='top', fontsize=8)
            
            plt.tight_layout()
            
            # Save the plot
            if save_plots:
                plot_filename = f'{plot_dir}/event_distribution_histogram_filtered_{stats["total_users"]}_users.png'
                plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
                plot_files.append(plot_filename)
                print(f"📁 Histogram saved to {plot_filename}")
            
            if show_plots:
                plt.show()
            else:
                plt.close()
        
        # Prepare results
        results = {
            'statistics': stats,
            'distribution_by_bins': distribution_counts,
            'raw_event_counts': event_counts.tolist(),
            'device_event_mapping': device_event_mapping,
            'plot_files': plot_files if save_plots else [],
            'plot_directory': plot_dir if save_plots else None,
            'filtered_analysis': device_ids is not None,
            'filter_device_count': len(device_ids) if device_ids else None,
            'outlier_info': {
                'p95_threshold': float(np.percentile(event_counts, 95)),
                'p99_threshold': float(np.percentile(event_counts, 99)),
                'outliers_excluded_from_main_plot': int(len(event_counts) - len(event_counts[event_counts <= np.percentile(event_counts, 95)]))
            }
        }
        
        # Create summary for standardized output
        summary = {
            'total_users_analyzed': len(event_counts),
            'distribution_summary': {
                'mean_events_per_user': stats['mean_events'],
                'median_events_per_user': stats['median_events'],
                'most_common_range': max(distribution_counts, key=distribution_counts.get),
                'users_with_1_event': distribution_counts.get('1', 0),
                'users_with_10plus_events': sum(v for k, v in distribution_counts.items() if k in ['10-19', '20-49', '50-99', '100+'])
            },
            'visualization_info': {
                'plots_created': len(plot_files),
                'plots_saved': save_plots,
                'plots_displayed': show_plots
            },
            'filter_info': {
                'filtered_analysis': device_ids is not None,
                'requested_devices': len(device_ids) if device_ids else None,
                'found_devices': len(event_counts)
            },
            'analysis_parameters': {
                'data_included': ['statistics', 'distribution_bins', 'raw_data', 'visualizations']
            }
        }
        
        print(f"✅ Event distribution analysis complete!")
        print(f"📊 Analyzed {len(event_counts)} users")
        print(f"📈 Mean events per user: {stats['mean_events']:.1f}")
        print(f"📊 Median events per user: {stats['median_events']:.1f}")
        print(f"🎯 Most common range: {max(distribution_counts, key=distribution_counts.get)} events")
        
        # Create standardized output
        standardized_output = self._create_standardized_output(
            'event_distribution_analysis', 
            results, 
            summary
        )
        
        # Optional export
        if export_path:
            self.export_analysis_results(standardized_output, export_path, export_format)
        
        return standardized_output

# STEP 1: FILTER FUNCTIONS - Take original database as input, return filtered data dictionary

def analyze_day1_returning_users(db_path: str) -> Dict:
    """
    FILTER FUNCTION: Analyze users who returned on day 2 from original database
    Returns filtered data dictionary that can be used by analysis functions
    """
    analyzer = UserBehaviorAnalyzer(db_path)
    standardized_result = analyzer.get_day1_actions_of_returning_users()
    
    # Extract the actual user data for use in analysis functions
    return standardized_result['results']

def analyze_users_without_day1_return(db_path: str) -> Dict:
    """
    FILTER FUNCTION: Analyze users who didn't return on day 2 from original database
    Returns filtered data dictionary that can be used by analysis functions
    """
    analyzer = UserBehaviorAnalyzer(db_path)
    standardized_result = analyzer.get_users_without_day1_return()
    
    # Extract the actual user data for use in analysis functions
    return standardized_result['results']

def filter_by_location(db_path: str, country: str = None, timezone: str = None) -> Dict:
    """
    FILTER FUNCTION: Filter users by location from original database
    Returns filtered data dictionary that can be used by analysis functions
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
    """
    analyzer = UserBehaviorAnalyzer(db_path)
    standardized_result = analyzer.get_user_info_by_device_list(device_ids)
    
    # Extract the actual user data for use in analysis functions
    return standardized_result['results']

# STEP 2: ANALYSIS FUNCTIONS - Take filtered data dictionary as input

def get_event_statistics(filtered_data: Dict) -> Dict:
    """
    ANALYSIS FUNCTION: Analyze event statistics from filtered data
    
    Args:
        filtered_data: Dictionary of filtered user data from filter functions
        
    Returns:
        Dictionary with event statistics analysis
    """
    print("📊 Analyzing event statistics from filtered data...")
    
    event_stats = {}
    total_users = len(filtered_data)
    
    for device_id, user_data in filtered_data.items():
        # Handle different data formats from different filter functions
        event_pairs = []
        
        if 'event_time_pairs' in user_data:
            # From get_users_by_device_ids or filter_by_location
            event_pairs = user_data['event_time_pairs']
        elif 'day1_actions' in user_data:
            # From returning/non-returning user analysis
            event_pairs = user_data['day1_actions']
            if 'day2_actions' in user_data:
                event_pairs.extend(user_data['day2_actions'])
        
        # Process events
        for event_pair in event_pairs:
            event_name = event_pair['event']
            hour = event_pair.get('hour', 0)
            day_of_week = event_pair.get('day_of_week', 'Unknown')
            
            if event_name not in event_stats:
                event_stats[event_name] = {
                    'total_occurrences': 0,
                    'unique_users': set(),
                    'hours_distribution': {},
                    'days_distribution': {}
                }
            
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
    
    # Convert sets to counts
    for event_name in event_stats:
        event_stats[event_name]['unique_user_count'] = len(event_stats[event_name]['unique_users'])
        event_stats[event_name]['unique_users'] = list(event_stats[event_name]['unique_users'])
        event_stats[event_name]['avg_events_per_user'] = (
            event_stats[event_name]['total_occurrences'] / 
            event_stats[event_name]['unique_user_count'] if event_stats[event_name]['unique_user_count'] > 0 else 0
        )
    
    print(f"✅ Analyzed {len(event_stats)} event types from {total_users} users")
    
    return {
        'analysis_info': {'analysis_type': 'event_statistics', 'total_users_analyzed': total_users},
        'summary': {'total_event_types': len(event_stats), 'total_users': total_users},
        'results': event_stats
    }

def get_events_by_popularity(filtered_data: Dict, ascending: bool = False) -> List[Tuple[str, Dict]]:
    """
    ANALYSIS FUNCTION: Get events sorted by popularity from filtered data
    
    Args:
        filtered_data: Dictionary of filtered user data from filter functions
        ascending: Whether to sort in ascending order
        
    Returns:
        List of tuples (event_name, event_stats) sorted by user count
    """
    event_stats_result = get_event_statistics(filtered_data)
    event_stats = event_stats_result['results']
    
    # Sort by unique user count
    sorted_events = sorted(
        event_stats.items(), 
        key=lambda x: x[1]['unique_user_count'], 
        reverse=not ascending
    )
    
    return sorted_events

def plot_event_per_user(filtered_data: Dict, device_ids: List[str] = None, show_plots: bool = True, save_plots: bool = True, plot_dir: str = './plots') -> Dict:
    """
    ANALYSIS FUNCTION: Plot event distribution from filtered data
    
    Args:
        filtered_data: Dictionary of filtered user data from filter functions
        device_ids: Optional list to further filter the data
        show_plots: Whether to display plots
        save_plots: Whether to save plots to files
        plot_dir: Directory to save plots
        
    Returns:
        Dictionary with plot analysis results
    """
    print("📈 Creating event distribution plots from filtered data...")
    
    # Further filter by device_ids if provided
    if device_ids:
        filtered_data = {k: v for k, v in filtered_data.items() if k in device_ids}
    
    # Collect event counts per user
    event_counts = []
    
    for device_id, user_data in filtered_data.items():
        # Handle different data formats
        total_events = 0
        
        if 'total_events' in user_data:
            total_events = user_data['total_events']
        elif 'day1_action_count' in user_data:
            total_events = user_data['day1_action_count']
            if 'day2_action_count' in user_data:
                total_events += user_data['day2_action_count']
        elif 'event_time_pairs' in user_data:
            total_events = len(user_data['event_time_pairs'])
        
        event_counts.append(total_events)
    
    if not event_counts:
        print("⚠️  No event data found in filtered data")
        return {'analysis_info': {'analysis_type': 'event_distribution', 'total_users': 0}, 'summary': {}, 'results': {}}
    
    # Create plots directory
    if save_plots:
        os.makedirs(plot_dir, exist_ok=True)
    
    # Convert to numpy array for easier analysis
    event_counts = np.array(event_counts)
    total_users = len(filtered_data)
    
    # Calculate statistics
    mean_events = np.mean(event_counts)
    median_events = np.median(event_counts)
    
    # Calculate percentiles for outlier detection
    p95 = np.percentile(event_counts, 95)
    
    # Filter out top 5% outliers for main histogram
    main_data = event_counts[event_counts <= p95]
    outliers_removed = len(event_counts) - len(main_data)
    min_outlier = np.min(event_counts[event_counts > p95]) if outliers_removed > 0 else 0
    max_outlier = np.max(event_counts[event_counts > p95]) if outliers_removed > 0 else 0
    
    # Create dual-panel figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # LEFT PANEL: Main Range (excluding top 5% outliers)
    n_bins = min(50, max(10, int(np.sqrt(len(main_data)))))
    ax1.hist(main_data, bins=n_bins, alpha=0.7, color='skyblue', edgecolor='black')
    
    # Add mean and median lines
    ax1.axvline(mean_events, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_events:.1f}')
    ax1.axvline(median_events, color='green', linestyle='--', linewidth=2, label=f'Median: {median_events:.1f}')
    
    ax1.set_xlabel('Number of Events per User')
    ax1.set_ylabel('Number of Users')
    ax1.set_title(f'Event Distribution - Main Range ({total_users} filtered users)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Add outlier information box
    if outliers_removed > 0:
        outlier_text = f'Excluded {outliers_removed} users with >{p95:.0f} events\n(Top 5% outliers: {min_outlier:.0f}-{max_outlier:.0f} events)'
        ax1.text(0.98, 0.85, outlier_text, transform=ax1.transAxes, 
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8),
                verticalalignment='top', horizontalalignment='right', fontsize=9)
    
    # RIGHT PANEL: Full Range with Log Scale
    # Use log-spaced bins for better visualization
    if np.max(event_counts) > 0:
        log_bins = np.logspace(0, np.log10(max(event_counts.max(), 1)), 30)
        ax2.hist(event_counts, bins=log_bins, alpha=0.7, color='lightcoral', edgecolor='black')
    
    ax2.set_xlabel('Number of Events per User (Log Scale)')
    ax2.set_ylabel('Number of Users')
    ax2.set_title(f'Event Distribution - Full Range (Log Scale) ({total_users} filtered users)')
    ax2.set_xscale('log')
    ax2.grid(True, alpha=0.3)
    
    # Add percentile markers
    percentiles = [50, 75, 90, 95, 99]
    for p in percentiles:
        if p <= 99:  # Only show percentiles that make sense
            value = np.percentile(event_counts, p)
            if value > 0:  # Only show if value is positive for log scale
                ax2.axvline(value, color='red', linestyle=':', alpha=0.7)
                ax2.text(value, ax2.get_ylim()[1] * 0.9, f'P{p}: {value:.0f}', 
                        rotation=90, ha='right', va='top', fontsize=8)
    
    plt.tight_layout()
    
    # Save the plot
    if save_plots:
        plot_filename = f'{plot_dir}/event_distribution_histogram_filtered_{total_users}_users.png'
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"📁 Histogram saved to {plot_filename}")
    
    if show_plots:
        plt.show()
    else:
        plt.close()
    
    # Calculate comprehensive statistics
    stats = {
        'total_users': total_users,
        'mean_events': float(mean_events),
        'median_events': float(median_events),
        'min_events': int(np.min(event_counts)),
        'max_events': int(np.max(event_counts)),
        'std_events': float(np.std(event_counts)),
        'outliers_removed': int(outliers_removed),
        'p95_threshold': float(p95),
        'percentiles': {f'p{p}': float(np.percentile(event_counts, p)) for p in [25, 50, 75, 90, 95, 99]}
    }
    
    print(f"✅ Event distribution analysis complete - {total_users} users analyzed")
    print(f"📊 Mean: {mean_events:.1f}, Median: {median_events:.1f}, Max: {np.max(event_counts)}")
    
    return {
        'analysis_info': {'analysis_type': 'event_distribution', 'plot_settings': {'show_plots': show_plots, 'save_plots': save_plots}},
        'summary': stats,
        'results': {'event_counts': event_counts.tolist(), 'main_data': main_data.tolist()}
    }

def find_frequent_events(filtered_data: Dict, min_sequence_length: int = 2, max_sequence_length: int = 5, min_users: int = 2, top_n: int = 10) -> Dict:
    """
    ANALYSIS FUNCTION: Find frequent event sequences from filtered data
    
    Args:
        filtered_data: Dictionary of filtered user data from filter functions
        min_sequence_length: Minimum length of sequences to find
        max_sequence_length: Maximum length of sequences to find
        min_users: Minimum number of users that must have the sequence
        top_n: Number of top sequences to return
        
    Returns:
        Dictionary with frequent event sequence analysis
    """
    print("🔍 Finding frequent event sequences from filtered data...")
    
    all_sequences = []
    
    for device_id, user_data in filtered_data.items():
        # Extract event sequence based on data format
        event_sequence = []
        
        if 'event_sequence' in user_data:
            # From returning/non-returning user analysis
            event_sequence = user_data['event_sequence']
        elif 'combined_event_sequence' in user_data:
            # From returning user analysis with both days
            event_sequence = user_data['combined_event_sequence']
        elif 'event_time_pairs' in user_data:
            # From other filter functions
            event_sequence = [event['event'] for event in user_data['event_time_pairs']]
        
        # Generate subsequences of different lengths
        for length in range(min_sequence_length, min(len(event_sequence) + 1, max_sequence_length + 1)):
            for i in range(len(event_sequence) - length + 1):
                sequence = tuple(event_sequence[i:i + length])
                all_sequences.append((sequence, device_id))
    
    # Count sequence occurrences and track users
    sequence_stats = {}
    for sequence, device_id in all_sequences:
        if sequence not in sequence_stats:
            sequence_stats[sequence] = {'count': 0, 'users': set()}
        sequence_stats[sequence]['count'] += 1
        sequence_stats[sequence]['users'].add(device_id)
    
    # Filter by minimum user requirement and convert to final format
    frequent_sequences = {}
    for sequence, stats in sequence_stats.items():
        if len(stats['users']) >= min_users:
            frequent_sequences[' → '.join(sequence)] = {
                'sequence': list(sequence),
                'total_occurrences': stats['count'],
                'unique_users': len(stats['users']),
                'user_list': list(stats['users']),
                'sequence_length': len(sequence),
                'avg_occurrences_per_user': stats['count'] / len(stats['users'])
            }
    
    # Sort by user count and get top N
    sorted_sequences = sorted(
        frequent_sequences.items(),
        key=lambda x: x[1]['unique_users'],
        reverse=True
    )[:top_n]
    
    result_sequences = dict(sorted_sequences)
    
    print(f"✅ Found {len(result_sequences)} frequent sequences from {len(filtered_data)} users")
    
    return {
        'analysis_info': {'analysis_type': 'frequent_sequences', 'parameters': {'min_sequence_length': min_sequence_length, 'max_sequence_length': max_sequence_length, 'min_users': min_users}},
        'summary': {'total_sequences_found': len(result_sequences), 'total_users_analyzed': len(filtered_data)},
        'results': result_sequences
    }

def filter_devices_by_events(filtered_data: Dict, min_events: int = 1, max_events: int = None, device_ids: List[str] = None) -> Dict:
    """
    ANALYSIS FUNCTION: Further filter devices by event count from already filtered data
    
    Args:
        filtered_data: Dictionary of filtered user data from filter functions
        min_events: Minimum number of events required
        max_events: Maximum number of events allowed (None for no limit)
        device_ids: Optional list to further filter by specific device IDs
        
    Returns:
        Dictionary with further filtered data
    """
    print(f"🔢 Further filtering devices by event count (min: {min_events}, max: {max_events})...")
    
    result_data = {}
    
    for device_id, user_data in filtered_data.items():
        # Skip if device_ids filter is provided and this device is not in it
        if device_ids and device_id not in device_ids:
            continue
        
        # Get event count based on data format
        event_count = 0
        
        if 'total_events' in user_data:
            event_count = user_data['total_events']
        elif 'day1_action_count' in user_data:
            event_count = user_data['day1_action_count']
            if 'day2_action_count' in user_data:
                event_count += user_data['day2_action_count']
        elif 'event_time_pairs' in user_data:
            event_count = len(user_data['event_time_pairs'])
        
        # Apply filters
        if event_count >= min_events:
            if max_events is None or event_count <= max_events:
                result_data[device_id] = user_data
    
    print(f"✅ Filtered to {len(result_data)} devices from {len(filtered_data)} original devices")
    
    return {
        'analysis_info': {'analysis_type': 'event_count_filter', 'parameters': {'min_events': min_events, 'max_events': max_events}},
        'summary': {'devices_found': len(result_data), 'original_devices': len(filtered_data)},
        'results': result_data
    }

def plot_event_counts(filtered_data: Dict, top_n: int = 20, show_plots: bool = True, save_plots: bool = True, plot_dir: str = './plots') -> Dict:
    """
    ANALYSIS FUNCTION: Plot count of each unique event type from filtered data
    
    Args:
        filtered_data: Dictionary of filtered user data from filter functions
        top_n: Number of top events to display (default: 20)
        show_plots: Whether to display plots
        save_plots: Whether to save plots to files
        plot_dir: Directory to save plots
        
    Returns:
        Dictionary with event count analysis results
    """
    print("📊 Creating event counts visualization from filtered data...")
    
    # Collect all events and count occurrences
    event_counts = {}
    total_events = 0
    total_users = len(filtered_data)
    
    for device_id, user_data in filtered_data.items():
        # Handle different data formats from different filter functions
        events_list = []
        
        if 'event_time_pairs' in user_data:
            # From get_users_by_device_ids or filter_by_location
            events_list = [event['event'] for event in user_data['event_time_pairs']]
        elif 'day1_actions' in user_data:
            # From returning/non-returning user analysis
            events_list = [event['event'] for event in user_data['day1_actions']]
            if 'day2_actions' in user_data:
                events_list.extend([event['event'] for event in user_data['day2_actions']])
        elif 'combined_event_sequence' in user_data:
            # From returning user analysis with combined sequence
            events_list = user_data['combined_event_sequence']
        elif 'event_sequence' in user_data:
            # From returning/non-returning user analysis with sequence
            events_list = user_data['event_sequence']
        
        # Count events
        for event in events_list:
            event_counts[event] = event_counts.get(event, 0) + 1
            total_events += 1
    
    if not event_counts:
        print("⚠️  No events found in filtered data")
        return {'analysis_info': {'analysis_type': 'event_counts', 'total_users': 0}, 'summary': {}, 'results': {}}
    
    # Create plots directory
    if save_plots:
        os.makedirs(plot_dir, exist_ok=True)
    
    # Sort events by count (descending) and get top N
    sorted_events = sorted(event_counts.items(), key=lambda x: x[1], reverse=True)
    top_events = sorted_events[:top_n] if len(sorted_events) > top_n else sorted_events
    
    # Prepare data for plotting
    event_names = [event[0] for event in top_events]
    counts = [event[1] for event in top_events]
    
    # Create horizontal bar chart
    plt.figure(figsize=(12, max(8, len(top_events) * 0.4)))
    
    # Create color map for better visualization
    colors = plt.cm.viridis(np.linspace(0, 1, len(top_events)))
    
    bars = plt.barh(range(len(event_names)), counts, color=colors)
    
    # Customize the plot
    plt.yticks(range(len(event_names)), event_names)
    plt.xlabel('Number of Occurrences')
    plt.ylabel('Event Type')
    plt.title(f'Top {len(top_events)} Event Types by Count\n({total_users} users, {total_events} total events)')
    plt.grid(True, alpha=0.3, axis='x')
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        width = bar.get_width()
        plt.text(width + max(counts) * 0.01, bar.get_y() + bar.get_height()/2, 
                f'{int(width)}', ha='left', va='center', fontsize=9)
    
    # Invert y-axis so highest count is at top
    plt.gca().invert_yaxis()
    
    plt.tight_layout()
    
    # Save the plot
    if save_plots:
        plot_filename = f'{plot_dir}/event_counts_top{len(top_events)}_filtered_{total_users}_users.png'
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"📁 Event counts plot saved to {plot_filename}")
    
    if show_plots:
        plt.show()
    else:
        plt.close()
    
    # Calculate comprehensive statistics
    total_unique_events = len(event_counts)
    coverage_percentage = (sum(counts) / total_events) * 100 if total_events > 0 else 0
    
    stats = {
        'total_users': total_users,
        'total_events': total_events,
        'total_unique_events': total_unique_events,
        'top_n_displayed': len(top_events),
        'coverage_percentage': coverage_percentage,
        'most_frequent_event': top_events[0][0] if top_events else None,
        'most_frequent_count': top_events[0][1] if top_events else 0,
        'least_frequent_in_top': top_events[-1][0] if top_events else None,
        'least_frequent_count_in_top': top_events[-1][1] if top_events else 0
    }
    
    print(f"✅ Event counts analysis complete - {total_unique_events} unique events from {total_users} users")
    print(f"📊 Most frequent: '{stats['most_frequent_event']}' ({stats['most_frequent_count']} times)")
    print(f"📈 Top {len(top_events)} events cover {coverage_percentage:.1f}% of all events")
    
    return {
        'analysis_info': {'analysis_type': 'event_counts', 'plot_settings': {'show_plots': show_plots, 'save_plots': save_plots, 'top_n': top_n}},
        'summary': stats,
        'results': {'all_event_counts': event_counts, 'top_events': dict(top_events)}
    }

def export_results(data, filename: str, format_type: str = 'json'):
    """
    Export analysis results or filtered data to file
    
    Args:
        data: Either standardized analysis result or raw filtered data dictionary
        filename: Output filename (without extension)
        format_type: 'json' or 'csv'
    """
    import json
    import pandas as pd
    from datetime import datetime
    
    if not data:
        print("⚠️  No data to export")
        return
    
    # Add file extension if not provided
    if not filename.endswith(('.json', '.csv')):
        filename = f"{filename}.{format_type}"
    
    try:
        if format_type.lower() == 'json':
            # Handle both standardized and raw data
            if isinstance(data, dict) and 'analysis_info' in data:
                # Standardized analysis result
                export_data = data
            else:
                # Raw filtered data - wrap in standardized format
                export_data = {
                    'analysis_info': {
                        'analysis_type': 'filtered_data_export',
                        'timestamp': datetime.now().isoformat(),
                        'total_records': len(data) if isinstance(data, dict) else 1,
                        'export_format': 'json'
                    },
                    'summary': {
                        'total_users': len(data) if isinstance(data, dict) else 1,
                        'data_type': 'filtered_user_data'
                    },
                    'results': data
                }
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False, default=str)
            print(f"📁 Data exported to {filename}")
            
        elif format_type.lower() == 'csv':
            # Convert to DataFrame for CSV export
            if isinstance(data, dict) and 'results' in data:
                # Standardized format - extract results
                df_data = data['results']
            else:
                # Raw filtered data
                df_data = data
            
            if isinstance(df_data, dict):
                # Convert dict of user data to DataFrame
                df = pd.DataFrame.from_dict(df_data, orient='index')
            elif isinstance(df_data, list):
                # Convert list to DataFrame
                df = pd.DataFrame(df_data)
            else:
                print("⚠️  Cannot convert data to CSV format")
                return
            
            df.to_csv(filename, index=True, encoding='utf-8')
            print(f"📁 Data exported to {filename}")
            
        else:
            print(f"⚠️  Unsupported format: {format_type}. Use 'json' or 'csv'")
            
    except Exception as e:
        print(f"❌ Export failed: {e}")

def export_filtered_data(filtered_data: Dict, filename: str, format_type: str = 'json'):
    """
    Convenience function to export filtered data from Step 1 filter functions
    
    Args:
        filtered_data: Dictionary of filtered user data from filter functions
        filename: Output filename (without extension)
        format_type: 'json' or 'csv'
    """
    export_results(filtered_data, filename, format_type)

def export_analysis_results(analysis_result: Dict, filename: str, format_type: str = 'json'):
    """
    Convenience function to export standardized analysis results from Step 2 functions
    
    Args:
        analysis_result: Standardized analysis result from analysis functions
        filename: Output filename (without extension) 
        format_type: 'json' or 'csv'
    """
    export_results(analysis_result, filename, format_type)

# Main function for tool demonstration
def main(args=None):
    """
    Main function demonstrating the two-step function architecture
    """
    
    # Default database path
    db_path = "/Users/shuhangge/Desktop/my_projects/Sekai/DataAgent/DataProcess/event_analysis_fake.db"
    
    print("🚀 User Behavior Analysis Tools - Two-Step Architecture")
    print("=" * 60)
    
    print("\n📖 ARCHITECTURE:")
    print("Step 1: FILTER FUNCTIONS - Take database, return filtered data")
    print("Step 2: ANALYSIS FUNCTIONS - Take filtered data, perform analysis")
    
    # Check if database exists
    if not os.path.exists(db_path):
        print(f"\n❌ Database not found: {db_path}")
        print("Please run the data processing pipeline first.")
        return
    
    print(f"\n✅ Database found: {db_path}")
    print("\n📋 STEP 1 - FILTER FUNCTIONS:")
    print("• analyze_day1_returning_users(db_path)")
    print("• analyze_users_without_day1_return(db_path)")  
    print("• filter_by_location(db_path, country, timezone)")
    print("• get_users_by_device_ids(db_path, device_ids)")
    
    print("\n📊 STEP 2 - ANALYSIS FUNCTIONS:")
    print("• get_event_statistics(filtered_data)")
    print("• filter_devices_by_events(filtered_data, min_events, max_events)")
    print("• plot_event_per_user(filtered_data, ...)")
    print("• plot_event_counts(filtered_data, top_n, ...)")
    print("• find_frequent_events(filtered_data, ...)")
    print("• get_events_by_popularity(filtered_data, ascending)")
    
    print("\n🔧 UTILITY FUNCTIONS:")
    print("• extract_device_ids(analysis_result)")
    print("• extract_user_data(analysis_result)") 
    print("• export_results(data, filename, format_type)")
    print("• export_filtered_data(filtered_data, filename, format_type)")
    print("• export_analysis_results(analysis_result, filename, format_type)")
    
    print("\n📚 For detailed documentation and examples:")
    print("See Tools/README_check_data.md")
    
    print("\n✅ Tools ready for use!")
    '''plot the day1 and noday1 users' event distribution and event counts'''
    returning_users = analyze_day1_returning_users(db_path)
    # non_returning_users = analyze_users_without_day1_return(db_path)
    '''Step 2: Analysis - Create visualizations'''
    # # plot_event_per_user(returning_users)     # Event count distribution
    # plot_event_counts(returning_users, top_n=25)  # Individual event frequencies
    # # plot_event_per_user(non_returning_users)     # Compare distributions  
    # plot_event_counts(non_returning_users, top_n=25)  # Compare event types

    # Export the filtered data
    UserBehaviorAnalyzer.export_analysis_results(returning_users, 'returning_users_data', 'csv')
    print(f"📊 Analyzed {len(returning_users)} returning users")

if __name__ == "__main__":
    main()