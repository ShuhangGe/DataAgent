#!/usr/bin/env python3
"""
Event Analysis Functions for User Behavior Analysis Tools
STEP 2 functions that take filtered data dictionaries as input and perform analysis
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

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

def get_users_with_search_terms_percentage(filtered_data: Dict, search_terms: List[str] = ['search'], case_sensitive: bool = False) -> Dict:
    """
    ANALYSIS FUNCTION: Analyze what percentage of users have actions containing given search terms
    
    Args:
        filtered_data: Dictionary of filtered user data from filter functions
        search_terms: List of terms to search for in event names (default: ['search'])
        case_sensitive: Whether to perform case-sensitive matching (default: False)
        
    Returns:
        Dictionary with user search percentage analysis results
    """
    print(f"🔍 Analyzing users with search terms: {search_terms}")
    
    total_users = len(filtered_data)
    users_with_search_terms = []
    users_without_search_terms = []
    search_events_found = {}
    
    if not search_terms:
        search_terms = ['search']
    
    # Convert search terms to appropriate case for matching
    if not case_sensitive:
        search_terms_lower = [term.lower() for term in search_terms]
    
    for device_id, user_data in filtered_data.items():
        # Extract all events for this user from different data formats
        user_events = []
        
        if 'event_time_pairs' in user_data:
            # From get_users_by_device_ids or filter_by_location
            user_events = [event['event'] for event in user_data['event_time_pairs']]
        elif 'day0_actions' in user_data:
            # From returning/non-returning user analysis (updated format)
            user_events = [event['event'] for event in user_data['day0_actions']]
            if 'day1_actions' in user_data:
                user_events.extend([event['event'] for event in user_data['day1_actions']])
        elif 'day1_actions' in user_data:
            # From returning/non-returning user analysis (legacy format)
            user_events = [event['event'] for event in user_data['day1_actions']]
            if 'day2_actions' in user_data:
                user_events.extend([event['event'] for event in user_data['day2_actions']])
        elif 'combined_event_sequence' in user_data:
            # From returning user analysis with combined sequence
            user_events = user_data['combined_event_sequence']
        elif 'event_sequence' in user_data:
            # From returning/non-returning user analysis with sequence
            user_events = user_data['event_sequence']
        
        # Check if any of this user's events contain search terms
        user_has_search_terms = False
        user_search_events = set()
        
        for event in user_events:
            event_check = event.lower() if not case_sensitive else event
            
            for term in (search_terms_lower if not case_sensitive else search_terms):
                if term in event_check:
                    user_has_search_terms = True
                    user_search_events.add(event)
                    
                    # Count occurrences of each search event type
                    if event not in search_events_found:
                        search_events_found[event] = 0
                    search_events_found[event] += 1
                    break  # Found search term in this event, no need to check other terms
        
        # Categorize user
        if user_has_search_terms:
            users_with_search_terms.append(device_id)
        else:
            users_without_search_terms.append(device_id)
    
    # Calculate percentage
    users_with_search_count = len(users_with_search_terms)
    users_without_search_count = len(users_without_search_terms)
    percentage_with_search = (users_with_search_count / total_users * 100) if total_users > 0 else 0
    percentage_without_search = (users_without_search_count / total_users * 100) if total_users > 0 else 0
    
    print(f"✅ Analysis complete: {users_with_search_count}/{total_users} users ({percentage_with_search:.1f}%) have search terms")
    
    # Create summary
    summary = {
        'total_users': total_users,
        'users_with_search_terms': users_with_search_count,
        'users_without_search_terms': users_without_search_count,
        'percentage_with_search_terms': round(percentage_with_search, 2),
        'percentage_without_search_terms': round(percentage_without_search, 2),
        'search_terms_used': search_terms,
        'case_sensitive': case_sensitive,
        'unique_search_events_found': len(search_events_found)
    }
    
    # Create results
    results = {
        'users_with_search_terms': users_with_search_terms,
        'users_without_search_terms': users_without_search_terms,
        'search_events_found': search_events_found,
        'search_events_sorted_by_frequency': sorted(search_events_found.items(), key=lambda x: x[1], reverse=True)
    }
    
    return {
        'analysis_info': {
            'analysis_type': 'users_with_search_terms_percentage', 
            'parameters': {
                'search_terms': search_terms,
                'case_sensitive': case_sensitive
            }
        },
        'summary': summary,
        'results': results
    } 