# User Behavior Analysis Tools Library

## Overview

This library provides composable building blocks for analyzing user behavior patterns from device event data stored in SQLite databases. All functions follow a **two-step architecture** for efficient and flexible analysis workflows.

## Two-Step Architecture

### **Step 1: Filter Functions** (Take original database as input)
These functions query the database and return filtered user data dictionaries:
- `analyze_day1_returning_users(db_path)`
- `analyze_users_without_day1_return(db_path)`
- `filter_by_location(db_path, country, timezone)`
- `get_users_by_device_ids(db_path, device_ids)`

### **Step 2: Analysis Functions** (Take filtered data as input)
These functions work on filtered data dictionaries for analysis:
- `get_event_statistics(filtered_data)`
- `filter_devices_by_events(filtered_data, min_events, max_events)`
- `plot_event_per_user(filtered_data, ...)`
- `find_frequent_events(filtered_data, ...)`
- `get_events_by_popularity(filtered_data, ascending)`

## Quick Start

```python
from Tools.Check_Data import *

# Step 1: Filter users
returning_users = analyze_day1_returning_users(db_path)

# Step 2: Analyze filtered data
stats = get_event_statistics(returning_users)
plot_event_per_user(returning_users)
sequences = find_frequent_events(returning_users)
```

---

## Function Reference

### 🔍 STEP 1 - FILTER FUNCTIONS

#### `analyze_day1_returning_users(db_path: str) -> Dict`
Find users who returned to the app on day 2, with both day 1 and day 2 events.
- **Input**: Database path
- **Output**: `{device_id: {day1_actions, day2_actions, combined_sequence, event_counts, timing}}`
- **Use case**: Retention analysis, onboarding effectiveness, day 1→2 transition patterns

#### `analyze_users_without_day1_return(db_path: str) -> Dict`
Find users who didn't return on day 2.
- **Input**: Database path
- **Output**: `{device_id: {first_day, day1_actions, later_activity, total_events}}`
- **Use case**: Churn analysis, identify drop-off patterns

#### `filter_by_location(db_path: str, country: str = None, timezone: str = None) -> Dict`
Filter users by country and/or timezone.
- **Input**: Database path, country (optional), timezone (optional)
- **Output**: Dictionary of user records matching location criteria
- **Use case**: Geographic analysis, regional behavior patterns

#### `get_users_by_device_ids(db_path: str, device_ids: List[str]) -> Dict`
Get detailed information for specific device IDs.
- **Input**: Database path, list of device IDs
- **Output**: `{device_id: {complete_user_data, event_details, time_analysis}}`
- **Use case**: Deep dive into specific users, cohort analysis

### 📊 STEP 2 - ANALYSIS FUNCTIONS

#### `get_event_statistics(filtered_data: Dict) -> Dict`
Analyze event types and usage patterns from filtered data.
- **Input**: Filtered user data dictionary
- **Output**: `{event_name: {total_occurrences, unique_users, time_distribution}}`
- **Use case**: Feature usage analysis, event popularity ranking

#### `filter_devices_by_events(filtered_data: Dict, min_events: int = 1, max_events: int = None, device_ids: List[str] = None) -> Dict`
Further filter devices by event count from already filtered data.
- **Input**: Filtered data, event count range, optional device ID filter
- **Output**: Further filtered user data dictionary
- **Use case**: Activity segmentation, engagement level analysis

#### `plot_event_per_user(filtered_data: Dict, device_ids: List[str] = None, show_plots: bool = True, save_plots: bool = True, plot_dir: str = './plots') -> Dict`
Create visualizations of event number distribution from filtered data.
- **Input**: Filtered data, optional device filter, plot options
- **Output**: Statistics + histogram visualization files
- **Use case**: Visual analysis, presentation materials, distribution insights

#### `plot_event_counts(filtered_data: Dict, top_n: int = 20, show_plots: bool = True, save_plots: bool = True, plot_dir: str = './plots') -> Dict`
Create horizontal bar chart showing count of each unique event type from filtered data.
- **Input**: Filtered data, number of top events to show, plot options
- **Output**: Horizontal bar chart with event counts + comprehensive statistics
- **Use case**: Event popularity analysis, feature usage tracking, identifying most common user actions

#### `find_frequent_events(filtered_data: Dict, min_sequence_length: int = 2, max_sequence_length: int = 5, min_users: int = 2, top_n: int = 10) -> Dict`
Discover common event sequences and user journey patterns from filtered data.
- **Input**: Filtered data, sequence parameters (length, user threshold)
- **Output**: `{sequence: {user_count, occurrences, position_analysis, examples}}`
- **Use case**: User journey analysis, workflow optimization, pattern discovery

#### `get_events_by_popularity(filtered_data: Dict, ascending: bool = False) -> List[Tuple[str, Dict]]`
Get events sorted by popularity from filtered data.
- **Input**: Filtered data, sort order
- **Output**: List of tuples (event_name, event_stats) sorted by user count
- **Use case**: Feature ranking, popularity analysis

### 🔧 UTILITY FUNCTIONS

#### `extract_device_ids(analysis_result: Dict) -> List[str]`
Extract device IDs from any standardized analysis result.
- **Purpose**: Chain analyses, filter subsequent operations
- **Example**: `device_ids = extract_device_ids(returning_users)`

#### `extract_user_data(analysis_result: Dict) -> Dict`
Get the actual data from standardized results (removes metadata).
- **Purpose**: Access raw results without analysis_info wrapper
- **Example**: `data = extract_user_data(analysis_result)`

#### `export_results(data, filename: str, format_type: str = 'json')`
Export analysis results or filtered data to file (handles both standardized and raw data).
- **Purpose**: Universal export function for any data type
- **Example**: `export_results(returning_users, 'data_export', 'csv')`
- **Formats**: 'json' (full data + metadata) or 'csv' (tabular format)

#### `export_filtered_data(filtered_data: Dict, filename: str, format_type: str = 'json')`
Convenience function to export filtered data from Step 1 filter functions.
- **Purpose**: Export raw filtered data with automatic formatting
- **Example**: `export_filtered_data(returning_users, 'returning_users', 'json')`

#### `export_analysis_results(analysis_result: Dict, filename: str, format_type: str = 'json')`
Convenience function to export standardized analysis results from Step 2 functions.
- **Purpose**: Export analysis results with metadata
- **Example**: `export_analysis_results(stats, 'event_analysis', 'csv')`

---

## Workflow Examples

### 🔗 Workflow 1: Analyze returning users' event patterns
```python
# Step 1: Filter - Get returning users
returning_users = analyze_day1_returning_users(db_path)

# Step 2: Analysis - Get their event statistics and visualizations
stats = get_event_statistics(returning_users)
plot_event_per_user(returning_users)
event_counts = plot_event_counts(returning_users, top_n=15)
sequences = find_frequent_events(returning_users)
```

### 🔗 Workflow 2: Comprehensive event analysis
```python
# Step 1: Filter - Get users without day 1 return
non_returning_users = analyze_users_without_day1_return(db_path)

# Step 2: Analysis - Compare event patterns
# Event distribution analysis
plot_event_per_user(non_returning_users)

# Event popularity analysis  
plot_event_counts(non_returning_users, top_n=20)

# Statistical comparison
stats = get_event_statistics(non_returning_users)
```

### 🔗 Workflow 3: Location-based analysis with event filtering
```python
# Step 1: Filter - Get US users
us_users = filter_by_location(db_path, country='United States')

# Step 2: Analysis - Filter by event count, then analyze
active_users = filter_devices_by_events(us_users, min_events=5)
us_user_data = extract_user_data(active_users)
popularity = get_events_by_popularity(us_user_data)
```

### 🔗 Workflow 4: Specific device analysis
```python
# Step 1: Filter - Get specific devices
device_list = ['device_1', 'device_2', 'device_3']
specific_users = get_users_by_device_ids(db_path, device_list)

# Step 2: Analysis - Analyze their behavior
stats = get_event_statistics(specific_users)
export_results(stats, 'specific_analysis.json')
```

### 🔗 Workflow 5: Export and save analysis pipeline
```python
# Step 1: Filter - Get returning users
returning_users = analyze_day1_returning_users(db_path)

# Export filtered data
export_filtered_data(returning_users, 'returning_users_raw_data', 'csv')

# Step 2: Analysis - Multiple analyses with exports
stats = get_event_statistics(returning_users)
export_analysis_results(stats, 'returning_users_stats', 'json')

event_counts = plot_event_counts(returning_users, top_n=20)
export_results(event_counts, 'returning_users_event_counts', 'csv')

sequences = find_frequent_events(returning_users)
export_results(sequences, 'returning_users_sequences', 'json')
```

### 🔗 Workflow 6: Combine multiple filters
```python
# Step 1: Multiple filters
returning = analyze_day1_returning_users(db_path)
device_ids = extract_device_ids({'results': returning})
us_returning = filter_by_location(db_path, country='United States')
us_returning_devices = {k: v for k, v in us_returning.items() if k in device_ids}

# Step 2: Analysis
final_stats = get_event_statistics(us_returning_devices)
```

### 🔗 Workflow 7: Event distribution of returning users
```python
# Chain filter and analysis functions
returning_users = analyze_day1_returning_users(db_path)
device_ids = extract_device_ids({'results': returning_users})
plot_event_per_user({'results': returning_users}, device_ids=device_ids)
```

### 🔗 Workflow 8: High-activity user deep dive
```python
# Multi-step filtering and analysis
us_users = filter_by_location(db_path, country='United States')
high_activity = filter_devices_by_events(us_users, min_events=10)
device_ids = extract_device_ids(high_activity)
user_details = get_users_by_device_ids(db_path, device_ids)
export_results(user_details, 'high_activity_users.csv', 'csv')
```

### 🔗 Workflow 9: Geographic retention analysis
```python
# Calculate retention rate for specific regions
us_users = filter_by_location(db_path, country='United States')
us_device_ids = extract_device_ids({'results': us_users})
us_returning = analyze_day1_returning_users(db_path)
us_returning_ids = extract_device_ids({'results': us_returning})
retention_rate = len(set(us_returning_ids) & set(us_device_ids)) / len(us_device_ids)
print(f"US retention rate: {retention_rate:.2%}")
```

---

## Standardized Output Format

All analysis functions return consistent format:
```python
{
    'analysis_info': {
        'analysis_type': 'function_name',
        'timestamp': 'ISO_datetime',
        'total_records': count,
        'database_path': 'path'
    },
    'summary': {
        'key_metrics': 'function_specific_summary',
        'parameters': 'analysis_parameters',
        'insights': 'high_level_findings'
    },
    'results': {
        'actual_analysis_data': 'varies_by_function'
    }
}
```

---

## Quick Reference Examples

### 📈 Basic retention analysis:
```python
returning = analyze_day1_returning_users('path/to/db')
export_results({'results': returning}, 'retention_analysis.csv', 'csv')
```

### 💾 Export filtered data:
```python
returning_users = analyze_day1_returning_users('path/to/db')
export_filtered_data(returning_users, 'returning_users_data', 'json')
export_filtered_data(returning_users, 'returning_users_data', 'csv')
```

### 📊 Visual engagement analysis:
```python
us_users = filter_by_location('path/to/db', country='United States')
plot_event_per_user(us_users, save_plots=True, plot_dir='./plots')
```

### 🔍 Deep dive into specific users:
```python
us_users = filter_by_location('path/to/db', country='United States')
high_activity = filter_devices_by_events(us_users, min_events=20)
details = get_users_by_device_ids('path/to/db', extract_device_ids(high_activity))
```

### 🌍 Geographic analysis:
```python
us_users = filter_by_location('path/to/db', country='United States')
plot_event_per_user(us_users, plot_dir='./us_analysis')
```

---

## Architecture Benefits

1. **Efficiency**: Filter functions query database once, analysis functions work on in-memory data
2. **Composability**: Functions can be chained easily without repeated database queries
3. **Flexibility**: Mix and match different filters with different analysis approaches
4. **Consistency**: All functions follow the same input/output patterns
5. **Performance**: Avoid repeated expensive database operations

## Usage Tips

- Always use filter functions first to get your target user set
- Analysis functions can be called multiple times on the same filtered data
- Use utility functions to extract data or device IDs for further processing
- Export results at any step for sharing or further analysis
- Combine multiple filter results using set operations on device IDs