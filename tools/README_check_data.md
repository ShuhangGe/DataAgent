# User Behavior Analysis Tools Library

## Overview

The `check_data.py` module is a comprehensive user behavior analysis tools library designed to analyze device event data from processed event dictionaries. It provides powerful functions for understanding user retention patterns, event statistics, and behavioral segmentation.

## 🎯 Core Capabilities

### **1. User Retention Analysis**
- **Day 1 Returning Users**: Identify users who returned the day after their first visit
- **Non-Returning Users**: Analyze users who didn't return on day 2
- **Event Sequence Analysis**: Track the exact sequence of events users performed

### **2. Event Analytics**
- **Event Statistics**: Get user counts and time patterns for each event type
- **Event Popularity Ranking**: Sort events by number of unique users
- **Specific Event Filtering**: Find users who performed specific events

### **3. User Segmentation**
- **Location-Based Filtering**: Filter users by country and timezone
- **Activity-Based Filtering**: Segment users by total event count
- **Device Information Retrieval**: Get complete user profiles by device ID

### **4. Cross-Analysis**
- **Multi-Function Analysis**: Combine different analyses in a single run
- **Comparative Studies**: Automatic comparison between user segments
- **Behavioral Pattern Discovery**: Identify differences between user groups

---

## 📋 Installation & Setup

### **Prerequisites**
```bash
pip install pandas sqlite3 argparse
```

### **Database Requirements**
- SQLite database with `device_event_dictionaries` table
- Required columns: `device_id`, `event_time_pairs`, `country`, `timezone`, `total_events`, etc.

---

## 🚀 Usage

### **Command Line Interface**

#### **Basic Usage**
```bash
# Run all analyses with default database
python tools/check_data.py

# Specify custom database
python tools/check_data.py --db_path /path/to/database.db
python tools/check_data.py -d database.db
```

#### **Single Function Analysis**
```bash
# Analyze returning users only
python tools/check_data.py -t returning

# Analyze event statistics only
python tools/check_data.py -t event_stats

# Filter devices by event count
python tools/check_data.py -t event_filter
```

#### **Multiple Function Analysis**
```bash
# Combine returning users analysis with event filtering
python tools/check_data.py -t returning event_filter

# Analyze both returning and non-returning users
python tools/check_data.py -t returning non_returning

# Run comprehensive analysis
python tools/check_data.py -t returning non_returning event_stats event_filter
```

#### **Available Test Functions**
- **`returning`**: Day 1 actions of returning users
- **`non_returning`**: Users without day 1 return
- **`location`**: Location-based filtering
- **`device_info`**: Device information retrieval
- **`event_stats`**: Event statistics and popularity
- **`event_filter`**: Event count filtering and analysis
- **`all`**: Run all analyses (default)

### **Programmatic Usage**

#### **Class-Based Usage**
```python
from tools.check_data import UserBehaviorAnalyzer

# Initialize analyzer
analyzer = UserBehaviorAnalyzer("database.db")

# Analyze returning users
returning_users = analyzer.get_day1_actions_of_returning_users()

# Get event statistics
event_stats = analyzer.get_event_user_counts()

# Filter by location
us_users = analyzer.filter_users_by_location(country="United States")

# Filter by event count
active_users = analyzer.filter_devices_by_event_count(min_events=10)
```

#### **Convenience Functions**
```python
from tools.check_data import (
    analyze_day1_returning_users,
    analyze_users_without_day1_return,
    get_event_statistics,
    filter_by_location
)

# Direct function calls
returning_users = analyze_day1_returning_users("database.db")
non_returning_users = analyze_users_without_day1_return("database.db")
event_stats = get_event_statistics("database.db")
us_users = filter_by_location("database.db", country="United States")
```

---

## 📊 Function Reference

### **1. User Retention Analysis**

#### **`get_day1_actions_of_returning_users()`**
**Purpose**: Identify users who returned the day after their first visit

**Returns**:
```python
{
  "device_123": {
    "first_day": "2023-01-01",
    "second_day": "2023-01-02",
    "day1_actions": [...],  # Detailed event objects
    "day1_unique_events": ["app_open_af", "page_view"],
    "event_sequence": ["app_open_af", "page_view", "sign_up"],  # Simple sequence
    "day1_action_count": 3,
    "timezone": "America/New_York",
    "country": "United States"
  }
}
```

#### **`get_users_without_day1_return()`**
**Purpose**: Analyze users who didn't return on day 2

**Returns**:
```python
{
  "device_456": {
    "first_day": "2023-01-01",
    "expected_return_day": "2023-01-02",
    "day1_actions": [...],
    "event_sequence": ["app_open_af", "page_view"],
    "total_active_days": 1,
    "had_later_activity": false,
    "days_between_first_last": 0,
    "timezone": "America/New_York",
    "country": "United States"
  }
}
```

### **2. Event Analytics**

#### **`get_event_user_counts()`**
**Purpose**: Get comprehensive statistics for each event type

**Returns**:
```python
{
  "page_view": {
    "total_occurrences": 1250,
    "unique_user_count": 450,
    "unique_users": ["device_1", "device_2", ...],
    "avg_events_per_user": 2.8,
    "hours_distribution": {9: 120, 10: 95, ...},
    "days_distribution": {"Monday": 180, "Tuesday": 165, ...},
    "user_event_counts": {"device_1": 3, "device_2": 1, ...}
  }
}
```

#### **`get_events_sorted_by_user_count(ascending=False)`**
**Purpose**: Get events ranked by popularity

**Returns**: List of tuples `(event_name, event_stats)` sorted by user count

### **3. User Segmentation**

#### **`filter_users_by_location(country=None, timezone=None)`**
**Purpose**: Filter users by geographic location

**Usage**:
```python
# Filter by country
us_users = analyzer.filter_users_by_location(country="United States")

# Filter by timezone
ny_users = analyzer.filter_users_by_location(timezone="America/New_York")

# Filter by both
ny_us_users = analyzer.filter_users_by_location(
    country="United States", 
    timezone="America/New_York"
)
```

#### **`filter_devices_by_event_count(min_events=1, max_events=None)`**
**Purpose**: Segment users by activity level

**Usage**:
```python
# Highly active users
high_activity = analyzer.filter_devices_by_event_count(min_events=10)

# Moderate activity users
moderate = analyzer.filter_devices_by_event_count(min_events=2, max_events=9)

# Single event users
single_event = analyzer.filter_devices_by_event_count(min_events=1, max_events=1)
```

#### **`filter_devices_by_specific_event(event_name, min_occurrences=1)`**
**Purpose**: Find users who performed specific events

**Usage**:
```python
# Users who signed up
signups = analyzer.filter_devices_by_specific_event("sign_up")

# Users who viewed pages multiple times
frequent_viewers = analyzer.filter_devices_by_specific_event("page_view", min_occurrences=3)
```

#### **`get_user_info_by_device_list(device_ids, include_event_details=True)`**
**Purpose**: Get complete user profiles for specific devices

**Usage**:
```python
device_list = ["device_123", "device_456"]
user_info = analyzer.get_user_info_by_device_list(device_list)
```

---

## 🔗 Cross-Analysis Features

When you combine certain analyses, the tool automatically performs intelligent cross-analysis:

### **Returning Users + Event Filter**
```bash
python tools/check_data.py -t returning event_filter
```
**Provides**:
- Average events per returning user
- High-activity returning users count
- Behavioral pattern analysis

### **Non-Returning + Event Filter**
```bash
python tools/check_data.py -t non_returning event_filter
```
**Provides**:
- Average events per non-returning user
- Activity pattern analysis

### **Complete Comparison**
```bash
python tools/check_data.py -t returning non_returning event_filter
```
**Provides**:
- Direct comparison between returning vs non-returning users
- Average event differences
- Behavioral pattern insights

---

## 📈 Sample Output

### **Command Line Output**
```
📁 Using database: /path/to/database.db
🧪 Running tests: returning, event_filter

============================================================
🧪 TESTING USER BEHAVIOR ANALYSIS TOOLS
============================================================

1️⃣ Testing: Day 1 actions of returning users
🔍 Analyzing day 1 actions of users who returned on day 2...
✅ Found 156 users who returned on day 2

Sample returning user: device_123
   First day: 2023-01-01
   Day 1 actions: ['app_open_af', 'page_view', 'sign_up']
   Action count: 3
   Event sequence: ['app_open_af', 'page_view', 'sign_up']

5️⃣ Testing: Filter devices by event count
🎯 Filtering devices by event count: min=10, max=None
✅ Found 45 devices matching event count criteria
Highly active devices (10+ events): 45

🔗 Cross-Analysis: Returning users + Event filtering
   Returning users average events: 8.5
   High activity returning users (10+ events): 23

✅ All selected tests completed successfully!
📋 Tests run: returning, event_filter
```

### **Exported Files**
The tool automatically exports results to JSON files:
- `returning_users_analysis.json`: Detailed returning user data
- `non_returning_users_analysis.json`: Non-returning user data
- `event_statistics.json`: Comprehensive event statistics

---

## 🎯 Use Cases

### **1. User Retention Analysis**
```python
# Identify what actions lead to user retention
returning_users = analyzer.get_day1_actions_of_returning_users()
non_returning_users = analyzer.get_users_without_day1_return()

# Compare event sequences
returning_sequences = [data['event_sequence'] for data in returning_users.values()]
non_returning_sequences = [data['event_sequence'] for data in non_returning_users.values()]
```

### **2. Feature Usage Analysis**
```python
# Find most popular features
event_stats = analyzer.get_event_user_counts()
sorted_events = analyzer.get_events_sorted_by_user_count(ascending=False)

# Identify power users of specific features
power_users = analyzer.filter_devices_by_specific_event("advanced_feature", min_occurrences=5)
```

### **3. User Segmentation**
```python
# Segment by geography and activity
us_users = analyzer.filter_users_by_location(country="United States")
high_activity_us = analyzer.filter_devices_by_event_count(min_events=10)

# Find overlap
high_activity_us_devices = set(us_users['device_id']) & set(high_activity_us['device_id'])
```

### **4. Conversion Funnel Analysis**
```python
# Track user progression
app_openers = analyzer.filter_devices_by_specific_event("app_open_af")
page_viewers = analyzer.filter_devices_by_specific_event("page_view")
signups = analyzer.filter_devices_by_specific_event("sign_up")

# Calculate conversion rates
open_to_view_rate = len(page_viewers) / len(app_openers)
view_to_signup_rate = len(signups) / len(page_viewers)
```

---

## 🛠️ Advanced Usage

### **Custom Analysis Workflow**
```python
# Initialize analyzer
analyzer = UserBehaviorAnalyzer("database.db")

# Step 1: Get returning users
returning_users = analyzer.get_day1_actions_of_returning_users()

# Step 2: Analyze their event patterns
returning_device_ids = list(returning_users.keys())
returning_profiles = analyzer.get_user_info_by_device_list(returning_device_ids)

# Step 3: Find common patterns
common_sequences = {}
for device_id, data in returning_users.items():
    sequence = tuple(data['event_sequence'])
    common_sequences[sequence] = common_sequences.get(sequence, 0) + 1

# Step 4: Export insights
analyzer.export_analysis_results(common_sequences, "common_patterns.json")
```

### **Batch Processing**
```python
# Process multiple databases
databases = ["db1.db", "db2.db", "db3.db"]
all_results = {}

for db in databases:
    analyzer = UserBehaviorAnalyzer(db)
    results = analyzer.get_day1_actions_of_returning_users()
    all_results[db] = results
```

---

## 📝 Data Structure Reference

### **Event Time Pairs Structure**
```python
{
  "event": "page_view",
  "timestamp": "2023-01-01T12:00:00+00:00",
  "sequence": 1,
  "hour": 12,
  "day_of_week": "Sunday",
  "date": "2023-01-01"
}
```

### **Device Record Structure**
```python
{
  "device_id": "device_123",
  "country": "United States",
  "timezone": "America/New_York",
  "total_events": 5,
  "first_event_time": "2023-01-01T12:00:00+00:00",
  "last_event_time": "2023-01-05T18:30:00+00:00",
  "time_span_hours": 96.5,
  "event_time_pairs": "[{...}, {...}]",  # JSON string
  "event_types": "[\"page_view\", \"sign_up\"]"  # JSON string
}
```

---

## 🔧 Configuration

### **Default Database Path**
Update the default path in the argument parser:
```python
parser.add_argument('--db_path', '-d', 
                    default="/your/custom/path/database.db",
                    help='Path to the SQLite database file')
```

### **Custom Export Paths**
Modify export paths in the main function:
```python
analyzer.export_analysis_results(results, "./custom_path/results.json")
```

---

## 🚨 Error Handling

The library includes comprehensive error handling:
- **Database Connection**: Validates database file existence
- **JSON Parsing**: Handles malformed event_time_pairs data
- **Missing Data**: Gracefully handles missing devices or events
- **Type Validation**: Ensures proper data types for analysis

---

## 🎉 Best Practices

### **1. Performance Optimization**
- Use specific test functions instead of 'all' for faster execution
- Combine related analyses in single runs to share data loading
- Export results for reuse instead of re-running analyses

### **2. Data Quality**
- Validate database schema before running analyses
- Check for sufficient data volume for meaningful insights
- Monitor for JSON parsing errors in event data

### **3. Analysis Workflow**
- Start with broad analyses (event_stats) to understand data
- Use cross-analysis features for deeper insights
- Export intermediate results for further processing

---

## 📞 Support

For issues or questions:
1. Check error messages for specific guidance
2. Validate database schema and data quality
3. Review function documentation and examples
4. Test with smaller datasets first

---

## 🔄 Version History

- **v1.0**: Initial release with basic user retention analysis
- **v1.1**: Added event statistics and popularity ranking
- **v1.2**: Introduced multi-function analysis and cross-analysis
- **v1.3**: Added event sequence tracking and enhanced filtering

---

This comprehensive tools library provides everything needed for sophisticated user behavior analysis, from basic retention metrics to advanced behavioral pattern discovery. The flexible API supports both quick command-line analysis and complex programmatic workflows. 