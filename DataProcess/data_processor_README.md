# Data Processor README

## Overview
The `data_processor.py` module is the core data processing engine that transforms raw CSV event data into structured dictionary-based records optimized for device behavior analysis. It implements filtering, grouping, and dictionary structuring to create a comprehensive database of device event patterns.

## 🎯 Primary Purpose
Transform raw event logs into **device-centric dictionary structures** where each device's complete event history is stored as a JSON object, enabling efficient analysis of user behavior patterns and temporal sequences.

---

## 📋 Module Components

### **1. `load_data()` Function**
**Purpose**: Loads and preprocesses raw CSV data

**Functionality**:
- Loads CSV from specified file path
- Converts timestamps to UTC datetime format (handles mixed formats)
- Cleans country and timezone fields (removes quotes)
- Provides data validation and error handling

**Input**: CSV file with columns: `event`, `timestamp`, `uuid`, `distinct_id`, `country`, `timezone`, `device_id`, `newDevice`

**Output**: Pandas DataFrame with cleaned and formatted data

```python
# Example usage
df = load_data()
# Returns: DataFrame with 8 columns, timestamps as datetime objects
```

---

### **2. `DataProcessor` Class**
**Purpose**: Main processing engine for device-centric data transformation

#### **Constructor: `__init__(db_path="event_analysis.db")`**
- Initializes SQLite database connection
- Sets up SQLAlchemy engine for data persistence
- Creates database file if it doesn't exist

#### **Core Method: `process_device_events_as_dict()`**
**Purpose**: Complete data processing pipeline that creates device event dictionaries

**Processing Steps**:

##### **Step 1: Data Loading**
```python
raw_data = load_data()
# Loads all raw CSV data
```

##### **Step 2: Apply Filters**
```python
# Filter 1: US Users Only
us_data = raw_data[raw_data['country'] == 'United States']

# Filter 2: Remove Click Events
click_devices = us_data[us_data['event'].str.contains('click', case=False)]['device_id'].unique()
filtered_data = us_data[~us_data['device_id'].isin(click_devices)]
```

**Filtering Logic**:
- ✅ **Geographic Filter**: Keep only `country == 'United States'`
- ✅ **Behavioral Filter**: Remove ALL events for devices that have ANY click-type events (case-insensitive)
- 🎯 **Impact**: Ensures clean dataset focused on US non-clicking users

##### **Step 3: Dictionary Structure Creation**
For each device, creates a comprehensive record with:

**Original Data Preservation**:
- All 8 original CSV columns maintained
- First event's data used for device-level attributes

**Event-Time Pairs Dictionary**:
```python
event_time_pair = {
    'event': row['event'],                    # Event name
    'timestamp': row['timestamp'].isoformat(), # ISO timestamp
    'sequence': len(event_time_pairs) + 1,    # Order number
    'hour': row['timestamp'].hour,            # Hour (0-23)
    'day_of_week': row['timestamp'].day_name(), # Day name
    'date': row['timestamp'].date().isoformat() # Date string
}
```

**Computed Analytics**:
- `total_events`: Count of events per device
- `first_event_time`: Earliest timestamp
- `last_event_time`: Latest timestamp  
- `event_types`: Unique event types as JSON array
- `time_span_hours`: Duration between first and last event

##### **Step 4: Database Persistence**
```python
device_df.to_sql('device_event_dictionaries', engine, if_exists='replace')
```

---

### **3. `_create_dict_summary()` Method**
**Purpose**: Generate comprehensive summary statistics

**Metrics Calculated**:

#### **Data Volume Metrics**:
- `total_raw_rows`: Original dataset size
- `total_us_rows`: After geographic filtering
- `devices_with_clicks`: Devices removed by click filter
- `final_unique_devices`: Final processed device count

#### **Event Statistics**:
- `total_events_processed`: Sum of all events
- `unique_event_types`: Count of distinct event types
- `avg_events_per_device`: Mean events per device
- `avg_time_span_hours`: Mean duration of device activity

#### **Device Distribution**:
- `single_event_devices`: Devices with only 1 event
- `multi_event_devices`: Devices with 2+ events
- `highly_active_devices`: Devices with 10+ events

**Output**: Summary saved to `device_dict_summary` table

---

### **4. `get_database_info()` Method**
**Purpose**: Database inspection and reporting

**Functionality**:
- Lists all created tables
- Shows row counts for each table
- Provides database overview for verification

---

## 🗄️ Database Schema

### **Primary Table: `device_event_dictionaries`**

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `device_id` | String | Unique device identifier | "device_12345" |
| `event` | String | First event name | "page_view" |
| `timestamp` | String | First event timestamp (ISO) | "2023-01-01T12:00:00+00:00" |
| `uuid` | String | User UUID from original data | "550e8400-e29b-..." |
| `distinct_id` | String | Analytics tracking ID | "user_analytics_123" |
| `country` | String | User country (always "United States") | "United States" |
| `timezone` | String | User timezone | "America/New_York" |
| `newDevice` | Boolean/String | New device flag | true |
| **`event_time_pairs`** | **JSON String** | **Complete event history** | **[{"event": "page_view", ...}, ...]** |
| `total_events` | Integer | Event count for device | 5 |
| `first_event_time` | String | Earliest timestamp | "2023-01-01T12:00:00+00:00" |
| `last_event_time` | String | Latest timestamp | "2023-01-05T18:30:00+00:00" |
| `event_types` | JSON String | All unique event types this device performed | ["page_view", "sign_up"] |
| `time_span_hours` | Float | Value: (max_timestamp - min_timestamp).total_seconds() / 3600 (float) Hours between first and last event (device engagement duration) | 96.5 |

### **Summary Table: `device_dict_summary`**

| Column | Type | Description |
|--------|------|-------------|
| `metric` | String | Metric name |
| `value` | Mixed | Metric value |
| `category` | String | Metric category |

---

## 🔄 Data Flow Architecture

```
Raw CSV Data
     ↓
[Load & Clean]
     ↓
Apply Geographic Filter (US only)
     ↓
Apply Behavioral Filter (Remove click devices)
     ↓
Group by device_id
     ↓
Create Event-Time Dictionaries
     ↓
Add Computed Analytics
     ↓
Save to Database
     ↓
Generate Summary Statistics
```

---

## 💡 Key Design Decisions

### **1. Device-Centric Approach**
- **Why**: Enables user journey analysis
- **How**: Group all events by `device_id`
- **Benefit**: Complete behavioral picture per device

### **2. Dictionary Structure for Events**
- **Why**: Flexible, queryable event storage
- **How**: JSON arrays with structured event objects
- **Benefit**: Preserves sequence, timing, and context

### **3. Dual Data Strategy**
- **Why**: Balance between raw preservation and analytics
- **How**: Keep original columns + add computed fields
- **Benefit**: Both detailed analysis and quick queries

### **4. Comprehensive Filtering**
- **Why**: Data quality and focus
- **How**: Geographic + behavioral filters
- **Benefit**: Clean, relevant dataset

---

## 🚀 Usage Examples

### **Basic Usage**
```python
from DataProcess.data_processor import DataProcessor

# Initialize processor
processor = DataProcessor()

# Run complete processing pipeline
result = processor.process_device_events_as_dict()

# Check database
processor.get_database_info()
```

### **Accessing Event-Time Pairs**
```python
import json
import pandas as pd
from sqlalchemy import create_engine

# Connect to database
engine = create_engine('sqlite:///event_analysis.db')
df = pd.read_sql('SELECT * FROM device_event_dictionaries', engine)

# Parse event-time pairs for a device
device_events = json.loads(df.iloc[0]['event_time_pairs'])
for event in device_events:
    print(f"Event: {event['event']} at {event['timestamp']}")
```

### **Analytics Queries**
```python
# Devices with most events
top_devices = df.nlargest(10, 'total_events')[['device_id', 'total_events']]

# Average session duration
avg_duration = df['time_span_hours'].mean()

# Event type distribution
all_event_types = []
for types_json in df['event_types']:
    all_event_types.extend(json.loads(types_json))
```

---

## ⚙️ Configuration

### **File Path Configuration**
Update the file path in `load_data()` function:
```python
file_path = "/path/to/your/data.csv"
```

### **Database Configuration**
Change database location:
```python
processor = DataProcessor(db_path="custom_database.db")
```

### **Filter Customization**
Modify filters in `process_device_events_as_dict()`:
```python
# Example: Change country filter
us_data = raw_data[raw_data['country'] == 'Canada']

# Example: Change click filter pattern
click_devices = us_data[us_data['event'].str.contains('purchase', case=False)]['device_id'].unique()
```

---

## 📊 Performance Characteristics

### **Memory Usage**
- **Efficient**: Processes data in chunks by device
- **Scalable**: Uses pandas GroupBy for large datasets
- **JSON Storage**: Compressed event histories

### **Processing Speed**
- **Fast Filtering**: Vectorized pandas operations
- **Optimized Grouping**: Single-pass device grouping
- **Batch Database Writes**: Bulk inserts for performance

### **Database Size**
- **Compact**: One row per device (vs. one row per event)
- **Rich**: Complete event history preserved
- **Indexed**: Device ID as primary key for fast queries

---

## 🛠️ Troubleshooting

### **Common Issues**

**1. File Not Found Error**
```
Solution: Update file_path in load_data() function
```

**2. Timestamp Parsing Errors**
```
Solution: Check timestamp format in CSV, adjust pd.to_datetime() parameters
```

**3. Memory Issues with Large Datasets**
```
Solution: Process in chunks or increase system memory
```

**4. Database Lock Errors**
```
Solution: Close existing database connections
```

### **Validation Checks**
- Ensure CSV has all required columns
- Verify timestamp format consistency
- Check for sufficient disk space
- Confirm database write permissions

---

## 🔮 Future Enhancements

### **Potential Improvements**
1. **Batch Processing**: Handle very large files in chunks
2. **Multiple File Support**: Process multiple CSV files
3. **Custom Filters**: Configurable filtering rules
4. **Data Validation**: Enhanced error checking
5. **Export Options**: Multiple output formats
6. **Incremental Updates**: Append new data to existing database

### **Integration Opportunities**
1. **Analytics Dashboard**: Real-time visualization
2. **Machine Learning**: Feature engineering for ML models
3. **API Layer**: REST API for data access
4. **Scheduling**: Automated periodic processing

---

## 📝 Summary

The `data_processor.py` module transforms raw event logs into a sophisticated, device-centric database optimized for behavioral analysis. By combining comprehensive filtering, dictionary-based event storage, and rich analytics, it provides both the raw detail needed for deep analysis and the summary statistics required for quick insights.

**Key Strengths**:
- ✅ Preserves complete event histories
- ✅ Maintains all original data
- ✅ Provides flexible JSON structures  
- ✅ Generates comprehensive analytics
- ✅ Optimizes for device-centric analysis
- ✅ Ensures data quality through filtering

This design enables sophisticated user journey analysis, temporal pattern discovery, and behavioral segmentation while maintaining data integrity and processing efficiency. 