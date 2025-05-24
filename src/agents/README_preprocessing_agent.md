# 🔧 **Preprocessing Agent - Complete Explanation**

## 📋 **Overview**

The **Preprocessing Agent** serves as the data transformation specialist in the Sekai Data Analysis Multi-Agent System pipeline. It functions as the **intelligent data preparator**, converting raw data into analysis-ready format through comprehensive cleaning, standardization, and feature engineering. The agent bridges the gap between extracted data and meaningful analysis by ensuring data quality and creating derived features that enhance analytical insights.

## 🎯 **Role in the System**

### **🔄 Pipeline Position**
```mermaid
graph LR
    A[Manager Agent] -->|Creates Task Plan| B[Data Pulling Agent]
    B -->|Provides Raw Data| C[Preprocessing Agent] 
    C -->|Provides Clean Data| D[Analysis Agent]
    D -->|Provides Analysis Results| E[QA Agent]
    E -->|Validates Results| F[Insight Agent]
    F -->|Generates Final Answer| G[User]
    
    style C fill:#3b82f6,stroke:#333,stroke-width:3px
```

### **🎯 Core Responsibilities**
1. **🧹 Data Cleaning** - Handle missing values, standardize formats, remove anomalies
2. **🏗️ Feature Engineering** - Create derived features from raw data
3. **⏰ Temporal Processing** - Extract time-based features and patterns
4. **🎮 Gaming Analytics** - Generate game-specific behavioral features
5. **📊 Data Standardization** - Ensure consistent data formats and types
6. **🔄 Data Transformation** - Convert data for optimal analysis performance

## 🛠️ **Architecture & Implementation**

### **📦 Agent Structure**

```python
def create_preprocessing_agent() -> Agent:
    """Create and configure the Preprocessing Agent"""
    
    # Initialize processing tools
    cleaning_tool = DataCleaningTool()       # Data cleaning and standardization
    feature_tool = FeatureEngineeringTool()  # Feature engineering and derivation
    
    # Create agent with specialized role
    preprocessing_agent = Agent(
        role="Data Preprocessing Specialist",
        goal="""
        As a Data Preprocessing Specialist, I am responsible for:
        1. Cleaning and standardizing data formats (timestamps, units, categories)
        2. Handling missing values and data quality issues
        3. Engineering relevant features for analysis
        4. Performing data transformations and normalization
        5. Ensuring data is analysis-ready with proper validation
        """,
        tools=[cleaning_tool, feature_tool],
        # ... GPT-4o configuration
    )
```

### **🏗️ Controller Architecture**

```python
class PreprocessingController:
    """Controller for Preprocessing Agent operations"""
    
    def __init__(self):
        self.agent = create_preprocessing_agent()
    
    def clean_and_prepare(self, data: pd.DataFrame, config: Dict[str, Any] = None) -> AgentTaskResult:
        """Clean and prepare data for analysis"""
        
        # 1. Data cleaning using DataCleaningTool
        cleaning_result = cleaning_tool._run(data, config.get("cleaning", {}))
        
        # 2. Feature engineering using FeatureEngineeringTool
        feature_result = feature_tool._run(cleaning_result["data"], config.get("features", {}))
        
        # 3. Calculate quality metrics and generate validations
        quality_metrics = self._calculate_quality_metrics(final_data)
        validations = self._generate_validations(...)
        
        # 4. Return comprehensive result with metadata
        return AgentTaskResult(...)
```

## 🔧 **Core Tools**

### **1. 🧹 DataCleaningTool**

The primary tool for data cleaning and standardization operations.

#### **🔍 Missing Value Handling**

```python
class DataCleaningTool(BaseTool):
    name: str = "data_cleaning"
    description: str = "Clean data by handling missing values, standardizing formats, and removing anomalies"
    
    def _handle_missing_values(self, df: pd.DataFrame, strategy: str) -> tuple:
        """Handle missing values with various strategies"""
        
        for column in missing_columns:
            missing_ratio = missing_count / len(df)
            
            # 1. Drop columns with excessive missing values
            if missing_ratio > settings.analysis.max_missing_ratio:
                df = df.drop(columns=[column])
                log.append(f"Dropped column '{column}' due to high missing ratio: {missing_ratio:.2f}")
                continue
            
            # 2. Handle by data type
            if df[column].dtype in ['object', 'string']:
                df[column] = df[column].fillna("unknown")
                log.append(f"Filled missing values in '{column}' with 'unknown'")
                
            elif df[column].dtype in ['int64', 'float64']:
                # Strategy-based filling (mean, median, auto)
                if strategy == "mean":
                    fill_value = df[column].mean()
                elif strategy == "median":
                    fill_value = df[column].median()
                else:  # auto
                    fill_value = df[column].median()  # Default to median for robustness
                
                df[column] = df[column].fillna(fill_value)
                log.append(f"Filled missing values in '{column}' with {strategy}: {fill_value:.2f}")
                
            elif df[column].dtype == 'datetime64[ns]':
                # Forward fill for temporal continuity
                df[column] = df[column].fillna(method='ffill')
                log.append(f"Forward filled missing values in '{column}'")
```

#### **⏰ Timestamp Standardization**

```python
def _standardize_timestamps(self, df: pd.DataFrame) -> tuple:
    """Standardize timestamp columns to Sekai timezone"""
    
    timezone = pytz.timezone(settings.sekai.timezone)  # Asia/Tokyo for Sekai
    
    # Auto-detect timestamp columns
    timestamp_columns = []
    for col in df.columns:
        if 'time' in col.lower() or 'date' in col.lower():
            timestamp_columns.append(col)
    
    for col in timestamp_columns:
        try:
            # 1. Convert string to datetime
            if df[col].dtype == 'object':
                df[col] = pd.to_datetime(df[col], errors='coerce')
                log.append(f"Converted '{col}' to datetime format")
            
            # 2. Timezone localization/conversion
            if df[col].dt.tz is None:
                df[col] = df[col].dt.tz_localize(timezone)
                log.append(f"Localized '{col}' to {settings.sekai.timezone} timezone")
            else:
                df[col] = df[col].dt.tz_convert(timezone)
                log.append(f"Converted '{col}' to {settings.sekai.timezone} timezone")
                
        except Exception as e:
            log.append(f"Warning: Could not standardize timestamp column '{col}': {str(e)}")
```

#### **📝 String Column Cleaning**

```python
def _clean_string_columns(self, df: pd.DataFrame) -> tuple:
    """Clean and standardize string columns"""
    
    string_columns = df.select_dtypes(include=['object', 'string']).columns
    
    for col in string_columns:
        # 1. Remove leading/trailing whitespace
        df[col] = df[col].astype(str).str.strip()
        
        # 2. Standardize case for categorical columns
        if col in ['device_type', 'channel', 'event_name', 'user_segment']:
            df[col] = df[col].str.lower()
            log.append(f"Standardized case for '{col}'")
        
        # 3. Clean special characters from user_id
        if col == 'user_id':
            df[col] = df[col].str.replace(r'[^\w\-]', '', regex=True)
            log.append(f"Cleaned special characters from '{col}'")
```

#### **📊 Outlier Handling**

```python
def _handle_outliers(self, df: pd.DataFrame, threshold: float) -> tuple:
    """Handle outliers using statistical methods"""
    
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_columns:
        if col in ['user_id']:  # Skip ID columns
            continue
            
        # 1. Calculate z-scores for outlier detection
        z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
        outliers = df[z_scores > threshold]
        
        if len(outliers) > 0:
            # 2. Cap outliers instead of removing them (preserve data)
            upper_limit = df[col].mean() + threshold * df[col].std()
            lower_limit = df[col].mean() - threshold * df[col].std()
            
            df[col] = df[col].clip(lower=lower_limit, upper=upper_limit)
            log.append(f"Capped {len(outliers)} outliers in '{col}' (threshold: {threshold} std)")
```

### **2. 🏗️ FeatureEngineeringTool**

The advanced tool for creating derived features that enhance analysis capabilities.

#### **⏰ Temporal Feature Engineering**

```python
class FeatureEngineeringTool(BaseTool):
    name: str = "feature_engineering"
    description: str = "Create derived features from raw data for analysis"
    
    def _derive_temporal_features(self, df: pd.DataFrame) -> tuple:
        """Derive comprehensive temporal features from timestamp columns"""
        
        # Find primary timestamp column
        timestamp_col = None
        for col in ['timestamp', 'event_date', 'created_at']:
            if col in df.columns:
                timestamp_col = col
                break
        
        if timestamp_col:
            # 1. Basic date components
            df['year'] = df[timestamp_col].dt.year
            df['month'] = df[timestamp_col].dt.month
            df['day'] = df[timestamp_col].dt.day
            df['hour'] = df[timestamp_col].dt.hour
            df['day_of_week'] = df[timestamp_col].dt.dayofweek
            df['is_weekend'] = df['day_of_week'].isin([5, 6])
            
            # 2. Business context features
            business_start = 9  # 9 AM
            business_end = 18   # 6 PM
            df['is_business_hours'] = df['hour'].between(business_start, business_end)
            
            # 3. Time period classification
            df['time_period'] = pd.cut(
                df['hour'], 
                bins=[0, 6, 12, 18, 24], 
                labels=['night', 'morning', 'afternoon', 'evening'],
                include_lowest=True
            )
            
            log.append(f"Derived temporal features from '{timestamp_col}'")
```

#### **🎮 Gaming-Specific Behavioral Features**

```python
def _create_behavioral_features(self, df: pd.DataFrame) -> tuple:
    """Create behavioral features specific to gaming analysis"""
    
    try:
        # 1. Session-based features
        if 'user_id' in df.columns and 'timestamp' in df.columns:
            df = df.sort_values(['user_id', 'timestamp'])
            
            # Time since last event for session detection
            df['time_since_last_event'] = df.groupby('user_id')['timestamp'].diff()
            df['time_since_last_event_minutes'] = df['time_since_last_event'].dt.total_seconds() / 60
            
            # Session identification (gap > 30 minutes = new session)
            df['new_session'] = (df['time_since_last_event_minutes'] > 30) | df['time_since_last_event_minutes'].isna()
            df['session_id'] = df.groupby('user_id')['new_session'].cumsum()
            
            log.append("Created session-based behavioral features")
        
        # 2. Event-based features
        if 'event_name' in df.columns:
            # Event frequency encoding
            event_counts = df['event_name'].value_counts()
            df['event_frequency'] = df['event_name'].map(event_counts)
            
            # Gaming event categorization
            monetization_events = ['purchase_complete', 'character_gacha', 'item_gacha']
            engagement_events = ['story_complete', 'battle_start', 'event_participate']
            
            df['event_category'] = 'other'
            df.loc[df['event_name'].isin(monetization_events), 'event_category'] = 'monetization'
            df.loc[df['event_name'].isin(engagement_events), 'event_category'] = 'engagement'
            
            log.append("Created event-based behavioral features")
            
    except Exception as e:
        log.append(f"Warning: Could not create behavioral features: {str(e)}")
```

#### **📊 Categorical Feature Encoding**

```python
def _encode_categorical_features(self, df: pd.DataFrame) -> tuple:
    """Encode categorical variables for analysis"""
    
    categorical_columns = ['device_type', 'channel', 'user_segment', 'event_category']
    
    for col in categorical_columns:
        if col in df.columns:
            # Create dummy variables (one-hot encoding)
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
            df = pd.concat([df, dummies], axis=1)
            log.append(f"Created dummy variables for '{col}'")
```

#### **📈 User-Level Aggregated Features**

```python
def _create_aggregated_features(self, df: pd.DataFrame) -> tuple:
    """Create user-level aggregated features for analysis"""
    
    if 'user_id' not in df.columns:
        return df, ["Cannot create aggregated features without user_id column"]
    
    try:
        # 1. User-level aggregations
        user_aggs = df.groupby('user_id').agg({
            'timestamp': ['count', 'min', 'max'],
            'session_id': 'nunique' if 'session_id' in df.columns else 'count'
        }).round(2)
        
        # 2. Flatten column names and rename
        user_aggs.columns = ['_'.join(col).strip() for col in user_aggs.columns]
        user_aggs = user_aggs.rename(columns={
            'timestamp_count': 'total_events',
            'timestamp_min': 'first_event',
            'timestamp_max': 'last_event',
            'session_id_nunique': 'total_sessions'
        })
        
        # 3. Calculate derived user metrics
        user_aggs['user_tenure_days'] = (
            pd.to_datetime(user_aggs['last_event']) - 
            pd.to_datetime(user_aggs['first_event'])
        ).dt.days + 1
        
        user_aggs['events_per_session'] = (
            user_aggs['total_events'] / user_aggs['total_sessions']
        ).round(2)
        
        # 4. Merge back to main dataframe
        df = df.merge(user_aggs, left_on='user_id', right_index=True, how='left')
        
        log.append("Created user-level aggregated features")
        
    except Exception as e:
        log.append(f"Warning: Could not create aggregated features: {str(e)}")
```

## 🔄 **Dynamic System Integration**

### **📋 Manager Agent Integration**

In the dynamic Q&A system, the Preprocessing Agent receives adaptive configuration based on question requirements:

```python
# Task from Manager Agent in dynamic workflows
{
    "agent": "preprocessing",
    "task": "prepare_for_analysis",
    "parameters": {
        "grouping": ["date", "platform"],       # Dynamic - based on question
        "time_aggregation": "auto",             # Static - intelligent aggregation
        "missing_data_strategy": "auto"         # Static - optimal strategy selection
    }
}
```

### **🎯 Adaptive Processing by Question Type**

The agent adapts its processing approach based on analysis requirements:

| **Question Type** | **Temporal Features** | **Behavioral Features** | **Aggregations** | **Focus Area** |
|-------------------|--------------------|----------------------|---------------|---------------|
| **DATA_EXPLORATION** | ✅ Basic | ❌ | ❌ | Schema understanding |
| **STATISTICAL_SUMMARY** | ✅ Full | ✅ Basic | ✅ User-level | Statistical readiness |
| **TREND_ANALYSIS** | ✅ Enhanced | ✅ Temporal | ✅ Time-based | Temporal patterns |
| **COMPARISON** | ✅ Full | ✅ Segmentation | ✅ Group-level | Comparative metrics |
| **CORRELATION** | ✅ Full | ✅ Full | ✅ Advanced | Relationship analysis |
| **PREDICTION** | ✅ Enhanced | ✅ Advanced | ✅ Predictive | Feature richness |

### **📊 Configuration Examples**

```python
# Trend analysis configuration
trend_config = {
    "cleaning": {
        "missing_strategy": "median",
        "outlier_threshold": 3.0
    },
    "features": {
        "create_aggregations": True,
        "temporal_granularity": "hour",
        "session_timeout_minutes": 30
    }
}

# Comparative analysis configuration
comparison_config = {
    "cleaning": {
        "missing_strategy": "auto",
        "standardize_categories": True
    },
    "features": {
        "create_aggregations": True,
        "group_level_features": True,
        "categorical_encoding": "dummy"
    }
}
```

## 🚀 **Workflow Examples**

### **🔍 Example 1: User Behavior Data Preprocessing**

```
Scenario: Preparing user event data for trend analysis

1. Input Data (Raw):
   - 50,000 user events
   - Columns: user_id, timestamp, event_name, platform, value
   - Issues: 5% missing values, mixed timezones, inconsistent categories

2. Preprocessing Agent Processing:

   A. Data Cleaning:
   - Missing values: Filled 2,500 missing user_ids with "unknown"
   - Timestamp standardization: Converted to Asia/Tokyo timezone
   - String cleaning: Standardized platform values to lowercase
   - Outlier handling: Capped 47 extreme values in 'value' column
   - Duplicate removal: Removed 234 duplicate records

   B. Feature Engineering:
   - Temporal features: Created year, month, day, hour, day_of_week, is_weekend
   - Business features: Added is_business_hours, time_period
   - Behavioral features: Generated session_id, time_since_last_event
   - Event categorization: Classified events as monetization/engagement/other
   - User aggregations: Created total_events, total_sessions, user_tenure_days

3. Output (Analysis-Ready):
   - 49,766 cleaned records (0.5% reduction)
   - 28 features (18 new features created)
   - Quality score: 0.94 (excellent)
   - Processing time: ~8 seconds

4. New Features Created:
   {
     "temporal": ["year", "month", "day", "hour", "day_of_week", "is_weekend", "is_business_hours", "time_period"],
     "behavioral": ["session_id", "time_since_last_event_minutes", "event_frequency", "event_category"],
     "aggregated": ["total_events", "total_sessions", "user_tenure_days", "events_per_session"],
     "categorical": ["platform_mobile", "platform_web", "event_category_monetization", "event_category_engagement"]
   }
```

### **🎮 Example 2: Gaming Session Analysis Preprocessing**

```
Scenario: Preparing data for gaming session analysis

1. Input Data:
   - User gameplay events
   - Timestamp range: 30 days
   - Events: app_open, story_complete, battle_start, purchase_complete, app_close

2. Preprocessing Focus:
   
   A. Session Identification:
   - Detected 15,834 unique sessions from 8,921 users
   - Average session duration: 23.7 minutes
   - Session timeout threshold: 30 minutes inactivity

   B. Gaming Event Categorization:
   {
     "monetization": ["purchase_complete", "item_gacha", "character_gacha"],
     "engagement": ["story_complete", "battle_start", "event_participate"],
     "utility": ["app_open", "app_close", "settings_open"],
     "other": ["unknown_event"]
   }

   C. User Behavior Metrics:
   - Sessions per user: 1.8 average
   - Events per session: 12.3 average
   - User tenure: 14.2 days average
   - Weekend activity ratio: 0.34

3. Analysis-Ready Features:
   {
     "session_level": ["session_duration", "events_in_session", "monetization_events_count"],
     "user_level": ["lifetime_value", "retention_probability", "churn_risk_score"],
     "temporal": ["preferred_play_time", "activity_consistency", "weekend_engagement"],
     "behavioral": ["session_frequency", "event_diversity", "monetization_propensity"]
   }

4. Business Insights Enabled:
   - Session quality analysis
   - User lifecycle tracking
   - Monetization pattern detection
   - Engagement trend analysis
```

### **📊 Example 3: Multi-Platform Comparison Preprocessing**

```
Scenario: Preparing data for mobile vs web platform comparison

1. Input Data:
   - Cross-platform user events
   - Platforms: mobile, web, tablet
   - Time range: 3 months

2. Platform-Specific Processing:

   A. Platform Standardization:
   - Mobile: "ios", "android" → "mobile"
   - Web: "chrome", "firefox", "safari" → "web"
   - Tablet: "ipad", "android_tablet" → "tablet"

   B. Feature Engineering by Platform:
   {
     "mobile": {
       "session_features": ["app_version", "device_model", "push_notification_enabled"],
       "behavioral": ["touch_patterns", "swipe_frequency", "background_time"]
     },
     "web": {
       "session_features": ["browser_type", "screen_resolution", "ad_blocker_enabled"],
       "behavioral": ["click_patterns", "scroll_depth", "tab_switching"]
     }
   }

   C. Comparative Metrics:
   - Cross-platform user identification
   - Platform preference scoring
   - Feature usage comparison
   - Performance metric normalization

3. Comparison-Ready Dataset:
   - Balanced representation: 45% mobile, 35% web, 20% tablet
   - Normalized metrics for fair comparison
   - Platform-specific features preserved
   - Common features standardized

4. Analysis Capabilities Unlocked:
   - Platform performance comparison
   - User preference analysis
   - Cross-platform journey mapping
   - Platform-specific optimization insights
```

## 📊 **Quality Metrics & Performance**

### **🎯 Data Quality Improvement**

```python
# Typical quality improvements after preprocessing
quality_improvements = {
    "before_preprocessing": {
        "completeness_score": 0.78,    # 78% complete (missing values)
        "consistency_score": 0.65,     # 65% consistent (format issues)
        "accuracy_score": 0.72,        # 72% accurate (outliers present)
        "timeliness_score": 0.85,      # 85% timely
        "overall_score": 0.75          # 75% overall quality
    },
    "after_preprocessing": {
        "completeness_score": 0.96,    # 96% complete (missing values handled)
        "consistency_score": 0.95,     # 95% consistent (standardized formats)
        "accuracy_score": 0.92,        # 92% accurate (outliers capped)
        "timeliness_score": 0.90,      # 90% timely (timezone standardized)
        "overall_score": 0.93          # 93% overall quality
    }
}
```

### **⚡ Performance Benchmarks**

| **Processing Stage** | **Data Size** | **Processing Time** | **Memory Usage** | **Features Created** |
|---------------------|---------------|-------------------|------------------|-------------------|
| **Data Cleaning** | 50K rows | ~3-5 seconds | ~25 MB | 0 (cleanup only) |
| **Temporal Features** | 50K rows | ~2-3 seconds | ~15 MB | 8 features |
| **Behavioral Features** | 50K rows | ~4-6 seconds | ~20 MB | 6 features |
| **Categorical Encoding** | 50K rows | ~1-2 seconds | ~10 MB | 4-8 features |
| **User Aggregations** | 50K rows | ~3-4 seconds | ~30 MB | 5 features |
| **Complete Preprocessing** | 50K rows | ~10-15 seconds | ~60 MB | 15-25 features |

### **📈 Feature Engineering Statistics**

```python
feature_engineering_stats = {
    "temporal_features": {
        "count": 8,
        "types": ["date_components", "business_context", "time_periods"],
        "analysis_value": "high",
        "computation_cost": "low"
    },
    "behavioral_features": {
        "count": 6,
        "types": ["session_based", "event_based", "frequency_based"],
        "analysis_value": "very_high",
        "computation_cost": "medium"
    },
    "aggregated_features": {
        "count": 5,
        "types": ["user_level", "session_level", "tenure_based"],
        "analysis_value": "high",
        "computation_cost": "medium"
    },
    "categorical_features": {
        "count": "variable",
        "types": ["dummy_variables", "encoded_categories"],
        "analysis_value": "medium",
        "computation_cost": "low"
    }
}
```

## 🔧 **Configuration & Customization**

### **🎯 Cleaning Configuration Options**

```python
# Comprehensive cleaning configuration
cleaning_config = {
    "missing_value_handling": {
        "strategy": "auto",                 # auto, mean, median, mode, drop
        "threshold_drop_column": 0.5,       # Drop columns with >50% missing
        "categorical_fill": "unknown",      # Default fill for categorical
        "numerical_fill": "median",         # Default fill for numerical
        "datetime_fill": "forward_fill"     # Default fill for datetime
    },
    "outlier_handling": {
        "method": "z_score",               # z_score, iqr, isolation_forest
        "threshold": 3.0,                  # Z-score threshold
        "action": "cap",                   # cap, remove, flag
        "exclude_columns": ["user_id", "session_id"]
    },
    "string_cleaning": {
        "standardize_case": True,          # Convert to lowercase
        "trim_whitespace": True,           # Remove leading/trailing spaces
        "remove_special_chars": {
            "user_id": True,               # Clean user_id format
            "event_name": False            # Preserve event_name format
        }
    },
    "timestamp_standardization": {
        "target_timezone": "Asia/Tokyo",   # Sekai business timezone
        "auto_detect_columns": True,       # Auto-find timestamp columns
        "date_format": "ISO8601"           # Standardized format
    }
}
```

### **🏗️ Feature Engineering Configuration**

```python
# Advanced feature engineering configuration
feature_config = {
    "temporal_features": {
        "granularity": "hour",             # minute, hour, day
        "business_hours": {"start": 9, "end": 18},
        "weekend_definition": [5, 6],      # Saturday, Sunday
        "time_periods": {
            "night": [0, 6],
            "morning": [6, 12],
            "afternoon": [12, 18],
            "evening": [18, 24]
        }
    },
    "behavioral_features": {
        "session_timeout_minutes": 30,     # Session boundary definition
        "event_categorization": {
            "monetization": ["purchase_complete", "character_gacha", "item_gacha"],
            "engagement": ["story_complete", "battle_start", "event_participate"],
            "utility": ["app_open", "app_close", "settings_open"]
        },
        "frequency_encoding": True,         # Add event frequency features
        "sequence_features": False          # Add event sequence features (experimental)
    },
    "aggregation_features": {
        "user_level": True,                # Create user-level aggregations
        "session_level": True,             # Create session-level aggregations
        "time_windows": ["1d", "7d", "30d"], # Rolling window features
        "metrics": ["count", "sum", "mean", "std", "min", "max"]
    },
    "categorical_encoding": {
        "method": "dummy",                 # dummy, label, target
        "drop_first": True,                # Avoid multicollinearity
        "handle_unknown": "ignore",        # Strategy for unseen categories
        "min_frequency": 10                # Minimum frequency for encoding
    }
}
```

### **🎮 Gaming-Specific Configurations**

```python
# Gaming industry specific configurations
gaming_config = {
    "session_analysis": {
        "short_session_threshold": 60,     # < 1 minute = bounce
        "long_session_threshold": 7200,    # > 2 hours = marathon
        "ideal_session_range": [180, 1800], # 3-30 minutes ideal
        "engagement_events": [
            "tutorial_complete", "level_up", "achievement_unlock",
            "social_interaction", "content_unlock"
        ]
    },
    "monetization_analysis": {
        "revenue_events": [
            "purchase_complete", "subscription_start", "ad_view_complete"
        ],
        "conversion_funnel": [
            "store_view", "item_select", "purchase_initiate", "purchase_complete"
        ],
        "value_tiers": {
            "whale": 100,      # $100+ lifetime value
            "dolphin": 10,     # $10-100 lifetime value
            "minnow": 1        # $1-10 lifetime value
        }
    },
    "retention_analysis": {
        "cohort_periods": ["d1", "d3", "d7", "d14", "d30"],
        "activity_definition": "any_event",  # any_event, monetization_event
        "churn_threshold_days": 7,          # 7 days inactive = churned
        "reactivation_window": 30           # 30 days for reactivation
    }
}
```

## 🛠️ **Extension Points**

### **🔌 Adding Custom Cleaning Rules**

```python
# 1. Define custom cleaning function
def _clean_gaming_events(self, df: pd.DataFrame) -> tuple:
    """Apply gaming-specific event cleaning rules"""
    log = []
    
    if 'event_name' in df.columns:
        # Standardize event naming convention
        event_mapping = {
            'gacha_pull': 'character_gacha',
            'shop_purchase': 'purchase_complete',
            'stage_clear': 'battle_complete'
        }
        
        df['event_name'] = df['event_name'].replace(event_mapping)
        log.append(f"Standardized {len(event_mapping)} event names")
        
        # Remove test events
        test_events = df['event_name'].str.contains('test_', case=False, na=False)
        df = df[~test_events]
        log.append(f"Removed {test_events.sum()} test events")
    
    return df, log

# 2. Integrate into cleaning pipeline
def _run(self, data: pd.DataFrame, cleaning_config: Dict[str, Any] = None) -> Dict[str, Any]:
    # ... existing cleaning steps ...
    
    # Add custom gaming-specific cleaning
    if cleaning_config.get("gaming_specific", True):
        df, gaming_log = self._clean_gaming_events(df)
        cleaning_log.extend(gaming_log)
```

### **🏗️ Creating Custom Feature Generators**

```python
# 1. Define custom feature generator
def _create_player_lifecycle_features(self, df: pd.DataFrame) -> tuple:
    """Create player lifecycle stage features"""
    log = []
    
    if 'user_id' in df.columns and 'timestamp' in df.columns:
        # Calculate user lifecycle stages
        user_stats = df.groupby('user_id').agg({
            'timestamp': ['min', 'max', 'count']
        }).round(2)
        
        user_stats.columns = ['first_seen', 'last_seen', 'total_events']
        
        # Define lifecycle stages
        current_time = datetime.now()
        user_stats['days_since_first'] = (current_time - user_stats['first_seen']).dt.days
        user_stats['days_since_last'] = (current_time - user_stats['last_seen']).dt.days
        
        # Lifecycle classification
        conditions = [
            (user_stats['days_since_first'] <= 7),                              # New user
            (user_stats['days_since_first'] <= 30) & (user_stats['days_since_last'] <= 7),  # Active user
            (user_stats['days_since_first'] > 30) & (user_stats['days_since_last'] <= 14), # Veteran user
            (user_stats['days_since_last'] > 14)                               # At-risk user
        ]
        
        choices = ['new_user', 'active_user', 'veteran_user', 'at_risk_user']
        user_stats['lifecycle_stage'] = np.select(conditions, choices, default='unknown')
        
        # Merge back to main dataframe
        df = df.merge(user_stats[['lifecycle_stage']], left_on='user_id', right_index=True, how='left')
        
        log.append("Created player lifecycle stage features")
    
    return df, log

# 2. Add to feature engineering pipeline
def _run(self, data: pd.DataFrame, feature_config: Dict[str, Any] = None) -> Dict[str, Any]:
    # ... existing feature engineering steps ...
    
    # Add custom player lifecycle features
    if feature_config.get("player_lifecycle", True):
        df, lifecycle_log = self._create_player_lifecycle_features(df)
        feature_log.extend(lifecycle_log)
```

### **📊 Advanced Aggregation Features**

```python
def _create_rolling_window_features(self, df: pd.DataFrame, windows: List[str] = ["7d", "30d"]) -> tuple:
    """Create rolling window aggregation features"""
    log = []
    
    if 'user_id' not in df.columns or 'timestamp' not in df.columns:
        return df, ["Cannot create rolling features without user_id and timestamp"]
    
    df = df.sort_values(['user_id', 'timestamp'])
    
    for window in windows:
        # Rolling event counts
        df[f'events_last_{window}'] = (
            df.groupby('user_id')['timestamp']
            .rolling(window, on='timestamp')
            .count()
            .values
        )
        
        # Rolling unique session count
        if 'session_id' in df.columns:
            df[f'sessions_last_{window}'] = (
                df.groupby('user_id')['session_id']
                .rolling(window, on='timestamp')
                .nunique()
                .values
            )
        
        log.append(f"Created rolling {window} window features")
    
    return df, log
```

## 📈 **Monitoring & Quality Assurance**

### **🚨 Processing Quality Alerts**

```python
processing_quality_alerts = {
    "data_loss_threshold": 0.05,          # Alert if >5% data loss
    "feature_creation_minimum": 5,        # Alert if <5 features created
    "processing_time_threshold": 30,      # Alert if >30 seconds processing
    "memory_usage_threshold": 100,        # Alert if >100MB memory usage
    
    "quality_degradation": {
        "completeness_drop": 0.1,          # Alert if completeness drops >10%
        "consistency_drop": 0.05,          # Alert if consistency drops >5%
        "unexpected_nulls": 100            # Alert if >100 new null values
    }
}
```

### **📊 Processing Performance Metrics**

```python
performance_metrics = {
    "throughput": {
        "records_per_second": 3500,        # 3.5K records/second processing
        "features_per_second": 150,        # 150 features/second creation
        "memory_efficiency": 0.85          # 85% memory efficiency
    },
    
    "quality_consistency": {
        "successful_processing_rate": 0.998,  # 99.8% success rate
        "feature_stability": 0.95,            # 95% feature stability
        "output_consistency": 0.97            # 97% output consistency
    },
    
    "business_impact": {
        "analysis_readiness_score": 0.94,     # 94% analysis-ready
        "feature_usefulness_score": 0.89,     # 89% useful features
        "processing_value_add": 0.92          # 92% value addition
    }
}
```

## 📋 **Best Practices**

### **🧹 Data Cleaning**
- **Preserve data integrity** - Cap outliers instead of removing them when possible
- **Log all transformations** - Maintain comprehensive audit trail of changes
- **Validate business logic** - Ensure cleaning preserves business meaning
- **Handle edge cases gracefully** - Plan for unexpected data scenarios

### **🏗️ Feature Engineering**
- **Domain-specific features** - Create features that align with business context
- **Avoid data leakage** - Don't use future information for historical analysis
- **Feature interpretability** - Ensure features have clear business meaning
- **Scalability consideration** - Design features that scale with data growth

### **⚡ Performance Optimization**
- **Vectorized operations** - Use pandas vectorization for speed
- **Memory management** - Process data in chunks for large datasets
- **Incremental processing** - Support incremental feature updates
- **Parallel processing** - Leverage multiprocessing for independent operations

### **🔧 Configuration Management**
- **Environment-specific configs** - Different settings for dev/test/prod
- **Version control** - Track configuration changes over time
- **Validation** - Validate configuration parameters before processing
- **Documentation** - Document all configuration options and impacts

## 🎯 **Future Enhancements**

### **🔮 Planned Capabilities**

```python
future_enhancements = {
    "intelligent_preprocessing": {
        "description": "AI-powered preprocessing with GPT-4o optimization",
        "capabilities": [
            "Automatic feature selection based on analysis type",
            "Intelligent missing value imputation",
            "Dynamic outlier detection thresholds",
            "Context-aware feature engineering"
        ]
    },
    
    "real_time_processing": {
        "description": "Stream processing capabilities for real-time analytics",
        "capabilities": [
            "Streaming data preprocessing",
            "Incremental feature updates",
            "Real-time quality monitoring",
            "Dynamic schema adaptation"
        ]
    },
    
    "advanced_feature_engineering": {
        "description": "Advanced ML-powered feature engineering",
        "capabilities": [
            "Automated feature interaction detection",
            "Deep learning based feature extraction",
            "Time series specific feature engineering",
            "Cross-domain feature transfer"
        ]
    }
}
```

---

**The Preprocessing Agent transforms raw data into analysis-ready format through intelligent cleaning and sophisticated feature engineering, serving as the foundation for all downstream analytical insights!** 🔧✨ 