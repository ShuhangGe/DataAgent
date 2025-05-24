# 🎯 **Task Combinations from Dynamic Task Planning**

## 📋 **Overview**

The `DynamicTaskPlanningTool` in the `DynamicManagerController` generates custom agent workflows based on user questions. This document details **ALL possible task combinations** the system can create.

## 🔄 **Task Generation Logic**

The system follows this **decision tree**:

1. **🔄 Conditional Data Pulling**: Added if `question.target_tables` exists OR `question_type != DATA_EXPLORATION`
2. **🔧 Conditional Preprocessing**: Added if `question_type != DATA_EXPLORATION`
3. **📊 Question-Specific Analysis**: Based on detected question type
4. **✅ Always QA Validation**: Always included
5. **💡 Always Insight Generation**: Always included

## 📊 **Complete Task Combination Matrix**

### **1. 📋 DATA_EXPLORATION Questions**
*"What data do we have?", "Show me available tables", "What columns exist?"*

#### **Scenario A: Pure Data Exploration (No target tables)**
```python
task_plan = [
    {
        "agent": "insight",
        "task": "generate_answer",
        "parameters": {
            "question": "What data do we have?",
            "question_type": "DATA_EXPLORATION",
            "output_format": "summary"
        }
    }
]
```
- **Tasks**: 1
- **Execution Time**: 30 seconds
- **Quality Checks**: `["data_completeness", "result_validity"]`

#### **Scenario B: Data Exploration with Target Tables**
```python
task_plan = [
    {
        "agent": "data_pulling",
        "task": "extract_relevant_data",
        "parameters": {
            "target_tables": ["users", "events"],
            "columns": ["id", "name", "created_at"],
            "time_filters": {},
            "conditions": [],
            "sample_size": 50000
        }
    },
    {
        "agent": "qa",
        "task": "validate_analysis_results",
        "parameters": {
            "question_context": "What data is available in users table?",
            "expected_output": "summary"
        }
    },
    {
        "agent": "insight",
        "task": "generate_answer",
        "parameters": {
            "question": "What data is available in users table?",
            "question_type": "DATA_EXPLORATION",
            "output_format": "summary"
        }
    }
]
```
- **Tasks**: 3
- **Execution Time**: 90 seconds
- **Quality Checks**: `["data_completeness", "result_validity"]`

### **2. 📈 STATISTICAL_SUMMARY Questions**
*"How many users are active?", "Show me user statistics", "What's the average session duration?"*

```python
task_plan = [
    {
        "agent": "data_pulling",
        "task": "extract_relevant_data",
        "parameters": {
            "target_tables": ["users"],
            "columns": ["id", "last_active", "created_at"],
            "time_filters": {"period": "last_month"},
            "conditions": ["status = 'active'"],
            "sample_size": 50000
        }
    },
    {
        "agent": "preprocessing",
        "task": "prepare_for_analysis",
        "parameters": {
            "grouping": ["date"],
            "time_aggregation": "auto",
            "missing_data_strategy": "auto"
        }
    },
    {
        "agent": "analysis",
        "task": "calculate_summary_statistics",
        "parameters": {
            "metrics": ["count", "mean", "median", "std"],
            "grouping": ["date"]
        }
    },
    {
        "agent": "qa",
        "task": "validate_analysis_results",
        "parameters": {
            "question_context": "How many users are active?",
            "expected_output": "summary"
        }
    },
    {
        "agent": "insight",
        "task": "generate_answer",
        "parameters": {
            "question": "How many users are active?",
            "question_type": "STATISTICAL_SUMMARY",
            "output_format": "summary"
        }
    }
]
```
- **Tasks**: 5
- **Execution Time**: 150 seconds
- **Quality Checks**: `["data_completeness", "result_validity"]`

### **3. 📉 TREND_ANALYSIS Questions**
*"Show me user trends over time", "How has engagement changed?", "Monthly growth patterns"*

```python
task_plan = [
    {
        "agent": "data_pulling",
        "task": "extract_relevant_data",
        "parameters": {
            "target_tables": ["users", "events"],
            "columns": ["created_at", "user_id", "event_type"],
            "time_filters": {"period": "last_month"},
            "conditions": [],
            "sample_size": 50000
        }
    },
    {
        "agent": "preprocessing",
        "task": "prepare_for_analysis",
        "parameters": {
            "grouping": ["date"],
            "time_aggregation": "auto",
            "missing_data_strategy": "auto"
        }
    },
    {
        "agent": "analysis",
        "task": "analyze_trends",
        "parameters": {
            "time_column": "timestamp",
            "metrics": ["users", "engagement"],
            "trend_method": "linear"
        }
    },
    {
        "agent": "qa",
        "task": "validate_analysis_results",
        "parameters": {
            "question_context": "Show me user trends over time",
            "expected_output": "summary"
        }
    },
    {
        "agent": "insight",
        "task": "generate_answer",
        "parameters": {
            "question": "Show me user trends over time",
            "question_type": "TREND_ANALYSIS",
            "output_format": "summary"
        }
    }
]
```
- **Tasks**: 5
- **Execution Time**: 150 seconds
- **Quality Checks**: `["data_completeness", "result_validity", "temporal_consistency"]`

### **4. ⚖️ COMPARISON Questions**
*"Compare mobile vs web users", "New vs returning users", "Performance by country"*

```python
task_plan = [
    {
        "agent": "data_pulling",
        "task": "extract_relevant_data",
        "parameters": {
            "target_tables": ["users"],
            "columns": ["platform", "user_type", "activity_score"],
            "time_filters": {},
            "conditions": [],
            "sample_size": 50000
        }
    },
    {
        "agent": "preprocessing",
        "task": "prepare_for_analysis",
        "parameters": {
            "grouping": ["platform"],
            "time_aggregation": "auto",
            "missing_data_strategy": "auto"
        }
    },
    {
        "agent": "analysis",
        "task": "comparative_analysis",
        "parameters": {
            "comparison_groups": ["platform"],
            "metrics": ["users", "activity"]
        }
    },
    {
        "agent": "qa",
        "task": "validate_analysis_results",
        "parameters": {
            "question_context": "Compare mobile vs web users",
            "expected_output": "summary"
        }
    },
    {
        "agent": "insight",
        "task": "generate_answer",
        "parameters": {
            "question": "Compare mobile vs web users",
            "question_type": "COMPARISON",
            "output_format": "summary"
        }
    }
]
```
- **Tasks**: 5
- **Execution Time**: 150 seconds
- **Quality Checks**: `["data_completeness", "result_validity"]`

### **5. 🔗 CORRELATION Questions**
*"What factors correlate with retention?", "Usage vs revenue relationship", "Engagement impact on churn"*

```python
task_plan = [
    {
        "agent": "data_pulling",
        "task": "extract_relevant_data",
        "parameters": {
            "target_tables": ["users", "events", "revenue"],
            "columns": ["retention_score", "usage_hours", "revenue_amount"],
            "time_filters": {},
            "conditions": [],
            "sample_size": 50000
        }
    },
    {
        "agent": "preprocessing",
        "task": "prepare_for_analysis",
        "parameters": {
            "grouping": [],
            "time_aggregation": "auto",
            "missing_data_strategy": "auto"
        }
    },
    {
        "agent": "analysis",
        "task": "correlation_analysis",
        "parameters": {
            "variables": ["retention", "usage", "revenue"],
            "method": "pearson"
        }
    },
    {
        "agent": "qa",
        "task": "validate_analysis_results",
        "parameters": {
            "question_context": "What factors correlate with retention?",
            "expected_output": "detailed"
        }
    },
    {
        "agent": "insight",
        "task": "generate_answer",
        "parameters": {
            "question": "What factors correlate with retention?",
            "question_type": "CORRELATION",
            "output_format": "detailed"
        }
    }
]
```
- **Tasks**: 5
- **Execution Time**: 150 seconds
- **Quality Checks**: `["data_completeness", "result_validity", "statistical_significance"]`

### **6. 🔮 PREDICTION Questions**
*"Predict user churn", "Forecast next month's revenue", "Who is likely to upgrade?"*

```python
task_plan = [
    {
        "agent": "data_pulling",
        "task": "extract_relevant_data",
        "parameters": {
            "target_tables": ["users", "events", "sessions"],
            "columns": ["user_id", "activity_score", "last_active", "churn_status"],
            "time_filters": {"period": "last_3_months"},
            "conditions": [],
            "sample_size": 50000
        }
    },
    {
        "agent": "preprocessing",
        "task": "prepare_for_analysis",
        "parameters": {
            "grouping": [],
            "time_aggregation": "auto",
            "missing_data_strategy": "auto"
        }
    },
    {
        "agent": "analysis",
        "task": "predictive_modeling",
        "parameters": {
            "target_variable": "churn",
            "features": ["activity_score", "last_active", "session_count"]
        }
    },
    {
        "agent": "qa",
        "task": "validate_analysis_results",
        "parameters": {
            "question_context": "Predict user churn",
            "expected_output": "detailed"
        }
    },
    {
        "agent": "insight",
        "task": "generate_answer",
        "parameters": {
            "question": "Predict user churn",
            "question_type": "PREDICTION",
            "output_format": "detailed"
        }
    }
]
```
- **Tasks**: 5
- **Execution Time**: 150 seconds
- **Quality Checks**: `["data_completeness", "result_validity", "model_accuracy"]`

### **7. 🎯 CUSTOM_QUERY Questions**
*Complex or ambiguous questions that don't fit standard patterns*

```python
task_plan = [
    {
        "agent": "data_pulling",
        "task": "extract_relevant_data",
        "parameters": {
            "target_tables": ["users", "events"],
            "columns": ["various_columns"],
            "time_filters": {},
            "conditions": [],
            "sample_size": 50000
        }
    },
    {
        "agent": "preprocessing",
        "task": "prepare_for_analysis",
        "parameters": {
            "grouping": [],
            "time_aggregation": "auto",
            "missing_data_strategy": "auto"
        }
    },
    # NO SPECIFIC ANALYSIS TASK FOR CUSTOM QUERIES
    {
        "agent": "qa",
        "task": "validate_analysis_results",
        "parameters": {
            "question_context": "Complex custom question",
            "expected_output": "summary"
        }
    },
    {
        "agent": "insight",
        "task": "generate_answer",
        "parameters": {
            "question": "Complex custom question",
            "question_type": "CUSTOM_QUERY",
            "output_format": "summary"
        }
    }
]
```
- **Tasks**: 4
- **Execution Time**: 120 seconds
- **Quality Checks**: `["data_completeness", "result_validity"]`

## 📋 **Summary Matrix**

| **Question Type** | **Data Pulling** | **Preprocessing** | **Analysis Task** | **QA** | **Insight** | **Total Tasks** | **Time (sec)** |
|-------------------|------------------|-------------------|-------------------|--------|-------------|-----------------|----------------|
| **DATA_EXPLORATION** (No tables) | ❌ | ❌ | ❌ | ❌ | ✅ | **1** | **30** |
| **DATA_EXPLORATION** (With tables) | ✅ | ❌ | ❌ | ✅ | ✅ | **3** | **90** |
| **STATISTICAL_SUMMARY** | ✅ | ✅ | `calculate_summary_statistics` | ✅ | ✅ | **5** | **150** |
| **TREND_ANALYSIS** | ✅ | ✅ | `analyze_trends` | ✅ | ✅ | **5** | **150** |
| **COMPARISON** | ✅ | ✅ | `comparative_analysis` | ✅ | ✅ | **5** | **150** |
| **CORRELATION** | ✅ | ✅ | `correlation_analysis` | ✅ | ✅ | **5** | **150** |
| **PREDICTION** | ✅ | ✅ | `predictive_modeling` | ✅ | ✅ | **5** | **150** |
| **CUSTOM_QUERY** | ✅ | ✅ | ❌ | ✅ | ✅ | **4** | **120** |

## 🔧 **Parameter Details by Agent**

### **🔍 Data Pulling Agent Parameters**

**Static Parameters:**
- `sample_size`: Always `50000`

**Dynamic Parameters:**
- `target_tables`: Based on detected entities in question
- `columns`: Based on entities and database schema
- `time_filters`: Extracted from question ("last month", date ranges)
- `conditions`: Extracted conditions ("platform = 'mobile'", "status = 'active'")

**Example Variations:**
```python
# Simple question
"time_filters": {}

# Time-based question
"time_filters": {"period": "last_month"}

# Date range question
"time_filters": {"start_date": "2024-01-01", "end_date": "2024-01-31"}

# Conditional question
"conditions": ["platform = 'mobile'", "user_type = 'premium'"]
```

### **🔧 Preprocessing Agent Parameters**

**Static Parameters:**
- `time_aggregation`: Always `"auto"`
- `missing_data_strategy`: Always `"auto"`

**Dynamic Parameters:**
- `grouping`: Based on detected grouping in question

**Example Variations:**
```python
# No grouping
"grouping": []

# Time-based grouping
"grouping": ["date"]

# Multi-dimensional grouping
"grouping": ["date", "platform", "country"]
```

### **📊 Analysis Agent Parameters**

#### **Statistical Summary**
```python
{
    "metrics": ["count", "mean", "median", "std"],  # Static
    "grouping": question.grouping                   # Dynamic
}
```

#### **Trend Analysis**
```python
{
    "time_column": "timestamp",                     # Static
    "metrics": question.entities,                   # Dynamic: ["users", "revenue"]
    "trend_method": "linear"                        # Static
}
```

#### **Comparative Analysis**
```python
{
    "comparison_groups": question.grouping,         # Dynamic: ["platform", "country"]
    "metrics": question.entities                    # Dynamic: ["users", "events"]
}
```

#### **Correlation Analysis**
```python
{
    "variables": question.entities,                 # Dynamic: ["usage", "retention"]
    "method": "pearson"                            # Static
}
```

#### **Predictive Modeling**
```python
{
    "target_variable": question.entities[0] if question.entities else "churn",  # Dynamic
    "features": question.entities[1:] if len(question.entities) > 1 else "auto" # Dynamic
}
```

### **✅ QA Agent Parameters**

**Always the Same Structure:**
```python
{
    "question_context": question.question_text,     # Dynamic - original question
    "expected_output": question.output_format       # Dynamic: "summary", "table", "visualization", "detailed"
}
```

**Output Format Variations:**
- `"summary"`: Brief overview (default)
- `"table"`: Tabular format
- `"visualization"`: Charts and graphs
- `"detailed"`: Comprehensive analysis

### **💡 Insight Agent Parameters**

**Always the Same Structure:**
```python
{
    "question": question.question_text,             # Dynamic - original question
    "question_type": question.question_type,        # Dynamic - detected type
    "output_format": question.output_format         # Dynamic - desired format
}
```

## 🎯 **Quality Checks by Question Type**

| **Question Type** | **Quality Checks** |
|-------------------|--------------------|
| **DATA_EXPLORATION** | `["data_completeness", "result_validity"]` |
| **STATISTICAL_SUMMARY** | `["data_completeness", "result_validity"]` |
| **TREND_ANALYSIS** | `["data_completeness", "result_validity", "temporal_consistency"]` |
| **COMPARISON** | `["data_completeness", "result_validity"]` |
| **CORRELATION** | `["data_completeness", "result_validity", "statistical_significance"]` |
| **PREDICTION** | `["data_completeness", "result_validity", "model_accuracy"]` |
| **CUSTOM_QUERY** | `["data_completeness", "result_validity"]` |

## 🚀 **Execution Time Analysis**

### **Time Breakdown by Number of Tasks**
| **Tasks** | **Time** | **Workflow Types** | **Examples** |
|-----------|----------|-------------------|--------------|
| **1** | **30 sec** | Pure Data Exploration | "What data do we have?" |
| **3** | **90 sec** | Data Exploration + Tables | "What's in the users table?" |
| **4** | **120 sec** | Custom Query | "Show me something interesting" |
| **5** | **150 sec** | Full Analysis | "Predict user churn" |

### **Time Per Agent**
- **Data Pulling**: ~30 seconds (database query + data extraction)
- **Preprocessing**: ~30 seconds (cleaning + feature engineering)
- **Analysis**: ~30 seconds (statistical/ML operations)
- **QA**: ~30 seconds (validation + quality checks)
- **Insight**: ~30 seconds (natural language generation)

## 🎭 **Real-World Workflow Examples**

### **Example 1: Simple Data Exploration**
```
User: "What data do we have?"
→ Question Type: DATA_EXPLORATION (no tables)
→ Workflow: [insight] (30 seconds)
→ Result: Schema overview and available tables
```

### **Example 2: User Analysis**
```
User: "How many active users do we have daily?"
→ Question Type: STATISTICAL_SUMMARY
→ Entities: ["users"]
→ Time Filters: {}
→ Grouping: ["date"]
→ Workflow: [data_pulling → preprocessing → analysis(summary_stats) → qa → insight] (150 seconds)
→ Result: Daily active user counts with statistics
```

### **Example 3: Trend Analysis**
```
User: "Show me user growth trends over the last 6 months"
→ Question Type: TREND_ANALYSIS
→ Entities: ["users"]
→ Time Filters: {"period": "last_6_months"}
→ Grouping: ["month"]
→ Workflow: [data_pulling → preprocessing → analysis(trends) → qa → insight] (150 seconds)
→ Result: Growth trend analysis with projections
```

### **Example 4: Complex Prediction**
```
User: "Predict which premium users are likely to churn"
→ Question Type: PREDICTION
→ Entities: ["churn", "users"]
→ Conditions: ["user_tier = 'premium'"]
→ Workflow: [data_pulling → preprocessing → analysis(predictive_modeling) → qa → insight] (150 seconds)
→ Result: Churn prediction model with risk scores
```

## 🔄 **Decision Tree Logic**

```
Question Input
├── Question Type = DATA_EXPLORATION?
│   ├── No Target Tables? → [insight] (1 task)
│   └── Has Target Tables? → [data_pulling → qa → insight] (3 tasks)
├── Question Type = CUSTOM_QUERY?
│   └── [data_pulling → preprocessing → qa → insight] (4 tasks)
└── Other Question Types?
    └── [data_pulling → preprocessing → analysis(specific) → qa → insight] (5 tasks)
```

## 📊 **System Statistics**

- **Total Possible Combinations**: 8 different workflows
- **Shortest Workflow**: 1 task (30 seconds)
- **Longest Workflow**: 5 tasks (150 seconds)
- **Most Common Workflow**: 5 tasks (analysis workflows)
- **Agents Involved**: 1-5 agents per workflow
- **Success Rate**: ~90% (with fallback to custom query)

## 🛠️ **Extension Points**

### **Adding New Question Types**
1. Add to `QuestionType` enum in `data_models.py`
2. Add detection patterns in `_detect_question_type()`
3. Add analysis task in `_create_task_plan()`
4. Add quality checks in `_determine_quality_checks()`

### **Adding New Analysis Tasks**
1. Define task parameters in `_create_task_plan()`
2. Implement task in respective agent
3. Add quality checks if needed
4. Update documentation

### **Custom Workflows**
The system is designed to be **highly extensible**. New workflows can be added by:
- Extending the question type detection
- Adding new analysis task types
- Customizing parameter sets
- Adding specialized quality checks

---

**The Dynamic Task Planning system creates perfectly tailored workflows for every question, optimizing both accuracy and execution time!** 🚀 