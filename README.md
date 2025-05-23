# 🎮 Recommendation Click Analysis System

**Simplified MVP for Understanding User Behavior**

A multi-agent system built with CrewAI to understand why users don't click on recommended content.

## 🎯 Focus

**Primary Goal**: Understand why users view but don't click on recommendations

**Data Focus**: Users who viewed but didn't click recommendations  
**MVP Approach**: Using only `timestamp` and `event` columns for analysis

## 📊 Expected Data Format

Your data should contain records of users who viewed recommendations but didn't click. Expected columns:

### Required (MVP)
- `event` - Event type (e.g., 'recommendation_view', 'app_open') 
- `timestamp` - Event timestamp in UTC
- `device_id` - Device identifier (used as user ID)

### Optional (Full Dataset)
- `uuid` - Event UUID
- `distinct_id` - Distinct user ID  
- `country` - User country
- `timezone` - User timezone
- `newDevice` - Whether device is new

### Data Cleaning
The system automatically removes invalid data:
- Users who opened the app without receiving exposure cards
- Events without proper recommendation exposure flow

## 🔧 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare Your Data
Place your recommendation data in CSV format:
```
data/input/recommendation_data.csv
```

### 3. Run Analysis
```bash
python main.py analyze
```

Follow the interactive prompts to:
1. Select analysis type
2. Configure data source  
3. Set analysis parameters

## 📈 Analysis Types

### 1. Recommendation Funnel Analysis
- **Purpose**: Analyze user journey from viewing to not clicking
- **Output**: Click-through rates, exposure metrics, user behavior patterns
- **Key Insights**: CTR performance vs industry benchmarks, exposure frequency issues

### 2. Time Pattern Analysis  
- **Purpose**: Identify when users are most/least likely to engage
- **Output**: Hourly and daily engagement patterns
- **Key Insights**: Optimal timing for recommendation delivery

### 3. User Behavior Analysis
- **Purpose**: Segment users based on engagement patterns
- **Output**: User segments, event patterns, behavioral insights
- **Key Insights**: High/low activity segments, personalization opportunities

## 🤖 Multi-Agent Architecture

The system uses 6 specialized agents:

1. **Manager Agent** - Orchestrates recommendation analysis workflow
2. **Data Pulling Agent** - Loads recommendation event data
3. **Preprocessing Agent** - Cleans invalid exposures, prepares data
4. **Analysis Agent** - Performs funnel, time pattern, and behavior analysis
5. **QA Agent** - Validates data quality and analysis accuracy
6. **Insight Agent** - Generates insights on why users don't click

## 📁 Output Files

Analysis generates several output files in `data/output/`:

- `recommendation_funnel_YYYYMMDD_HHMMSS.json` - Funnel metrics
- `user_behavior_YYYYMMDD_HHMMSS.csv` - User-level behavior data
- `time_patterns_YYYYMMDD_HHMMSS.json` - Time-based patterns
- `recommendation_insights_YYYYMMDD_HHMMSS.json` - Business insights
- `recommendation_summary_YYYYMMDD_HHMMSS.md` - Executive summary

## 💡 Key Insights Generated

The system provides actionable insights such as:

- **CTR Performance**: How your click-through rate compares to industry standards (8-12%)
- **Timing Optimization**: Best hours/days for recommendation delivery
- **User Segmentation**: High/medium/low activity user groups
- **Frequency Issues**: Over-exposure without clicks (recommendation fatigue)
- **Data Quality**: Invalid exposures and tracking issues

## 🔍 CLI Commands

```bash
# Run interactive analysis
python main.py analyze

# Check system status
python main.py check-system

# List available agents
python main.py list-agents

# Show version info
python main.py version
```

## ⚙️ Configuration

Key settings in `src/config/settings.py`:

```python
# OpenAI Configuration
OPENAI_API_KEY = "your-api-key"
model = "gpt-4o"

# Analysis Settings  
chunk_size = 10000
sample_size = 100000
min_data_quality_score = 0.8

# Paths
input_data_path = "data/input"
output_data_path = "data/output"
```

## 📝 Example Usage

1. **Prepare Data**: CSV file with recommendation events
2. **Run Analysis**: `python main.py analyze`
3. **Select Type**: Choose "Recommendation Funnel Analysis"  
4. **Configure Source**: Point to your CSV file
5. **Review Results**: Check generated insights and recommendations

## 🎯 Business Value

This system helps you:

- **Identify** why users don't engage with recommendations
- **Optimize** recommendation timing and frequency
- **Segment** users for personalized strategies  
- **Improve** overall click-through rates
- **Fix** data tracking and exposure issues

## 🔧 Technical Requirements

- Python 3.11+
- OpenAI API key
- CrewAI 0.70.1
- 8GB+ RAM for large datasets
- CSV/Database data source

## 📞 Support

For questions about the recommendation analysis system:

1. Check the CLI help: `python main.py --help`
2. Review system status: `python main.py check-system`
3. Verify data format matches expected columns
4. Ensure proper OpenAI API configuration

---

*Built with CrewAI + OpenAI GPT-4o for robust recommendation analysis* 