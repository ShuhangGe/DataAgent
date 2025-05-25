# Event-Based User Analysis System

A modular data analysis system that processes event logs and performs user segmentation based on registration time and behavioral patterns. **Now powered by LangGraph** for reliable, stateful agent workflows with enhanced control and observability.

## Architecture

🔄 **LangGraph-Powered Pipeline:**
1. **Data Processor** (`DataProcess/`) - Processes raw CSV data and saves device event dictionaries to database
2. **LangGraph Agent** (`langgraph_agent.py`) - Stateful, controllable agent that analyzes device behavior patterns

## Why LangGraph?

Based on [LangGraph's capabilities](https://langchain-ai.github.io/langgraph/), our system now provides:

✅ **Controllable Workflows**: Step-by-step execution control and monitoring  
✅ **Stateful Analysis**: Maintains context and conversation history  
✅ **Error Recovery**: Built-in fault tolerance and retry mechanisms  
✅ **Human-in-the-Loop**: Easy integration of approval workflows  
✅ **Observability**: Full workflow monitoring and debugging  
✅ **Production Ready**: Reliable agents designed for real-world deployment  

## Requirements Implemented

✅ **Data Input & Preprocessing:**
- Filter dataset to include only users from the United States
- Remove all users who have clicked (events contain click-type actions)
- Use `device_id` as the unique user identifier (primary key)
- Save event-time pairs as dictionary structures

✅ **LangGraph Agent Analysis:**
- Multi-step workflow for device behavior analysis
- Stateful conversation management
- Generate insights about user patterns and temporal behaviors
- Create actionable business recommendations
- Built-in error handling and recovery

✅ **Technical Requirements:**
- Uses `pandas` for data handling
- Assumes timestamps are in UTC
- Modular script with clear function separation
- **Uses LangGraph as the agent framework** (replaces CrewAI)
- Database persistence for processed data
- Clean, readable code with comprehensive monitoring

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up OpenAI API key (required for LangGraph agent):
```bash
export OPENAI_API_KEY="your-api-key-here"
```

3. **Step 1: Process Data** (Run once or when data changes)
```bash
python run_data_processing.py
```

4. **Step 2: Run LangGraph Agent Analysis** (Can run multiple times)
```bash
python main.py
```

## Project Structure

```
├── main.py                    # Main entry point (LangGraph agent analysis)
├── run_data_processing.py     # Data processing entry point
├── langgraph_agent.py         # LangGraph-based behavior analysis agent
├── DataProcess/               # Data processing package
│   ├── __init__.py
│   ├── data_processor.py      # Data processing pipeline
│   ├── data_analysis.py       # Core analysis functions
│   └── data_processor_README.md  # Detailed processor documentation
├── requirements.txt           # LangGraph and processing dependencies
├── event_analysis.db          # SQLite database (created by data processor)
└── README.md                  # This file
```

## LangGraph Workflow

The LangGraph agent follows this stateful workflow:

```
Initialize Analysis
        ↓
Load Database Data
        ↓
Analyze Behavior Patterns  ← LLM Analysis
        ↓
Generate Business Insights ← LLM Insights
        ↓
Create Recommendations    ← LLM Strategy
        ↓
Finalize Report
```

Each step maintains state and can be interrupted, resumed, or modified.

## Database Tables

The data processor creates these tables:
- `device_event_dictionaries` - Device records with event-time pairs as JSON
- `device_dict_summary` - Processing and summary statistics

## Benefits of LangGraph Architecture

✅ **Reliability** - [LangGraph's stateful design](https://github.com/langchain-ai/langgraph) handles complex workflows robustly  
✅ **Control** - Step-by-step execution with checkpoints and rollback  
✅ **Observability** - Full workflow monitoring and debugging capabilities  
✅ **Scalability** - Production-ready agent deployment with LangGraph Platform  
✅ **Flexibility** - Easy to modify workflows and add human approval steps  
✅ **State Management** - Persistent conversation context across sessions  

## LangGraph Features Used

🎯 **StateGraph**: Defines the multi-step analysis workflow  
�� **State Management**: Maintains analysis context throughout execution  
🔄 **Node-Based Processing**: Each analysis step as a separate, controllable node  
🛡️ **Error Handling**: Built-in recovery and fault tolerance  
📊 **Structured Output**: Type-safe state management with TypedDict  

## Output

**Data Processor Output:**
- Database tables with device event dictionaries
- Processing statistics and data quality metrics

**LangGraph Agent Output:**
- Stateful behavior pattern analysis
- Business intelligence insights with metrics
- Strategic recommendations with priorities
- Complete workflow state for follow-up analysis

## Advanced LangGraph Capabilities

The system is designed to leverage LangGraph's advanced features:

🔄 **Streaming**: Real-time analysis progress (can be enabled)  
👥 **Human-in-the-Loop**: Easy addition of approval workflows  
🔍 **Debugging**: Step-by-step execution inspection  
📈 **Scaling**: Compatible with LangGraph Platform for production deployment  
🎯 **Customization**: Modular workflow nodes for easy modification  

## Migration from CrewAI

This system has been upgraded from CrewAI to LangGraph for:
- Better reliability and production readiness
- Enhanced state management and persistence
- Superior error handling and recovery
- More controllable agent behavior
- Built-in observability and debugging

## Learn More

- 📚 [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- 🐙 [LangGraph GitHub](https://github.com/langchain-ai/langgraph)
- 🎓 [LangChain Academy](https://academy.langchain.com/)

This architecture follows workflow.txt requirements while providing enterprise-grade reliability and control through LangGraph's stateful agent framework. 