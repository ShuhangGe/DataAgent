# Event-Driven User Behavior Analysis System with LangGraph

## Project Overview

This project implements an advanced, event-driven user behavior analysis system. It processes raw event logs, transforms them into a structured format, and then employs a sophisticated agent built with **LangGraph** to perform in-depth analysis. The primary goal is to understand user interaction patterns, identify key event relationships, and generate actionable insights and recommendations.

The system is divided into two main components:
1.  **Data Processing Pipeline**: A robust Python-based ETL (Extract, Transform, Load) process that cleans raw event data, filters it according to specified criteria, and stores the processed information in a structured format within a SQLite database. Each user's (identified by `device_id`) event history is preserved as a JSON object, detailing sequences of events with associated timestamps and derived temporal features (hour, day of week, etc.).
2.  **LangGraph Analysis Agent**: A stateful, multi-step analytical agent that consumes the processed data from the database. This agent leverages the LangGraph framework to orchestrate a series of analytical tasks, including event sequence analysis, temporal pattern detection, event relationship modeling, and LLM-powered insight generation. The output is a comprehensive report detailing user behavior patterns, key metrics, and strategic recommendations.

This project adheres to the requirements outlined in `development_doc/workflow.txt` while significantly enhancing the reliability, control, and observability of the analysis process through the adoption of LangGraph.

## Core Objectives

-   **Process Event Logs**: Handle raw CSV event data containing user interactions.
-   **Data Refinement**: Filter data for specific criteria (e.g., US-based users, excluding users with 'click' events) and use `device_id` as the primary user key.
-   **Structured Storage**: Save processed event sequences for each device as JSON objects in a SQLite database, including detailed event-time pairs and computed analytical fields (total events, first/last event time, event types, time span).
-   **Event Relationship Analysis**: Identify patterns in how events occur in sequence, their timing, and their co-occurrence.
-   **Behavioral Insights**: Utilize a Large LanguageModel (LLM) to interpret quantitative findings and generate qualitative insights about user behavior.
-   **Actionable Recommendations**: Produce data-driven and AI-enhanced recommendations for product or service optimization.

## Why LangGraph?

LangGraph was chosen as the framework for the analysis agent due to its powerful capabilities for building stateful, controllable, and production-ready applications with LLMs. Key advantages include:

-   **Cyclic Workflows**: LangGraph excels at creating workflows with cycles, allowing for iterative refinement, error handling, and human-in-the-loop interventions—capabilities that are crucial for complex analytical tasks.
-   **State Management**: It provides robust state management, ensuring that information is seamlessly passed between different steps (nodes) of the analysis. The state is explicit and can be persisted.
-   **Modularity and Control**: Workflows are defined as graphs where each node represents a specific function or tool. This modularity allows for clear separation of concerns and fine-grained control over the execution flow.
-   **Observability**: LangGraph offers excellent tools for tracing and debugging, making it easier to understand the agent's behavior and diagnose issues.
-   **Human-in-the-Loop**: The framework is designed to easily incorporate human review and approval steps, which can be vital for validating insights or recommendations before actioning them.
-   **Production Readiness**: LangGraph is built with production deployments in mind, offering features that support reliability and scalability.

## Project Architecture

The system follows a two-stage architecture:

1.  **Stage 1: Data Processing (`run_data_processing.py`)**
    *   **Input**: Raw CSV event data.
    *   **Process**: Managed by `DataProcess/data_processor.py`.
        *   Loads data using `pandas`.
        *   Applies preprocessing filters (US-only, no click events).
        *   Groups events by `device_id`.
        *   For each device, transforms its event history into a JSON array of `event_time_pairs`. Each pair includes: `event`, `timestamp`, `sequence` number, `hour`, `day_of_week`, and `date`.
        *   Computes additional analytical fields for each device record: `total_events`, `first_event_time`, `last_event_time`, `event_types` (list of unique events), and `time_span_hours`.
    *   **Output**: Stores the enriched device records in the `processed_events` table in a SQLite database (`user_events.db` by default).
    *   **Documentation**: Detailed information about this stage can be found in `DataProcess/data_processor_README.md`.

2.  **Stage 2: LangGraph Agent Analysis (`main.py` -> `langgraph_agent.py`)**
    *   **Input**: Processed data from the `processed_events` table in the SQLite database.
    *   **Process**: Managed by `langgraph_agent.py`, which defines and executes a LangGraph `StateGraph`.
The workflow consists of the following nodes:
        1.  `initialize_analysis`: Sets up the initial state.
        2.  `load_processed_data`: Loads data from the `processed_events` table.
        3.  `analyze_event_patterns`: Performs detailed quantitative analysis on event sequences, temporal aspects, and relationships using the `EventRelationshipAnalyzer` class.
        4.  `generate_behavioral_insights`: Uses an LLM (e.g., GPT-4o-mini) to interpret the patterns and generate qualitative insights.
        5.  `create_recommendations`: Combines rule-based logic and LLM-generated insights to formulate actionable recommendations.
        6.  `finalize_analysis`: Prepares a final summary of the analysis.
    *   **Output**: Console output detailing the analysis steps, key findings, insights, and recommendations. The final state of the LangGraph workflow is also returned, containing all collected data and results.
    *   **Documentation**: Detailed information about this agent can be found in `langgraph_agent_README.md`.

## Setup and Execution

### Prerequisites

-   Python 3.8+
-   An OpenAI API Key (for LLM-powered insights)

### Installation

1.  Clone the repository (if applicable).
2.  Install required Python packages:
    ```bash
    pip install -r requirements.txt
    ```
3.  Set your OpenAI API key as an environment variable:
    ```bash
    export OPENAI_API_KEY="your_openai_api_key_here"
    ```
    *(Note: On Windows, use `set OPENAI_API_KEY=your_openai_api_key_here` in Command Prompt or `$env:OPENAI_API_KEY="your_openai_api_key_here"` in PowerShell.)*

### Running the System

**Step 1: Data Processing**

This step needs to be run first to populate the database with processed event data. Execute it once, or whenever your raw input CSV data changes.

```bash
python run_data_processing.py [--input_file path/to/your/input.csv] [--db_file path/to/your/database.db]
```
-   `--input_file`: (Optional) Path to the raw input CSV file. Defaults to `DataProcess/data/sample_events.csv`.
-   `--db_file`: (Optional) Path to the SQLite database file to be created/updated. Defaults to `user_events.db`.

**Step 2: LangGraph Agent Analysis**

Once the database is populated, you can run the analysis agent. This can be executed multiple times on the same processed data.

```bash
python main.py [--db_file path/to/your/database.db]
```
-   `--db_file`: (Optional) Path to the SQLite database file containing processed data. Defaults to `user_events.db`.

The agent will print its progress, findings, and recommendations to the console.

## Project Structure

```
DataAgent/
├── main.py                    # Entry point for LangGraph agent analysis
├── run_data_processing.py     # Entry point for the data processing pipeline
|
├── langgraph_agent.py         # Defines the LangGraph agent and its workflow
├── langgraph_agent_README.md  # Detailed documentation for the LangGraph agent
|
├── DataProcess/               # Package for data processing logic
│   ├── __init__.py
│   ├── data_processor.py      # Core data processing and transformation script
│   ├── data_analysis.py       # (Legacy, potentially for utility functions if refactored)
│   ├── data_processor_README.md # Detailed documentation for the data_processor.py
│   └── data/                  # Directory for sample or input data
│       └── sample_events.csv  # Sample input CSV data
|
├── requirements.txt           # Lists all Python dependencies for the project
├── user_events.db             # SQLite database created by data_processor.py (example name)
├── development_doc/
│   └── workflow.txt           # Original requirements document
└── README.md                  # This main project README file
```

## Key Technologies Used

-   **Python**: Core programming language.
-   **LangGraph**: Framework for building stateful, multi-actor applications with LLMs.
-   **LangChain**: Used for LLM interaction components (e.g., `ChatOpenAI`, message schemas).
-   **Pandas**: For efficient data manipulation and analysis during the data processing stage.
-   **NumPy**: For numerical operations, especially in the analysis components.
-   **SQLite**: For lightweight, file-based database storage of processed data.
-   **OpenAI API**: For accessing GPT models for insight generation.

## Detailed Component Explanations

### 1. Data Processing (`DataProcess/data_processor.py`)

This script is responsible for the ETL process. It reads a raw CSV file, applies a series of transformations and enrichments, and stores the result in a SQLite database. Key operations include:

-   **Filtering**: Selects only US-based users and excludes any user who has performed a "click" event.
-   **User Identification**: Uses `device_id` as the unique identifier.
-   **Event Sequencing**: For each `device_id`, all associated events are ordered by timestamp.
-   **JSON Transformation**: The sequence of events for each device is transformed into a JSON array of `event_time_pairs`. Each object in this array contains:
    -   `event`: The name of the event.
    -   `timestamp`: The original UTC timestamp of the event.
    -   `sequence`: An integer indicating the order of the event in the user's session.
    -   `hour`: The hour of the day (0-23) when the event occurred.
    -   `day_of_week`: The day of the week (e.g., "Monday", "Tuesday") of the event.
    -   `date`: The date (YYYY-MM-DD) of the event.
-   **Feature Engineering**: Computes additional summary statistics for each device record stored in the database:
    -   `total_events`: Total number of events for the device.
    -   `first_event_time`: Timestamp of the very first event from the device.
    -   `last_event_time`: Timestamp of the most recent event from the device.
    -   `event_types`: A sorted list of unique event types performed by the device.
    -   `time_span_hours`: The duration in hours between the first and last event for the device.

Refer to `DataProcess/data_processor_README.md` for more granular details on its operation and database schema.

### 2. LangGraph Analysis Agent (`langgraph_agent.py`)

This is the analytical core of the project. It defines a stateful graph using LangGraph where each node performs a specific part of the analysis. The `EventRelationshipAnalyzer` class within this script contains the detailed logic for quantitative analysis.

-   **Workflow Nodes**: As detailed in the "Agent Workflow" section, steps include initialization, data loading, pattern analysis (sequences, temporal, relationships), LLM-based insight generation, recommendation creation, and finalization.
-   **`EventRelationshipAnalyzer` Class**: This class provides methods to:
    -   Identify common event sequences and transitions.
    -   Analyze temporal distributions (hourly, daily), session durations, and time gaps between events.
    -   Examine event co-occurrence, frequencies, and calculate conversion rates for predefined funnels.
-   **LLM Integration**: The `generate_behavioral_insights` node compiles quantitative findings and prompts an LLM to provide a narrative interpretation, uncovering deeper behavioral patterns and potential reasons behind the observed data.

Refer to `langgraph_agent_README.md` for an in-depth explanation of the agent's architecture, state, methods, and outputs.

## Conclusion

This Event-Driven User Behavior Analysis System, powered by LangGraph, offers a robust and insightful approach to understanding user interactions. By combining structured data processing with advanced AI-driven analysis, it provides a powerful tool for deriving actionable intelligence from event logs, ultimately enabling data-informed decision-making. 