#!/usr/bin/env python3
"""
LangGraph-based Event Relationship Analysis Agent
Analyzes event patterns and relationships in user behavior data
"""

import os
import json
import sqlite3
import pandas as pd
from typing import TypedDict, List, Dict, Any
from datetime import datetime, timedelta
import numpy as np
from collections import Counter, defaultdict

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

# State definition for the workflow
class AnalysisState(TypedDict):
    messages: List[str]
    database_path: str
    database_config: Dict[str, Any]  # Database configuration (path, table_name)
    llm_config: Dict[str, Any]  # LLM configuration (model, temperature, api_key)
    raw_data: pd.DataFrame
    event_patterns: Dict[str, Any]
    behavioral_insights: Dict[str, Any]
    recommendations: List[str]
    current_step: str
    error_message: str
    print_details: bool

class EventRelationshipAnalyzer:
    """Advanced analyzer for event relationships and behavioral patterns
    """
    
    def __init__(self, llm):
        self.llm = llm
    
    def analyze_event_sequences(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze event sequences and transitions
        get event sequences for each user and event transitions for the whole datsset."""
        sequences = {}
        transitions = defaultdict(lambda: defaultdict(int))
        session_patterns = []
        
        for _, row in df.iterrows():
            device_id = row['device_id']
            event_pairs = json.loads(row['event_time_pairs'])
            
            # Extract event sequence
            events = [pair['event'] for pair in event_pairs]
            sequences[device_id] = events
            
            # Analyze transitions
            for i in range(len(events) - 1):
                current_event = events[i]
                next_event = events[i + 1]
                transitions[current_event][next_event] += 1
            
            # Session pattern analysis
            session_patterns.append({
                'device_id': device_id,
                'session_length': len(events),
                'unique_events': len(set(events)),
                'first_event': events[0] if events else None,
                'last_event': events[-1] if events else None,
                'dominant_event': Counter(events).most_common(1)[0] if events else None
            })
        
        return {
            'sequences': sequences,
            'transitions': dict(transitions),
            'session_patterns': session_patterns,
            'total_sessions': len(sequences)
        }
    
    def analyze_temporal_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze temporal patterns in events"""
        temporal_analysis = {
            'hourly_distribution': defaultdict(int),#caculate the number of event happened in each hour and day.
            'daily_distribution': defaultdict(int),
            'session_durations': [],# Y
            'inter_event_gaps': [],
            'peak_activity_hours': [],
            'user_activity_patterns': {}
        }
        
        for _, row in df.iterrows():
            device_id = row['device_id']
            event_pairs = json.loads(row['event_time_pairs'])
            
            if not event_pairs:
                continue
                
            # Hourly and daily distribution
            for pair in event_pairs:
                hour = pair['hour']
                day = pair['day_of_week']
                temporal_analysis['hourly_distribution'][hour] += 1
                temporal_analysis['daily_distribution'][day] += 1
            
            # Session duration
            first_time = pd.to_datetime(event_pairs[0]['timestamp'])
            last_time = pd.to_datetime(event_pairs[-1]['timestamp'])
            duration_minutes = (last_time - first_time).total_seconds() / 60
            temporal_analysis['session_durations'].append(duration_minutes)
            
            # Inter-event gaps
            gaps = []
            for i in range(len(event_pairs) - 1):
                current_time = pd.to_datetime(event_pairs[i]['timestamp'])
                next_time = pd.to_datetime(event_pairs[i + 1]['timestamp'])
                gap_seconds = (next_time - current_time).total_seconds()
                gaps.append(gap_seconds)
            
            temporal_analysis['inter_event_gaps'].extend(gaps)
            
            # User activity pattern
            temporal_analysis['user_activity_patterns'][device_id] = {
                'total_events': len(event_pairs),
                'session_duration_minutes': duration_minutes,
                'avg_gap_seconds': np.mean(gaps) if gaps else 0,
                'activity_hours': list(set(pair['hour'] for pair in event_pairs)),
                'activity_days': list(set(pair['day_of_week'] for pair in event_pairs))
            }
        
        # Calculate peak activity hours
        hourly_counts = temporal_analysis['hourly_distribution']
        if hourly_counts:
            peak_hours = sorted(hourly_counts.items(), key=lambda x: x[1], reverse=True)[:3]
            temporal_analysis['peak_activity_hours'] = peak_hours
        
        return temporal_analysis
    
    def analyze_event_relationships(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze relationships between different events"""
        relationships = {
            'event_co_occurrence': defaultdict(lambda: defaultdict(int)),
            'event_frequency': defaultdict(int),
            'event_clustering': {},
            'funnel_analysis': {},
            'conversion_patterns': {}
        }
        
        all_events = set()
        
        for _, row in df.iterrows():
            event_pairs = json.loads(row['event_time_pairs'])
            events = [pair['event'] for pair in event_pairs]
            unique_events = set(events)
            all_events.update(unique_events)
            
            # Event frequency
            for event in events:
                relationships['event_frequency'][event] += 1
            
            # Co-occurrence analysis
            for i, event1 in enumerate(unique_events):
                for event2 in unique_events:
                    if event1 != event2:
                        relationships['event_co_occurrence'][event1][event2] += 1
        
        # Funnel analysis - common event sequences
        common_sequences = self._find_common_sequences(df)
        relationships['funnel_analysis'] = common_sequences
        
        # Conversion patterns
        relationships['conversion_patterns'] = self._analyze_conversions(df)
        
        return relationships
    
    def _find_common_sequences(self, df: pd.DataFrame, min_length=2, top_n=10) -> Dict[str, Any]:
        """Find most common event sequences"""
        sequence_counts = defaultdict(int)
        
        for _, row in df.iterrows():
            event_pairs = json.loads(row['event_time_pairs'])
            events = [pair['event'] for pair in event_pairs]
            
            # Generate subsequences of different lengths
            for length in range(min_length, min(len(events) + 1, 6)):  # Max length 5
                for i in range(len(events) - length + 1):
                    sequence = tuple(events[i:i + length])
                    sequence_counts[sequence] += 1
        
        # Get top sequences
        top_sequences = sorted(sequence_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]
        
        return {
            'top_sequences': [(list(seq), count) for seq, count in top_sequences],
            'total_unique_sequences': len(sequence_counts)
        }
    
    def _analyze_conversions(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze conversion patterns between events"""
        conversions = {
            'app_open_to_engagement': 0,
            'anonymous_to_engagement': 0,
            'screen_view_to_action': 0,
            'total_sessions': 0
        }
        
        engagement_events = {'expose_foru_sekai_card', 'app_open_af'}
        
        for _, row in df.iterrows():
            event_pairs = json.loads(row['event_time_pairs'])
            events = [pair['event'] for pair in event_pairs]
            conversions['total_sessions'] += 1
            
            # App open to engagement
            if 'app_open' in events:
                if any(event in engagement_events for event in events):
                    conversions['app_open_to_engagement'] += 1
            
            # Anonymous login to engagement
            if 'anonymous_login' in events:
                if any(event in engagement_events for event in events):
                    conversions['anonymous_to_engagement'] += 1
            
            # Screen view to action
            if 'screen_view' in events:
                screen_index = events.index('screen_view')
                if screen_index < len(events) - 1:  # Not the last event
                    conversions['screen_view_to_action'] += 1
        
        # Calculate conversion rates
        if conversions['total_sessions'] > 0:
            # Iterate over a copy of keys to allow modification
            for key in list(conversions.keys()): 
                if key != 'total_sessions' and not key.endswith('_rate'): # Ensure we only process original counts
                    rate = conversions[key] / conversions['total_sessions']
                    conversions[f'{key}_rate'] = rate
        
        return conversions

def initialize_analysis(state: AnalysisState) -> AnalysisState:
    """Initialize the analysis workflow"""
    state["messages"].append("🚀 Starting Event Relationship Analysis...")
    
    # Display LLM configuration
    llm_config = state.get("llm_config", {})
    if llm_config:
        model = llm_config.get("model", "Unknown")
        temperature = llm_config.get("temperature", "Unknown")
        api_key_status = "✅ Configured" if llm_config.get("api_key") else "❌ Missing"
        state["messages"].append(f"🤖 LLM Configuration: {model} (temp: {temperature}, API key: {api_key_status})")
    
    state["current_step"] = "initialization"
    
    # Set default database path if not provided
    if not state.get("database_path"):
        state["database_path"] = "DataProcess/event_analysis.db"
    
    return state

def load_processed_data(state: AnalysisState) -> AnalysisState:
    """Load processed data from database"""
    try:
        state["messages"].append(f"📊 Loading processed event data from: {state['database_path']}...")
        
        # Get table name from database config if available, otherwise use default
        database_config = state.get("database_config", {})
        table_name = database_config.get("table_name", "device_event_dictionaries")
        
        conn = sqlite3.connect(state["database_path"])
        query = f"""
        SELECT device_id, event, timestamp, uuid, distinct_id, country, timezone, newDevice,
               event_time_pairs, total_events, first_event_time, last_event_time, 
               event_types, time_span_hours
        FROM {table_name}
        """
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        if df.empty:
            err_msg = f"No processed data found in database table '{table_name}'."
            state["error_message"] = err_msg
            state["messages"].append(f"❌ {err_msg}")
            state["raw_data"] = pd.DataFrame() # Ensure raw_data is an empty DataFrame
            return state
        
        state["raw_data"] = df
        state["messages"].append(f"✅ Loaded {len(df)} user sessions with event sequences from table '{table_name}'.")
        state["current_step"] = "data_loaded"
        state["error_message"] = "" # Clear previous errors
        
    except Exception as e:
        err_msg = f"Error loading data: {type(e).__name__} - {str(e)}"
        state["error_message"] = err_msg
        state["messages"].append(f"❌ {err_msg}")
        state["raw_data"] = pd.DataFrame() # Ensure raw_data is an empty DataFrame on error
    
    return state

def analyze_event_patterns(state: AnalysisState) -> AnalysisState:
    """Analyze event patterns and relationships"""
    try:
        state["messages"].append("🔍 Analyzing event patterns and relationships...")
        if state.get("raw_data") is None or state["raw_data"].empty:
            err_msg = "Cannot analyze event patterns: Raw data is empty or not loaded."
            state["error_message"] = err_msg
            state["messages"].append(f"⚠️ {err_msg}")
            state["event_patterns"] = {} # Ensure key exists
            return state

        df = state["raw_data"]
        analyzer = EventRelationshipAnalyzer(None)
        
        sequence_analysis = analyzer.analyze_event_sequences(df)
        temporal_analysis = analyzer.analyze_temporal_patterns(df)
        relationship_analysis = analyzer.analyze_event_relationships(df)
        
        state["event_patterns"] = {
            'sequences': sequence_analysis,
            'temporal': temporal_analysis,
            'relationships': relationship_analysis
        }
        state["messages"].append("✅ Event pattern analysis completed.")
        state["current_step"] = "patterns_analyzed"
        state["error_message"] = ""
        
    except Exception as e:
        err_msg = f"Error in pattern analysis: {type(e).__name__} - {str(e)}"
        state["error_message"] = err_msg
        state["messages"].append(f"❌ {err_msg}")
        state["event_patterns"] = {} # Ensure key exists even on error
    
    return state

def generate_behavioral_insights(state: AnalysisState) -> AnalysisState:
    """Generate behavioral insights using LLM"""
    try:
        state["messages"].append("🧠 Generating behavioral insights with AI...")
        if not state.get("event_patterns") or not state["event_patterns"].get("relationships"):
            err_msg = "Cannot generate insights: Event patterns or relationships not found. Previous step likely failed."
            state["error_message"] = err_msg
            state["messages"].append(f"⚠️ {err_msg}")
            state["behavioral_insights"] = {} # Ensure key exists
            return state

        # Get LLM configuration from state
        llm_config = state.get("llm_config", {})
        api_key = llm_config.get("api_key")
        model = llm_config.get("model", "gpt-4o-mini")
        temperature = llm_config.get("temperature", 0.3)
        
        if not api_key:
            err_msg = "OpenAI API key not provided in LLM configuration."
            state["error_message"] = err_msg
            state["messages"].append(f"❌ Critical: {err_msg}")
            state["behavioral_insights"] = {}
            return state

        state["messages"].append(f"🤖 Using LLM: {model} (temperature: {temperature})")
        llm = ChatOpenAI(model=model, temperature=temperature, api_key=api_key)
        patterns = state["event_patterns"]
        
        # Simplified summary for LLM to reduce complexity if needed, or ensure robust access
        analysis_summary = {
            'total_users': len(state["raw_data"]) if not state["raw_data"].empty else 0,
            'top_event_sequences': patterns.get('relationships', {}).get('funnel_analysis', {}).get('top_sequences', [])[:5],
            'event_frequency': dict(patterns.get('relationships', {}).get('event_frequency', {})),
            'conversion_rates': patterns.get('relationships', {}).get('conversion_patterns', {}),
            'peak_activity_hours': patterns.get('temporal', {}).get('peak_activity_hours', []),
            'avg_session_duration': np.mean(patterns.get('temporal', {}).get('session_durations', [0])) if patterns.get('temporal', {}).get('session_durations') else 0,
            'top_transitions': {k: dict(v) for k, v in list(patterns.get('sequences', {}).get('transitions', {}).items())[:5]}
        }
        
        prompt = f"""
        Analyze the following user behavior data and provide insights about event relationships and user patterns:
        Data Summary: {json.dumps(analysis_summary, indent=2, default=str)}
        Please provide: Key behavioral patterns, event relationship insights, user engagement patterns, potential optimization opportunities, and anomalies.
        Focus on actionable insights. Keep the response concise.
        expose_foru_sekai_card means the user exposed to the 'for you' recommandation
        """
        messages = [
            SystemMessage(content="You are an expert data analyst specializing in user behavior and event analytics."),
            HumanMessage(content=prompt)
        ]
        response = llm.invoke(messages)
        state["behavioral_insights"] = {'ai_analysis': response.content, 'summary_stats': analysis_summary}
        state["messages"].append("✅ Behavioral insights generated.")
        state["current_step"] = "insights_generated"
        state["error_message"] = ""

    except Exception as e:
        err_msg = f"Error generating insights: {type(e).__name__} - {str(e)}"
        state["error_message"] = err_msg
        state["messages"].append(f"❌ {err_msg}")
        state["behavioral_insights"] = {}
    return state

def create_recommendations(state: AnalysisState) -> AnalysisState:
    """Create actionable recommendations"""
    try:
        state["messages"].append("💡 Creating actionable recommendations...")
        if not state.get("event_patterns") or not state.get("behavioral_insights"):
            err_msg = "Cannot create recommendations: Missing event patterns or behavioral insights. Previous steps likely failed."
            state["error_message"] = err_msg
            state["messages"].append(f"⚠️ {err_msg}")
            state["recommendations"] = []
            return state

        patterns = state["event_patterns"]
        insights = state["behavioral_insights"]
        recommendations = []

        conversions = patterns.get('relationships', {}).get('conversion_patterns', {})
        if 'app_open_to_engagement_rate' in conversions and conversions['app_open_to_engagement_rate'] < 0.5:
            recommendations.append(f"Low app open to engagement rate ({conversions['app_open_to_engagement_rate']:.2%}). Improve onboarding.")

        session_durations = patterns.get('temporal', {}).get('session_durations', [])
        if session_durations and np.mean(session_durations) < 5:
            recommendations.append(f"Short average session duration ({np.mean(session_durations):.1f} min). Increase engagement.")

        event_freq = patterns.get('relationships', {}).get('event_frequency', {})
        if event_freq: # Check if event_freq is not empty
            most_common_event = max(event_freq.items(), key=lambda x: x[1], default=(None, 0))
            if most_common_event[0] == 'expose_foru_sekai_card':
                recommendations.append("High card exposure. Analyze effectiveness.")
        
        peak_hours = patterns.get('temporal', {}).get('peak_activity_hours', [])
        if peak_hours:
            top_hour = peak_hours[0][0]
            recommendations.append(f"Peak activity at hour {top_hour}. Target notifications.")

        if insights.get('ai_analysis'):
            recommendations.append("Refer to AI Analysis for further detailed recommendations.") # Summarized
        
        state["recommendations"] = recommendations
        state["messages"].append(f"✅ Generated {len(recommendations)} recommendations.")
        state["current_step"] = "recommendations_created"
        state["error_message"] = ""
        
    except Exception as e:
        err_msg = f"Error creating recommendations: {type(e).__name__} - {str(e)}"
        state["error_message"] = err_msg
        state["messages"].append(f"❌ {err_msg}")
        state["recommendations"] = []
    return state

def finalize_analysis(state: AnalysisState) -> AnalysisState:
    """Finalize the analysis and prepare results"""
    state["messages"].append("📋 Finalizing event relationship analysis...")
    
    # Create comprehensive summary
    summary = {
        'analysis_timestamp': datetime.now().isoformat(),
        'total_users_analyzed': len(state["raw_data"]) if "raw_data" in state else 0,
        'event_patterns_found': len(state.get("event_patterns", {}).get("sequences", {}).get("sequences", {})),
        'recommendations_count': len(state.get("recommendations", [])),
        'analysis_steps_completed': state["current_step"]
    }
    
    state["messages"].append(f"✅ Analysis complete! Summary: {summary}")
    state["current_step"] = "completed"
    
    return state

def create_event_analysis_workflow():
    """Create the LangGraph workflow for event relationship analysis"""
    
    # Create the state graph
    workflow = StateGraph(AnalysisState)
    
    # Add nodes
    workflow.add_node("initialize", initialize_analysis)
    workflow.add_node("load_data", load_processed_data)
    workflow.add_node("analyze_patterns", analyze_event_patterns)
    workflow.add_node("generate_insights", generate_behavioral_insights)
    workflow.add_node("create_recommendations", create_recommendations)
    workflow.add_node("finalize", finalize_analysis)
    
    # Add edges
    workflow.add_edge("initialize", "load_data")
    workflow.add_edge("load_data", "analyze_patterns")
    workflow.add_edge("analyze_patterns", "generate_insights")
    workflow.add_edge("generate_insights", "create_recommendations")
    workflow.add_edge("create_recommendations", "finalize")
    workflow.add_edge("finalize", END)
    
    # Set entry point
    workflow.set_entry_point("initialize")
    
    return workflow.compile()

def run_event_relationship_analysis(database_path: str = "DataProcess/event_analysis.db", print_details: bool = False, llm_config: Dict[str, Any] = None, database_config: Dict[str, Any] = None):
    """Run the complete event relationship analysis
    
    Args:
        database_path: Path to the SQLite database containing processed event data
        print_details: Whether to print detailed analysis results
        llm_config: Dictionary containing LLM configuration:
                   - model: LLM model name (e.g., "gpt-4o-mini", "gpt-4", "gpt-3.5-turbo")
                   - temperature: Temperature for LLM responses (0.0-2.0)
                   - api_key: OpenAI API key
        database_config: Dictionary containing database configuration:
                        - path: Database file path
                        - table_name: Table name containing processed data
    """
    # Set default LLM config if not provided
    if llm_config is None:
        llm_config = {
            "model": "gpt-4o-mini",
            "temperature": 0.3,
            "api_key": os.getenv("OPENAI_API_KEY")
        }
    
    # Set default database config if not provided
    if database_config is None:
        database_config = {
            "path": database_path,
            "table_name": "device_event_dictionaries"
        }
    
    initial_state = AnalysisState(
        messages=[],
        database_path=database_path,
        database_config=database_config,
        llm_config=llm_config,
        raw_data=pd.DataFrame(),
        event_patterns={},
        behavioral_insights={},
        recommendations=[],
        current_step="",
        error_message="",
        print_details=print_details
    )
    workflow = create_event_analysis_workflow()
    print("🚀 Starting Event Relationship Analysis with LangGraph...")
    print("=" * 60)
    final_state = None
    try:
        final_state = workflow.invoke(initial_state)
        print("\n📋 WORKFLOW LOG:")
        print("-" * 60)
        for message in final_state["messages"]:
            print(message)
        
        if final_state.get("error_message"):
            print(f"\n❌ ANALYSIS FAILED: {final_state['error_message']}")
            return {
                "success": False,
                "error_message": final_state['error_message'],
                "current_step": final_state.get('current_step', 'unknown'),
                "log_messages": final_state.get('messages', [])
            }
        
        print("\n📊 ANALYSIS SUMMARY (Defaults):")
        print("=" * 60)
        
        raw_data_df = final_state.get("raw_data")
        if raw_data_df is not None and not raw_data_df.empty:
             print(f"📈 Total user sessions analyzed: {len(raw_data_df)}")
        else:
            print("📈 No user sessions loaded or analyzed.")

        ai_insights_text = final_state.get("behavioral_insights", {}).get("ai_analysis", "")
        if ai_insights_text:
            print("\n🧠 AI Behavioral Insights (Snippet):")
            print(ai_insights_text[:300] + "... (run with --print_details for full text)" if len(ai_insights_text) > 300 else ai_insights_text)
        else:
            print("\n🧠 AI Behavioral Insights: Not generated or empty.")
        
        recommendations_list = final_state.get("recommendations", [])
        if recommendations_list:
            print("\n💡 Recommendations:")
            for i, rec in enumerate(recommendations_list[:3], 1): # Show first 3 by default
                print(f"  {i}. {rec}")
            if len(recommendations_list) > 3:
                print(f"  ... and {len(recommendations_list) - 3} more (run with --print_details for all).")
        else:
            print("\n💡 Recommendations: None generated.")

        if final_state.get("print_details", False) and final_state.get("event_patterns"):
            print("\n🔍 DETAILED EVENT PATTERN INSIGHTS (print_details=True)")
            print("-" * 40)
            patterns = final_state["event_patterns"]
            if patterns.get("sequences"):
                seq_data = patterns["sequences"]
                print(f"Total user sessions in patterns: {seq_data.get('total_sessions', 'N/A')}") # Added for clarity
                if seq_data.get("transitions"):
                    print("\n🔄 Top Event Transitions (Sample):")
                    for event, transitions in list(seq_data["transitions"].items())[:3]: # Show top 3 originating events
                        if transitions:
                            # Sort transitions for this event and take top 2
                            sorted_transitions = sorted(transitions.items(), key=lambda item: item[1], reverse=True)[:2]
                            for next_event, count in sorted_transitions:
                                print(f"  {event} → {next_event} ({count} times)")
            if patterns.get("temporal"):
                temp_data = patterns["temporal"]
                if temp_data.get("peak_activity_hours"):
                     print(f"\n⏰ Peak Activity Hours: {temp_data['peak_activity_hours']}")
                if temp_data.get("session_durations"):
                    avg_duration = np.mean(temp_data["session_durations"]) if temp_data["session_durations"] else 0
                    print(f"\n⏱️  Average Session Duration: {avg_duration:.1f} minutes")
            if patterns.get("relationships"):
                rel_data = patterns["relationships"]
                if rel_data.get("event_frequency"):
                    print("\n📊 Most Frequent Events (Top 5):")
                    top_events = sorted(rel_data["event_frequency"].items(), key=lambda x: x[1], reverse=True)[:5]
                    for event, count in top_events:
                        print(f"  {event}: {count} occurrences")
                if rel_data.get("conversion_patterns"):
                    conv_data = rel_data["conversion_patterns"]
                    print("\n📈 Conversion Rates:")
                    for key, value in conv_data.items():
                        if key.endswith('_rate'):
                            print(f"  {key.replace('_rate', '').replace('_', ' ').title()}: {value:.2%}")
            # If --print_details is true, also print full AI insights and all recommendations if not already covered
            if ai_insights_text:
                print("\n🧠 Full AI Behavioral Insights (print_details=True):")
                print(ai_insights_text)
            if recommendations_list:
                print("\n💡 All Recommendations (print_details=True):")
                for i, rec in enumerate(recommendations_list, 1):
                    print(f"  {i}. {rec}")
        
        print("\n✅ Event relationship analysis process completed successfully!")
        final_state_to_return = final_state.copy()
        final_state_to_return["success"] = True
        return final_state_to_return
        
    except Exception as e:
        err_msg = f"Critical error in workflow execution: {type(e).__name__} - {str(e)}"
        print(f"❌ {err_msg}")
        return {
            "success": False,
            "error_message": err_msg,
            "current_step": final_state.get('current_step', 'unknown') if final_state else 'unknown',
            "log_messages": final_state.get('messages', []) if final_state else [err_msg]
        }

if __name__ == "__main__":
    # Example: Run with detailed printing
    # Make sure OPENAI_API_KEY is set in your environment
    # And DataProcess/event_analysis.db exists and is populated
    # For testing, you might want to create a small dummy DB if data processing is long
    
    # Ensure you have run `python run_data_processing.py` first
    # and set your OPENAI_API_KEY
    
    print("Example run from langgraph_agent.py (__main__)")
    # result = run_event_relationship_analysis(print_details=True)
    # print("\n--- Full Result State (example) ---")
    # if result.get("success"):
    #     print(f"Overall Success: {result['success']}")
    #     print(f"Behavioral Insights Summary: {result.get('behavioral_insights', {}).get('ai_analysis', '')[:200]}...") # Print first 200 chars
    #     print(f"Recommendations: {result.get('recommendations')}")
    # else:
    #     print(f"Overall Success: {result.get('success', False)}")
    #     print(f"Error: {result.get('error_message')}")
    #     print(f"Step: {result.get('current_step')}")

    # To run from main.py, it will call run_event_relationship_analysis
    pass # Keep __main__ minimal or for specific tests of this module