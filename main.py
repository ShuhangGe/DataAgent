#!/usr/bin/env python3
"""
Event-Based User Analysis System - LangGraph Architecture
Now powered by LangGraph for reliable, stateful agent workflows
"""

import os
import sys
import argparse
import json # For saving results to JSON
from langgraph_agent import run_event_relationship_analysis
import pandas as pd

# Helper to convert DataFrame to JSON serializable format
def convert_state_to_json_serializable(state_dict):
    serializable_dict = {}
    for key, value in state_dict.items():
        if isinstance(value, pd.DataFrame):
            serializable_dict[key] = value.to_dict(orient='records')
        elif isinstance(value, dict):
            serializable_dict[key] = convert_state_to_json_serializable(value) # Recursively convert nested dicts
        elif isinstance(value, list):
            serializable_dict[key] = [
                item.to_dict(orient='records') if isinstance(item, pd.DataFrame) 
                else convert_state_to_json_serializable(item) if isinstance(item, dict) 
                else item 
                for item in value
            ]
        elif hasattr(value, 'isoformat'): # For datetime objects
             serializable_dict[key] = value.isoformat()
        else:
            try:
                json.dumps(value) # Test if serializable
                serializable_dict[key] = value
            except (TypeError, OverflowError):
                serializable_dict[key] = str(value) # Fallback to string
    return serializable_dict

def main():
    """
    Main entry point for the event-based user analysis system.
    
    New LangGraph Architecture:
    1. Data processing happens separately and saves to database
    2. LangGraph agent provides controllable, stateful analysis workflows
    3. Better reliability and human-in-the-loop capabilities
    """
    
    parser = argparse.ArgumentParser(description="Run LangGraph Event Analysis Agent.")
    parser.add_argument(
        "--db_file", 
        type=str, 
        default="DataProcess/event_analysis.db",  # Default to where DataProcessor saves it
        help="Path to the SQLite database file containing processed event data."
    )
    parser.add_argument(
        "--print_details",
        action="store_true",
        help="Print detailed event pattern insights upon successful completion."
    )
    parser.add_argument(
        "--output_file", # New argument for output file
        type=str,
        default='results.json', # No output file by default
        help="Path to save the analysis results as a JSON file (e.g., results.json)."
    )
    args = parser.parse_args()
    db_path = args.db_file

    print("Event-Based User Analysis System")
    print("Powered by LangGraph for Reliable Agent Workflows")
    print("=" * 60)
    
    if not os.path.exists(db_path):
        print(f"❌ Database not found at: {db_path}!")
        print("\nTo get started:")
        print("1. Ensure 'DataProcess/data_processor.py' has run and created the database.")
        print("   Expected input for data_processor.py is currently hardcoded to:")
        print("   /Users/shuhangge/Desktop/my_projects/Sekai/DataAgent/development_doc/mock_data.csv")
        print("   And its default output database is: DataProcess/event_analysis.db")
        print("   You can run it via: python run_data_processing.py")
        print("\n2. Then run this LangGraph agent analysis, optionally specifying the DB path:")
        print(f"   python main.py [--db_file {db_path}] [--print_details] [--output_file results.json]")
        print("\nLangGraph provides:")
        print("  🔄 Stateful workflows with checkpoints")
        print("  🎯 Controllable agent behavior")
        print("  🔍 Step-by-step execution tracking")
        print("  🛡️  Built-in error handling and recovery")
        return False
    
    # Check for OpenAI API Key early
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Critical Error: OPENAI_API_KEY environment variable is not set.")
        print("Please set it before running the analysis: export OPENAI_API_KEY=\"your_key\"")
        return False
    
    try:
        print(f"✅ Database found at {db_path}. Initializing LangGraph agent...")
        print("📊 LangGraph benefits:")
        print("  • Stateful conversation management")
        print("  • Controllable multi-step workflows")
        print("  • Built-in persistence and checkpointing")
        print("  • Human-in-the-loop capabilities")
        print("  • Better reliability than traditional agent frameworks")
        
        # Pass the print_details flag to the agent runner
        results = run_event_relationship_analysis(database_path=db_path, print_details=args.print_details)
        
        if results.get("success"):
            print("\n🎉 LangGraph Analysis Workflow Completed Successfully!")
            if args.output_file:
                try:
                    print(f"\n💾 Saving full analysis results to: {args.output_file}")
                    # Convert pandas DataFrames in results to a serializable format
                    serializable_results = convert_state_to_json_serializable(results)
                    with open(args.output_file, 'w') as f:
                        json.dump(serializable_results, f, indent=4)
                    print(f"✅ Results saved successfully.")
                except Exception as e:
                    print(f"❌ Error saving results to file: {type(e).__name__} - {str(e)}")
        else:
            print("\n❌ LangGraph Analysis Workflow Encountered an Error.")
            print(f"   Final Step Reached: {results.get('current_step', 'N/A')}")
            print(f"   Error Message: {results.get('error_message', 'No specific error message provided.')}")
            print("   Check the detailed workflow log above for more information.")
            if args.output_file:
                try:
                    print(f"\n💾 Saving FAILED analysis state to: {args.output_file}")
                    serializable_results = convert_state_to_json_serializable(results)
                    with open(args.output_file, 'w') as f:
                        json.dump(serializable_results, f, indent=4)
                    print(f"✅ Failed state saved.")
                except Exception as e:
                    print(f"❌ Error saving failed state to file: {type(e).__name__} - {str(e)}")
            return False
        
    except Exception as e:
        print(f"❌ Critical error in main.py execution: {type(e).__name__} - {str(e)}")
        # This catches errors outside the LangGraph workflow itself (e.g., issues in main.py logic)
        return False
    
    return True

def show_langgraph_info():
    """Display information about LangGraph capabilities"""
    print("\n🚀 About LangGraph Integration:")
    print("=" * 50)
    print("LangGraph provides advanced agent capabilities:")
    print("  🎯 Controllable Workflows: Step-by-step execution control")
    print("  💾 State Persistence: Maintains context across sessions")
    print("  🔄 Streaming Support: Real-time progress tracking")
    print("  🛡️  Error Recovery: Built-in fault tolerance")
    print("  👥 Human-in-the-Loop: Easy approval workflows")
    print("  📊 Observability: Full workflow monitoring")
    print("  🔧 Customizable: Flexible node and edge definitions")
    print("\nThis replaces CrewAI with more reliable, production-ready agents.")

if __name__ == "__main__":
    # Show LangGraph information
    show_langgraph_info()
    
    # Run main analysis
    success = main()
    
    if success:
        print("\n✨ LangGraph-powered analysis system finished.")
        print("🔗 Learn more: https://langchain-ai.github.io/langgraph/")
    else:
        print("\n🔧 Analysis did not complete successfully. Please review logs and configurations.")
        print("📚 LangGraph docs: https://langchain-ai.github.io/langgraph/") 