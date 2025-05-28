#!/usr/bin/env python3
"""
Event-Based User Analysis System - LangGraph Architecture
Now powered by LangGraph for reliable, stateful agent workflows
"""

import os
import sys
import json # For saving results to JSON
from langgraph_agent import run_event_relationship_analysis
from config_loader import load_config
from report_generator import generate_analysis_report

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

def save_results(results: dict, output_config: dict) -> bool:
    """Save analysis results in the configured format
    
    Args:
        results: Analysis results from LangGraph workflow
        output_config: Output configuration
        
    Returns:
        True if saved successfully, False otherwise
    """
    output_file = output_config['default_file']
    output_format = output_config.get('format', 'json').lower()
    
    try:
        if output_format == 'report':
            # Generate markdown report
            print(f"\n📋 Generating analysis report: {output_file}")
            success = generate_analysis_report(results, output_config, output_file)
            if success:
                print(f"✅ Analysis report saved successfully.")
                return True
            else:
                print(f"❌ Failed to generate analysis report.")
                return False
        
        else:
            # Save as JSON (fallback)
            print(f"\n💾 Saving analysis results as JSON: {output_file}")
            serializable_results = convert_state_to_json_serializable(results)
            
            # Apply output configuration for JSON
            if not output_config.get('include_raw_data', False) and 'raw_data' in serializable_results:
                del serializable_results['raw_data']
            
            # For JSON, use pretty_print setting (default True for backward compatibility)
            indent = 4 if output_config.get('pretty_print', True) else None
            with open(output_file, 'w') as f:
                json.dump(serializable_results, f, indent=indent)
            print(f"✅ JSON results saved successfully.")
            return True
            
    except Exception as e:
        print(f"❌ Error saving results: {type(e).__name__} - {str(e)}")
        return False

def main():
    """
    Main entry point for the event-based user analysis system.
    
    New LangGraph Architecture:
    1. Data processing happens separately and saves to database
    2. LangGraph agent provides controllable, stateful analysis workflows
    3. Better reliability and human-in-the-loop capabilities
    """
    
    # Get config file path from command line argument or use default
    config_file = "config.yaml"
    if len(sys.argv) > 1:
        config_file = sys.argv[1]

    print("Event-Based User Analysis System")
    print("Powered by LangGraph for Reliable Agent Workflows")
    print("=" * 60)
    
    # Load configuration
    try:
        config = load_config(config_file)
        config.print_config_summary()
    except Exception as e:
        print(f"❌ Error loading configuration: {e}")
        return False
    
    # Get configuration values
    db_config = config.get_database_config()
    llm_config = config.get_llm_config()
    analysis_config = config.get_analysis_config()
    output_config = config.get_output_config()
    
    db_path = db_config['path']
    
    # Validate database path
    if not os.path.exists(db_path):
        print(f"❌ Database not found at: {db_path}!")
        print("\nTo get started:")
        print("1. Ensure 'DataProcess/data_processor.py' has run and created the database.")
        print("   You can run it via: python run_data_processing.py")
        print("\n2. Then run this LangGraph agent analysis:")
        print(f"   python main.py [config_file.yaml]")
        print("\nLangGraph provides:")
        print("  🔄 Stateful workflows with checkpoints")
        print("  🎯 Controllable agent behavior")
        print("  🔍 Step-by-step execution tracking")
        print("  🛡️  Built-in error handling and recovery")
        return False
    
    # Validate LLM configuration
    api_key = llm_config.get('api_key')
    if not api_key:
        print("❌ Critical Error: OpenAI API key is required.")
        print("Please provide it via:")
        print("  1. Environment variable: export OPENAI_API_KEY='your_key'")
        print("  2. Config file: Set llm.api_key in config.yaml")
        return False
    
    try:
        print(f"✅ Database found at {db_path}. Initializing LangGraph agent...")
        print(f"🤖 LLM Configuration:")
        print(f"   Model: {llm_config['model']}")
        print(f"   Temperature: {llm_config['temperature']}")
        print(f"   API Key: {'✅ Provided' if api_key else '❌ Missing'}")
        print(f"📄 Output Configuration:")
        print(f"   Format: {output_config.get('format', 'json').upper()}")
        print(f"   File: {output_config['default_file']}")

        print("📊 LangGraph benefits:")
        print("  • Stateful conversation management")
        print("  • Controllable multi-step workflows")
        print("  • Built-in persistence and checkpointing")
        print("  • Human-in-the-loop capabilities")
        print("  • Better reliability than traditional agent frameworks")
        
        # Run analysis with configuration
        results = run_event_relationship_analysis(
            database_path=db_path, 
            print_details=analysis_config['print_details'],
            llm_config=llm_config,
            database_config=db_config
        )
        
        if results.get("success"):
            print("\n🎉 LangGraph Analysis Workflow Completed Successfully!")
            
            # Save results if configured
            if output_config['auto_save']:
                save_success = save_results(results, output_config)
                if not save_success:
                    print("⚠️  Analysis completed but output saving failed.")

        else:
            print("\n❌ LangGraph Analysis Workflow Encountered an Error.")
            print(f"   Final Step Reached: {results.get('current_step', 'N/A')}")
            print(f"   Error Message: {results.get('error_message', 'No specific error message provided.')}")
            print("   Check the detailed workflow log above for more information.")
            
            # Save failed state if configured
            if output_config['auto_save']:
                print(f"\n💾 Saving FAILED analysis state...")
                save_success = save_results(results, output_config)
                if save_success:
                    print(f"✅ Failed state saved for debugging.")
                else:
                    print(f"❌ Could not save failed state.")

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
    print("\n🤖 Configurable LLM Support:")
    print("  • Multiple OpenAI models (GPT-4, GPT-3.5-turbo, GPT-4o-mini)")
    print("  • Adjustable temperature for creativity vs focus")
    print("  • Flexible API key configuration")
    print("  • Easy model switching for different use cases")
    print("\n⚙️  Configuration-Driven:")
    print("  • YAML configuration files for all settings")
    print("  • Environment variable overrides")
    print("  • Validation and defaults for all parameters")
    print("  • Simple usage: python main.py [config.yaml]")
    print("\n📋 Report Generation:")
    print("  • Comprehensive markdown reports")
    print("  • Executive summaries and key insights")
    print("  • Statistical analysis and visualizations")
    print("  • AI-generated insights and recommendations")


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