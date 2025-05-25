#!/usr/bin/env python3
"""
Event-Based User Analysis System - LangGraph Architecture
Now powered by LangGraph for reliable, stateful agent workflows
"""

import os
import sys
from langgraph_agent import run_langgraph_analysis

def main():
    """
    Main entry point for the event-based user analysis system.
    
    New LangGraph Architecture:
    1. Data processing happens separately and saves to database
    2. LangGraph agent provides controllable, stateful analysis workflows
    3. Better reliability and human-in-the-loop capabilities
    """
    
    print("Event-Based User Analysis System")
    print("Powered by LangGraph for Reliable Agent Workflows")
    print("=" * 60)
    
    # Check if database exists
    db_path = "event_analysis.db"
    
    if not os.path.exists(db_path):
        print("❌ Database not found!")
        print("\nTo get started:")
        print("1. First run data processing:")
        print("   python run_data_processing.py")
        print("\n2. Then run the LangGraph agent analysis:")
        print("   python main.py")
        print("\nLangGraph provides:")
        print("  🔄 Stateful workflows with checkpoints")
        print("  🎯 Controllable agent behavior")
        print("  🔍 Step-by-step execution tracking")
        print("  🛡️  Built-in error handling and recovery")
        return False
    
    try:
        print("✅ Database found. Initializing LangGraph agent...")
        print("📊 LangGraph benefits:")
        print("  • Stateful conversation management")
        print("  • Controllable multi-step workflows")
        print("  • Built-in persistence and checkpointing")
        print("  • Human-in-the-loop capabilities")
        print("  • Better reliability than traditional agent frameworks")
        
        # Run the LangGraph-driven analysis
        results = run_langgraph_analysis(db_path)
        
        if results["success"]:
            print("\n🎉 LangGraph Analysis Completed Successfully!")
            print("\nAnalysis Results:")
            
            # Display key findings
            raw_data = results.get("raw_data", {})
            if "device_events" in raw_data:
                device_df = raw_data["device_events"]
                total_devices = len(device_df)
                total_events = device_df['total_events'].sum()
                avg_events = total_events / total_devices if total_devices > 0 else 0
                
                print(f"  📱 Total devices analyzed: {total_devices:,}")
                print(f"  📊 Total events processed: {total_events:,}")
                print(f"  📈 Average events per device: {avg_events:.1f}")
            
            # Show insights availability
            insights = results.get("insights", {})
            if insights:
                print(f"\n🧠 Generated Insights:")
                if "behavior_analysis" in insights:
                    print(f"  ✅ Behavior Pattern Analysis")
                if "business_insights" in insights:
                    print(f"  ✅ Business Intelligence Insights")
                if "recommendations" in insights:
                    print(f"  ✅ Strategic Recommendations")
            
            print(f"\n🔧 LangGraph Features Used:")
            print(f"  • Stateful workflow execution")
            print(f"  • Multi-step analysis pipeline")
            print(f"  • Error handling and recovery")
            print(f"  • Structured state management")
            
            print(f"\n💾 All analysis data retained in workflow state")
            print(f"🔄 Ready for follow-up analysis or human review")
            
        else:
            print(f"\n❌ LangGraph Analysis failed: {results.get('error')}")
            print("🔧 LangGraph provides detailed error tracking for debugging")
            return False
        
    except Exception as e:
        print(f"❌ Error running LangGraph analysis: {str(e)}")
        print("💡 LangGraph's stateful nature helps with error recovery")
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
        print("\n✨ LangGraph-powered analysis system ready!")
        print("🔗 Learn more: https://langchain-ai.github.io/langgraph/")
    else:
        print("\n🔧 Check database setup and try again")
        print("📚 LangGraph docs: https://langchain-ai.github.io/langgraph/") 