#!/usr/bin/env python3
"""
LangGraph-based Device Behavior Analysis Agent
Replaces CrewAI with LangGraph for better control and reliability
"""

import json
import pandas as pd
import sqlite3
from sqlalchemy import create_engine
from typing import Dict, Any, List, TypedDict, Annotated
from datetime import datetime

# LangGraph imports
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolExecutor
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.pydantic_v1 import BaseModel, Field

# State management for LangGraph
class AnalysisState(TypedDict):
    """State object for the device behavior analysis workflow"""
    messages: Annotated[List[Any], "The conversation messages"]
    database_path: str
    raw_data: Dict[str, Any]
    insights: Dict[str, Any]
    recommendations: List[str]
    current_step: str
    error_message: str

class DeviceBehaviorAnalyst:
    """
    LangGraph-based agent for device behavior analysis
    Provides controllable, stateful analysis workflow
    """
    
    def __init__(self, db_path="event_analysis.db", llm_model="gpt-3.5-turbo"):
        self.db_path = db_path
        self.engine = create_engine(f'sqlite:///{db_path}')
        
        # Initialize LLM
        self.llm = ChatOpenAI(model=llm_model, temperature=0.1)
        
        # Build the analysis workflow graph
        self.workflow = self._build_workflow()
        
    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow for device behavior analysis"""
        
        # Define the workflow graph
        workflow = StateGraph(AnalysisState)
        
        # Add nodes (analysis steps)
        workflow.add_node("initialize", self._initialize_analysis)
        workflow.add_node("load_data", self._load_database_data)
        workflow.add_node("analyze_patterns", self._analyze_behavior_patterns)
        workflow.add_node("generate_insights", self._generate_insights)
        workflow.add_node("create_recommendations", self._create_recommendations)
        workflow.add_node("finalize_report", self._finalize_report)
        
        # Define the workflow edges (control flow)
        workflow.set_entry_point("initialize")
        workflow.add_edge("initialize", "load_data")
        workflow.add_edge("load_data", "analyze_patterns")
        workflow.add_edge("analyze_patterns", "generate_insights")
        workflow.add_edge("generate_insights", "create_recommendations")
        workflow.add_edge("create_recommendations", "finalize_report")
        workflow.add_edge("finalize_report", END)
        
        return workflow.compile()

    def _initialize_analysis(self, state: AnalysisState) -> AnalysisState:
        """Initialize the analysis workflow"""
        print("🚀 Initializing LangGraph Device Behavior Analysis")
        print("=" * 60)
        
        state["database_path"] = self.db_path
        state["current_step"] = "initialize"
        state["raw_data"] = {}
        state["insights"] = {}
        state["recommendations"] = []
        state["error_message"] = ""
        
        # Add system message
        state["messages"] = [
            SystemMessage(content="""You are an expert device behavior analyst. Your role is to:
            1. Analyze device event patterns from structured database data
            2. Generate actionable insights about user behavior
            3. Provide strategic recommendations for business growth
            4. Focus on data-driven conclusions with specific metrics
            
            You work with device-centric data where each device has event-time pairs stored as JSON.
            """)
        ]
        
        return state

    def _load_database_data(self, state: AnalysisState) -> AnalysisState:
        """Load and validate data from database"""
        print("\n📊 Loading Device Data from Database")
        state["current_step"] = "load_data"
        
        try:
            # Check database tables
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                tables = [table[0] for table in cursor.fetchall()]
                
            if not tables:
                state["error_message"] = "No tables found in database"
                return state
                
            print(f"✅ Available tables: {tables}")
            
            # Load device event dictionaries (primary data)
            if 'device_event_dictionaries' in tables:
                df = pd.read_sql('SELECT * FROM device_event_dictionaries', self.engine)
                state["raw_data"]["device_events"] = df
                print(f"📱 Loaded {len(df)} device records")
                
                # Parse a sample of event-time pairs for analysis
                sample_events = []
                for idx, row in df.head(5).iterrows():
                    event_pairs = json.loads(row['event_time_pairs'])
                    sample_events.append({
                        'device_id': row['device_id'],
                        'total_events': row['total_events'],
                        'event_pairs': event_pairs[:3]  # First 3 events
                    })
                state["raw_data"]["sample_events"] = sample_events
            
            # Load summary statistics
            if 'device_dict_summary' in tables:
                summary_df = pd.read_sql('SELECT * FROM device_dict_summary', self.engine)
                state["raw_data"]["summary"] = summary_df
                print(f"📈 Loaded {len(summary_df)} summary metrics")
            
            state["messages"].append(
                HumanMessage(content="Database data loaded successfully. Ready for analysis.")
            )
            
        except Exception as e:
            state["error_message"] = f"Database loading error: {str(e)}"
            print(f"❌ Error loading data: {str(e)}")
            
        return state

    def _analyze_behavior_patterns(self, state: AnalysisState) -> AnalysisState:
        """Analyze device behavior patterns using LLM"""
        print("\n🔍 Analyzing Device Behavior Patterns")
        state["current_step"] = "analyze_patterns"
        
        if state["error_message"]:
            return state
            
        try:
            device_df = state["raw_data"]["device_events"]
            
            # Prepare analysis data
            analysis_data = {
                "total_devices": len(device_df),
                "event_distribution": device_df['total_events'].describe().to_dict(),
                "time_span_distribution": device_df['time_span_hours'].describe().to_dict(),
                "sample_event_patterns": state["raw_data"]["sample_events"]
            }
            
            # Create analysis prompt
            prompt = ChatPromptTemplate.from_messages([
                ("system", state["messages"][0].content),
                ("human", """Analyze the following device behavior data:
                
                Total Devices: {total_devices}
                
                Event Count Distribution:
                {event_distribution}
                
                Time Span Distribution (hours):
                {time_span_distribution}
                
                Sample Event Patterns:
                {sample_patterns}
                
                Provide detailed behavioral pattern analysis focusing on:
                1. User engagement levels (single vs multi-event users)
                2. Temporal behavior patterns
                3. Device usage intensity
                4. Notable trends or anomalies
                
                Be specific with numbers and percentages.""")
            ])
            
            # Generate analysis using LLM
            chain = prompt | self.llm
            response = chain.invoke({
                "total_devices": analysis_data["total_devices"],
                "event_distribution": json.dumps(analysis_data["event_distribution"], indent=2),
                "time_span_distribution": json.dumps(analysis_data["time_span_distribution"], indent=2),
                "sample_patterns": json.dumps(analysis_data["sample_event_patterns"], indent=2)
            })
            
            state["insights"]["behavior_analysis"] = response.content
            state["messages"].append(response)
            
            print("✅ Behavior pattern analysis completed")
            
        except Exception as e:
            state["error_message"] = f"Pattern analysis error: {str(e)}"
            print(f"❌ Error in pattern analysis: {str(e)}")
            
        return state

    def _generate_insights(self, state: AnalysisState) -> AnalysisState:
        """Generate business insights from the analysis"""
        print("\n💡 Generating Business Insights")
        state["current_step"] = "generate_insights"
        
        if state["error_message"]:
            return state
            
        try:
            device_df = state["raw_data"]["device_events"]
            summary_df = state["raw_data"].get("summary", pd.DataFrame())
            
            # Calculate key metrics
            metrics = {
                "total_devices": len(device_df),
                "avg_events_per_device": device_df['total_events'].mean(),
                "single_event_devices": (device_df['total_events'] == 1).sum(),
                "multi_event_devices": (device_df['total_events'] > 1).sum(),
                "highly_active_devices": (device_df['total_events'] >= 10).sum(),
                "avg_time_span": device_df['time_span_hours'].mean()
            }
            
            # Create insights prompt
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a business intelligence analyst specializing in user behavior metrics."),
                ("human", """Based on the device behavior analysis, generate strategic business insights:
                
                Key Metrics:
                - Total Devices: {total_devices}
                - Average Events per Device: {avg_events:.2f}
                - Single-Event Devices: {single_event} ({single_pct:.1f}%)
                - Multi-Event Devices: {multi_event} ({multi_pct:.1f}%)
                - Highly Active Devices (10+ events): {highly_active} ({active_pct:.1f}%)
                - Average Time Span: {avg_time_span:.1f} hours
                
                Previous Analysis:
                {behavior_analysis}
                
                Generate 5-7 specific business insights focusing on:
                1. User engagement quality
                2. Retention indicators
                3. User journey effectiveness
                4. Revenue opportunities
                5. Risk factors
                
                Each insight should include specific metrics and business implications.""")
            ])
            
            # Calculate percentages
            total = metrics["total_devices"]
            single_pct = (metrics["single_event_devices"] / total * 100) if total > 0 else 0
            multi_pct = (metrics["multi_event_devices"] / total * 100) if total > 0 else 0
            active_pct = (metrics["highly_active_devices"] / total * 100) if total > 0 else 0
            
            chain = prompt | self.llm
            response = chain.invoke({
                "total_devices": metrics["total_devices"],
                "avg_events": metrics["avg_events_per_device"],
                "single_event": metrics["single_event_devices"],
                "single_pct": single_pct,
                "multi_event": metrics["multi_event_devices"],
                "multi_pct": multi_pct,
                "highly_active": metrics["highly_active_devices"],
                "active_pct": active_pct,
                "avg_time_span": metrics["avg_time_span"],
                "behavior_analysis": state["insights"]["behavior_analysis"]
            })
            
            state["insights"]["business_insights"] = response.content
            state["messages"].append(response)
            
            print("✅ Business insights generated")
            
        except Exception as e:
            state["error_message"] = f"Insights generation error: {str(e)}"
            print(f"❌ Error generating insights: {str(e)}")
            
        return state

    def _create_recommendations(self, state: AnalysisState) -> AnalysisState:
        """Create actionable recommendations"""
        print("\n🎯 Creating Strategic Recommendations")
        state["current_step"] = "create_recommendations"
        
        if state["error_message"]:
            return state
            
        try:
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a strategic business consultant specializing in user engagement and growth."),
                ("human", """Based on the device behavior analysis and business insights, create 5-8 specific, 
                actionable recommendations for improving user engagement and business outcomes:
                
                Business Insights:
                {business_insights}
                
                Behavior Analysis:
                {behavior_analysis}
                
                Each recommendation should include:
                1. Specific action to take
                2. Expected impact/outcome
                3. Priority level (High/Medium/Low)
                4. Implementation timeline
                5. Success metrics to track
                
                Focus on practical, data-driven recommendations that address the identified patterns.""")
            ])
            
            chain = prompt | self.llm
            response = chain.invoke({
                "business_insights": state["insights"]["business_insights"],
                "behavior_analysis": state["insights"]["behavior_analysis"]
            })
            
            state["insights"]["recommendations"] = response.content
            state["messages"].append(response)
            
            print("✅ Strategic recommendations created")
            
        except Exception as e:
            state["error_message"] = f"Recommendations error: {str(e)}"
            print(f"❌ Error creating recommendations: {str(e)}")
            
        return state

    def _finalize_report(self, state: AnalysisState) -> AnalysisState:
        """Finalize the analysis report"""
        print("\n📋 Finalizing Analysis Report")
        state["current_step"] = "finalize_report"
        
        if state["error_message"]:
            print(f"❌ Analysis completed with errors: {state['error_message']}")
            return state
            
        # Create final summary
        device_df = state["raw_data"]["device_events"]
        total_devices = len(device_df)
        total_events = device_df['total_events'].sum()
        
        print(f"\n📊 FINAL ANALYSIS SUMMARY")
        print(f"{'='*50}")
        print(f"Total Devices Analyzed: {total_devices:,}")
        print(f"Total Events Processed: {total_events:,}")
        print(f"Average Events per Device: {total_events/total_devices:.1f}")
        print(f"Analysis Framework: LangGraph with OpenAI")
        print(f"{'='*50}")
        
        print("\n✅ LangGraph Device Behavior Analysis Completed Successfully!")
        
        return state

    def run_analysis(self) -> Dict[str, Any]:
        """Execute the complete LangGraph analysis workflow"""
        print("🚀 Starting LangGraph-Based Device Behavior Analysis")
        print("📊 This analysis uses LangGraph for reliable, stateful workflows")
        print("=" * 60)
        
        # Initialize state
        initial_state = {
            "messages": [],
            "database_path": self.db_path,
            "raw_data": {},
            "insights": {},
            "recommendations": [],
            "current_step": "",
            "error_message": ""
        }
        
        try:
            # Execute the workflow
            final_state = self.workflow.invoke(initial_state)
            
            # Return results
            return {
                "success": not bool(final_state["error_message"]),
                "insights": final_state["insights"],
                "raw_data": final_state["raw_data"],
                "error": final_state["error_message"],
                "workflow_state": final_state["current_step"]
            }
            
        except Exception as e:
            print(f"❌ Workflow execution error: {str(e)}")
            return {
                "success": False,
                "insights": {},
                "raw_data": {},
                "error": str(e),
                "workflow_state": "error"
            }

def run_langgraph_analysis(db_path="event_analysis.db"):
    """Main function to run LangGraph-based analysis"""
    try:
        # Initialize and run the LangGraph agent
        analyst = DeviceBehaviorAnalyst(db_path=db_path)
        results = analyst.run_analysis()
        
        if results["success"]:
            print("\n🎉 LangGraph Analysis Results Available!")
            print("📋 Business Insights Generated")
            print("🎯 Strategic Recommendations Created")
            print("💾 All analysis data retained in workflow state")
        else:
            print(f"\n❌ Analysis failed: {results['error']}")
            
        return results
        
    except Exception as e:
        print(f"❌ Error running LangGraph analysis: {str(e)}")
        return {"success": False, "error": str(e)}

if __name__ == "__main__":
    # Run standalone analysis
    results = run_langgraph_analysis()
    
    if results["success"]:
        print("\n📊 Analysis completed successfully!")
        print("🔍 Check the insights for detailed findings")
    else:
        print(f"\n❌ Analysis failed: {results.get('error')}") 