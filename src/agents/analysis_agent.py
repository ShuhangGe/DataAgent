"""
Analysis Agent - Recommendation Click Analysis for Understanding User Behavior
Simplified for MVP: Focus on why users don't click recommendations
"""

from crewai import Agent
from crewai.tools import BaseTool
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from pathlib import Path

from src.config.settings import settings
from src.models.data_models import (
    AnalysisType, ValidationResult, ValidationLevel, 
    AgentTaskResult, AgentStatus, RecommendationAnalysisResult
)

class RecommendationFunnelTool(BaseTool):
    """Tool for analyzing recommendation view to click funnel"""
    
    name: str = "recommendation_funnel"
    description: str = "Analyze user journey from viewing recommendations to (not) clicking"
    
    def _run(self, data: pd.DataFrame, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute recommendation funnel analysis"""
        try:
            # Ensure required columns exist (MVP: only timestamp and event)
            required_cols = ["device_id", "timestamp", "event"]
            missing_cols = [col for col in required_cols if col not in data.columns]
            if missing_cols:
                return {"error": f"Missing required columns: {missing_cols}"}
            
            # Convert timestamp to datetime and ensure UTC
            data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True)
            
            # Clean data: Remove users who opened app without exposure
            cleaned_data = self._clean_invalid_exposures(data)
            
            # Calculate funnel metrics
            funnel_metrics = self._calculate_funnel_metrics(cleaned_data)
            
            # Generate output files
            output_files = self._save_funnel_results(funnel_metrics, cleaned_data)
            
            return {
                "funnel_metrics": funnel_metrics,
                "cleaned_data_stats": {
                    "original_events": len(data),
                    "valid_events": len(cleaned_data),
                    "removed_events": len(data) - len(cleaned_data)
                },
                "output_files": output_files,
                "status": "success"
            }
            
        except Exception as e:
            return {"error": f"Recommendation funnel analysis failed: {str(e)}"}
    
    def _clean_invalid_exposures(self, data: pd.DataFrame) -> pd.DataFrame:
        """Remove users who opened app without receiving exposure cards"""
        # Group by device_id to analyze user sessions
        user_sessions = []
        
        for device_id in data["device_id"].unique():
            user_data = data[data["device_id"] == device_id].sort_values("timestamp")
            
            # Check if user has valid exposure flow
            has_exposure = any("recommendation" in event.lower() or "exposure" in event.lower() 
                             for event in user_data["event"].values)
            
            # Only include users who have valid recommendation exposures
            if has_exposure:
                user_sessions.append(user_data)
        
        if user_sessions:
            return pd.concat(user_sessions, ignore_index=True)
        else:
            return pd.DataFrame(columns=data.columns)
    
    def _calculate_funnel_metrics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate recommendation funnel metrics"""
        if data.empty:
            return {
                "total_users": 0,
                "exposure_users": 0,
                "click_users": 0,
                "no_click_users": 0,
                "click_through_rate": 0.0,
                "avg_exposures_per_user": 0.0
            }
        
        # Identify event types
        exposure_events = data[data["event"].str.contains("recommendation|exposure", case=False, na=False)]
        click_events = data[data["event"].str.contains("click|tap", case=False, na=False)]
        
        # Calculate metrics
        total_users = data["device_id"].nunique()
        exposure_users = exposure_events["device_id"].nunique()
        click_users = click_events["device_id"].nunique()
        no_click_users = exposure_users - click_users
        
        click_through_rate = click_users / exposure_users if exposure_users > 0 else 0.0
        avg_exposures_per_user = len(exposure_events) / exposure_users if exposure_users > 0 else 0.0
        
        return {
            "total_users": total_users,
            "exposure_users": exposure_users,
            "click_users": click_users,
            "no_click_users": no_click_users,
            "click_through_rate": click_through_rate,
            "avg_exposures_per_user": avg_exposures_per_user,
            "total_exposures": len(exposure_events),
            "total_clicks": len(click_events)
        }
    
    def _save_funnel_results(self, metrics: Dict[str, Any], data: pd.DataFrame) -> List[str]:
        """Save funnel analysis results"""
        output_dir = Path(settings.paths.output_data_path)
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_files = []
        
        # Save funnel metrics
        metrics_file = output_dir / f"recommendation_funnel_{timestamp}.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2, default=str)
        output_files.append(str(metrics_file))
        
        # Save user-level data
        if not data.empty:
            user_summary = data.groupby("device_id").agg({
                "event": "count",
                "timestamp": ["min", "max"]
            }).reset_index()
            user_summary.columns = ["device_id", "total_events", "first_event", "last_event"]
            
            user_file = output_dir / f"user_behavior_{timestamp}.csv"
            user_summary.to_csv(user_file, index=False)
            output_files.append(str(user_file))
        
        return output_files

class TimePatternAnalysisTool(BaseTool):
    """Tool for analyzing when users are most/least likely to engage with recommendations"""
    
    name: str = "time_pattern_analysis"
    description: str = "Analyze time-based patterns in recommendation engagement"
    
    def _run(self, data: pd.DataFrame, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute time pattern analysis"""
        try:
            if data.empty:
                return {"error": "No data available for time pattern analysis"}
            
            # Ensure timestamp is datetime
            data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True)
            
            # Add time components
            data["hour"] = data["timestamp"].dt.hour
            data["day_of_week"] = data["timestamp"].dt.day_name()
            data["date"] = data["timestamp"].dt.date
            
            # Analyze patterns
            hourly_patterns = self._analyze_hourly_patterns(data)
            daily_patterns = self._analyze_daily_patterns(data)
            
            # Generate output files
            output_files = self._save_time_patterns(hourly_patterns, daily_patterns)
            
            return {
                "hourly_patterns": hourly_patterns,
                "daily_patterns": daily_patterns,
                "output_files": output_files,
                "status": "success"
            }
            
        except Exception as e:
            return {"error": f"Time pattern analysis failed: {str(e)}"}
    
    def _analyze_hourly_patterns(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze hourly patterns in user behavior"""
        hourly_stats = data.groupby("hour").agg({
            "device_id": "nunique",
            "event": "count"
        }).reset_index()
        hourly_stats.columns = ["hour", "unique_users", "total_events"]
        
        # Calculate engagement rate by hour
        hourly_stats["events_per_user"] = hourly_stats["total_events"] / hourly_stats["unique_users"]
        
        return {
            "hourly_distribution": hourly_stats.to_dict("records"),
            "peak_hour": hourly_stats.loc[hourly_stats["events_per_user"].idxmax(), "hour"],
            "low_hour": hourly_stats.loc[hourly_stats["events_per_user"].idxmin(), "hour"]
        }
    
    def _analyze_daily_patterns(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze daily patterns in user behavior"""
        daily_stats = data.groupby("day_of_week").agg({
            "device_id": "nunique",
            "event": "count"
        }).reset_index()
        daily_stats.columns = ["day_of_week", "unique_users", "total_events"]
        
        # Calculate engagement rate by day
        daily_stats["events_per_user"] = daily_stats["total_events"] / daily_stats["unique_users"]
        
        return {
            "daily_distribution": daily_stats.to_dict("records"),
            "peak_day": daily_stats.loc[daily_stats["events_per_user"].idxmax(), "day_of_week"],
            "low_day": daily_stats.loc[daily_stats["events_per_user"].idxmin(), "day_of_week"]
        }
    
    def _save_time_patterns(self, hourly: Dict[str, Any], daily: Dict[str, Any]) -> List[str]:
        """Save time pattern analysis results"""
        output_dir = Path(settings.paths.output_data_path)
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_files = []
        
        # Save time patterns
        patterns_file = output_dir / f"time_patterns_{timestamp}.json"
        with open(patterns_file, 'w') as f:
            json.dump({"hourly": hourly, "daily": daily}, f, indent=2, default=str)
        output_files.append(str(patterns_file))
        
        return output_files

class UserBehaviorAnalysisTool(BaseTool):
    """Tool for analyzing user behavior patterns with recommendations"""
    
    name: str = "user_behavior_analysis"
    description: str = "Analyze user behavior patterns to understand recommendation engagement"
    
    def _run(self, data: pd.DataFrame, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute user behavior analysis"""
        try:
            if data.empty:
                return {"error": "No data available for user behavior analysis"}
            
            # Analyze user segments
            user_segments = self._analyze_user_segments(data)
            
            # Analyze event sequences
            event_patterns = self._analyze_event_patterns(data)
            
            # Generate output files
            output_files = self._save_behavior_analysis(user_segments, event_patterns)
            
            return {
                "user_segments": user_segments,
                "event_patterns": event_patterns,
                "output_files": output_files,
                "status": "success"
            }
            
        except Exception as e:
            return {"error": f"User behavior analysis failed: {str(e)}"}
    
    def _analyze_user_segments(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Segment users based on their engagement patterns"""
        user_stats = data.groupby("device_id").agg({
            "event": "count",
            "timestamp": ["min", "max"]
        }).reset_index()
        user_stats.columns = ["device_id", "total_events", "first_activity", "last_activity"]
        
        # Calculate activity span
        user_stats["activity_span_hours"] = (
            pd.to_datetime(user_stats["last_activity"]) - pd.to_datetime(user_stats["first_activity"])
        ).dt.total_seconds() / 3600
        
        # Simple segmentation based on activity
        user_stats["segment"] = pd.cut(
            user_stats["total_events"], 
            bins=[0, 1, 5, 20, float('inf')], 
            labels=["Single Event", "Low Activity", "Medium Activity", "High Activity"]
        )
        
        segment_summary = user_stats.groupby("segment").agg({
            "device_id": "count",
            "total_events": "mean",
            "activity_span_hours": "mean"
        }).reset_index()
        segment_summary.columns = ["segment", "user_count", "avg_events", "avg_span_hours"]
        
        return {
            "segment_distribution": segment_summary.to_dict("records"),
            "total_users": len(user_stats)
        }
    
    def _analyze_event_patterns(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze common event sequences and patterns"""
        # Count event types
        event_counts = data["event"].value_counts().to_dict()
        
        # Analyze most common events
        top_events = dict(list(event_counts.items())[:10])
        
        return {
            "event_distribution": event_counts,
            "top_events": top_events,
            "unique_events": len(event_counts),
            "total_events": len(data)
        }
    
    def _save_behavior_analysis(self, segments: Dict[str, Any], patterns: Dict[str, Any]) -> List[str]:
        """Save user behavior analysis results"""
        output_dir = Path(settings.paths.output_data_path)
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_files = []
        
        # Save behavior analysis
        behavior_file = output_dir / f"user_behavior_{timestamp}.json"
        with open(behavior_file, 'w') as f:
            json.dump({"segments": segments, "patterns": patterns}, f, indent=2, default=str)
        output_files.append(str(behavior_file))
        
        return output_files

def create_analysis_agent() -> Agent:
    """Create and configure the simplified Analysis Agent for recommendation analysis"""
    
    # Initialize simplified tools
    funnel_tool = RecommendationFunnelTool()
    time_tool = TimePatternAnalysisTool()
    behavior_tool = UserBehaviorAnalysisTool()
    
    # Create agent
    analysis_agent = Agent(
        role="Recommendation Analysis Specialist",
        goal="""
        As a Recommendation Analysis Specialist, I focus specifically on:
        1. Understanding why users don't click on recommended content
        2. Analyzing user behavior patterns around recommendations
        3. Identifying time-based patterns in recommendation engagement
        4. Cleaning invalid data (users without proper exposure)
        5. Providing actionable insights for improving recommendation CTR
        """,
        backstory="""
        I am a specialist in recommendation system analysis with deep expertise in understanding
        user behavior patterns. I focus specifically on analyzing why users view but don't click
        on recommended content. My expertise includes funnel analysis, time-based behavior patterns,
        and data cleaning for recommendation systems.
        """,
        tools=[funnel_tool, time_tool, behavior_tool],
        verbose=settings.agent.verbose,
        allow_delegation=False,
        max_execution_time=settings.agent.max_execution_time,
        llm_config={
            "model": settings.openai.model,
            "temperature": settings.openai.temperature,
            "max_tokens": settings.openai.max_tokens
        }
    )
    
    return analysis_agent

class AnalysisController:
    """Simplified controller for Recommendation Analysis operations"""
    
    def __init__(self):
        self.agent = create_analysis_agent()
    
    def execute_analysis(self, analysis_type: str, data: pd.DataFrame, parameters: Dict[str, Any]) -> AgentTaskResult:
        """Execute recommendation analysis based on type"""
        start_time = datetime.now()
        
        try:
            if analysis_type == "recommendation_funnel":
                tool = RecommendationFunnelTool()
                result = tool._run(data, parameters)
            elif analysis_type == "time_pattern_analysis":
                tool = TimePatternAnalysisTool()
                result = tool._run(data, parameters)
            elif analysis_type == "user_behavior_analysis":
                tool = UserBehaviorAnalysisTool()
                result = tool._run(data, parameters)
            else:
                return AgentTaskResult(
                    agent_name="analysis",
                    task_id=f"execute_{analysis_type}",
                    status=AgentStatus.FAILED,
                    start_time=start_time,
                    end_time=datetime.now(),
                    error_message=f"Unsupported analysis type: {analysis_type}"
                )
            
            if "error" in result:
                return AgentTaskResult(
                    agent_name="analysis",
                    task_id=f"execute_{analysis_type}",
                    status=AgentStatus.FAILED,
                    start_time=start_time,
                    end_time=datetime.now(),
                    error_message=result["error"]
                )
            
            return AgentTaskResult(
                agent_name="analysis",
                task_id=f"execute_{analysis_type}",
                status=AgentStatus.COMPLETED,
                start_time=start_time,
                end_time=datetime.now(),
                output_data=result,
                output_files=result.get("output_files", []),
                metadata={
                    "analysis_type": analysis_type,
                    "parameters": parameters
                }
            )
            
        except Exception as e:
            return AgentTaskResult(
                agent_name="analysis",
                task_id=f"execute_{analysis_type}",
                status=AgentStatus.FAILED,
                start_time=start_time,
                end_time=datetime.now(),
                error_message=f"Analysis execution failed: {str(e)}"
            )

# Export the controller
__all__ = ["AnalysisController", "create_analysis_agent"] 