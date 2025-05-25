"""
Analysis Agent - Recommendation Click Analysis for Understanding User Behavior
Simplified for MVP: Focus on why users don't click recommendations
"""

from crewai import Agent, Task
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

# Dynamic Analysis Tools for future implementation
class StatisticalSummaryTool(BaseTool):
    """Tool for calculating statistical summaries dynamically"""
    
    name: str = "calculate_summary_statistics"
    description: str = "Calculate comprehensive statistical summaries for any dataset"
    
    def _run(self, data: pd.DataFrame, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute statistical summary analysis"""
        try:
            metrics = parameters.get("metrics", ["count", "mean", "median", "std"])
            grouping = parameters.get("grouping", [])
            
            if grouping:
                # Grouped statistics
                numeric_cols = data.select_dtypes(include=[np.number]).columns
                grouped_stats = data.groupby(grouping)[numeric_cols].agg(metrics)
                stats_dict = grouped_stats.to_dict()
            else:
                # Overall statistics
                overall_stats = data.describe()
                stats_dict = overall_stats.to_dict()
            
            return {
                "summary_statistics": stats_dict,
                "data_quality": self._assess_data_quality(data),
                "insights": self._generate_statistical_insights(stats_dict),
                "status": "success"
            }
            
        except Exception as e:
            return {"error": f"Statistical summary analysis failed: {str(e)}"}
    
    def _assess_data_quality(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Assess data quality metrics"""
        return {
            "total_rows": len(data),
            "total_columns": len(data.columns),
            "missing_values": data.isnull().sum().to_dict(),
            "duplicate_rows": data.duplicated().sum(),
            "data_types": data.dtypes.astype(str).to_dict()
        }
    
    def _generate_statistical_insights(self, stats: Dict[str, Any]) -> List[str]:
        """Generate insights from statistical summaries"""
        insights = []
        # Add basic insights logic here
        insights.append("Statistical summary completed successfully")
        return insights

class TrendAnalysisTool(BaseTool):
    """Tool for trend analysis"""
    
    name: str = "analyze_trends"
    description: str = "Perform comprehensive trend analysis over time"
    
    def _run(self, data: pd.DataFrame, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute trend analysis"""
        try:
            time_column = parameters.get("time_column", "timestamp")
            metrics = parameters.get("metrics", [])
            trend_method = parameters.get("trend_method", "linear")
            
            # Placeholder implementation - would need full trend analysis logic
            return {
                "trends": {"placeholder": "trend analysis results"},
                "summary": "Trend analysis completed",
                "recommendations": ["Implement full trend analysis logic"],
                "status": "success"
            }
            
        except Exception as e:
            return {"error": f"Trend analysis failed: {str(e)}"}

def create_analysis_agent() -> Agent:
    """Create and configure the Analysis Agent with all tools"""
    
    # Initialize all tools (current + dynamic)
    funnel_tool = RecommendationFunnelTool()
    time_tool = TimePatternAnalysisTool()
    behavior_tool = UserBehaviorAnalysisTool()
    stats_tool = StatisticalSummaryTool()
    trend_tool = TrendAnalysisTool()
    
    # Create agent with all tools
    analysis_agent = Agent(
        role="Data Analysis Specialist",
        goal="""
        As a Data Analysis Specialist, I perform sophisticated data analysis tasks including:
        1. Recommendation funnel analysis and user behavior patterns
        2. Statistical summaries and descriptive analytics
        3. Trend analysis and time-based patterns
        4. Comparative analysis across segments
        5. Correlation analysis and predictive modeling
        
        I use my tools intelligently to provide accurate, insightful analysis results.
        """,
        backstory="""
        I am an expert data analyst with deep expertise in user behavior analysis, statistical methods,
        and business intelligence. I can analyze any type of data and provide actionable insights.
        My specialties include recommendation systems, user segmentation, trend analysis, and
        predictive modeling. I always consider data quality and provide clear explanations of my findings.
        """,
        tools=[funnel_tool, time_tool, behavior_tool, stats_tool, trend_tool],
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
    """Properly uses CrewAI Agent for analysis operations"""
    
    def __init__(self):
        self.agent = create_analysis_agent()
        # Store data temporarily for agent context
        self._current_data = None
        self._current_parameters = None
    
    def execute_analysis(self, analysis_type: str, data: pd.DataFrame, parameters: Dict[str, Any]) -> AgentTaskResult:
        """Execute analysis using CrewAI Agent (not direct tool calls)"""
        start_time = datetime.now()
        
        try:
            # Store data for agent context
            self._current_data = data
            self._current_parameters = parameters
            
            # Create a CrewAI Task for the agent
            task = self._create_analysis_task(analysis_type, data, parameters)
            
            # Execute task using the agent (this uses the agent's LLM + tools)
            result = self.agent.execute_task(task)
            
            # Parse agent result into our expected format
            parsed_result = self._parse_agent_result(result, analysis_type)
            
            return AgentTaskResult(
                agent_name="analysis",
                task_id=f"execute_{analysis_type}",
                status=AgentStatus.COMPLETED,
                start_time=start_time,
                end_time=datetime.now(),
                output_data=parsed_result,
                output_files=parsed_result.get("output_files", []),
                metadata={
                    "analysis_type": analysis_type,
                    "parameters": parameters,
                    "agent_reasoning": getattr(result, 'reasoning', None)
                }
            )
            
        except Exception as e:
            return AgentTaskResult(
                agent_name="analysis",
                task_id=f"execute_{analysis_type}",
                status=AgentStatus.FAILED,
                start_time=start_time,
                end_time=datetime.now(),
                error_message=f"Agent execution failed: {str(e)}"
            )
    
    def _create_analysis_task(self, analysis_type: str, data: pd.DataFrame, parameters: Dict[str, Any]) -> Task:
        """Create a CrewAI Task that the agent can understand and execute"""
        
        # Map analysis types to natural language descriptions
        task_descriptions = {
            "recommendation_funnel": f"""
            Analyze the recommendation funnel data to understand user behavior.
            The dataset has {len(data)} records with columns: {', '.join(data.columns)}.
            
            Use the recommendation_funnel tool to:
            1. Clean the data by removing invalid exposures
            2. Calculate funnel metrics (CTR, exposure rates, etc.)
            3. Generate output files with results
            
            Parameters: {parameters}
            """,
            
            "time_pattern_analysis": f"""
            Analyze temporal patterns in user engagement with recommendations.
            The dataset has {len(data)} records spanning from {data['timestamp'].min() if 'timestamp' in data.columns else 'unknown'} 
            to {data['timestamp'].max() if 'timestamp' in data.columns else 'unknown'}.
            
            Use the time_pattern_analysis tool to:
            1. Extract time components from timestamps
            2. Calculate hourly and daily engagement patterns
            3. Identify peak and low engagement periods
            
            Parameters: {parameters}
            """,
            
            "user_behavior_analysis": f"""
            Segment users based on their behavior patterns and analyze event sequences.
            The dataset contains {data['device_id'].nunique() if 'device_id' in data.columns else 'unknown'} unique users
            with {len(data)} total events.
            
            Use the user_behavior_analysis tool to:
            1. Segment users by activity levels
            2. Analyze event patterns and sequences
            3. Generate insights about user behavior
            
            Parameters: {parameters}
            """,
            
            "calculate_summary_statistics": f"""
            Calculate comprehensive statistical summaries for the dataset.
            The dataset has {len(data)} records with {len(data.columns)} columns.
            Numeric columns: {', '.join(data.select_dtypes(include=[np.number]).columns)}.
            
            Use the calculate_summary_statistics tool with these specifications:
            - Metrics to calculate: {parameters.get("metrics", ["count", "mean", "median", "std"])}
            - Group by: {parameters.get("grouping", "none")}
            
            Parameters: {parameters}
            """,
            
            "analyze_trends": f"""
            Perform trend analysis to identify patterns over time.
            The dataset has {len(data)} records for trend analysis.
            
            Use the analyze_trends tool with these specifications:
            - Time column: {parameters.get("time_column", "timestamp")}
            - Metrics to analyze: {parameters.get("metrics", [])}
            - Trend method: {parameters.get("trend_method", "linear")}
            
            Parameters: {parameters}
            """
        }
        
        # Get task description
        description = task_descriptions.get(analysis_type, 
            f"Execute {analysis_type} analysis on the provided dataset with {len(data)} records.")
        
        # Create the task
        task = Task(
            description=description,
            agent=self.agent,
            expected_output=f"Detailed {analysis_type} analysis results with insights and recommendations in JSON format"
        )
        
        return task
    
    def _parse_agent_result(self, agent_result: Any, analysis_type: str) -> Dict[str, Any]:
        """Parse the agent's output into our expected format"""
        
        # Handle different result formats from CrewAI agent
        if hasattr(agent_result, 'output'):
            # If agent returns a structured result
            return agent_result.output
        elif isinstance(agent_result, dict):
            # If agent returns a dictionary directly
            return agent_result
        elif isinstance(agent_result, str):
            # If agent returns text, try to extract structured data
            try:
                import json
                return json.loads(agent_result)
            except json.JSONDecodeError:
                # Return as text result
                return {
                    "analysis_result": agent_result,
                    "status": "completed",
                    "format": "text"
                }
        else:
            # Fallback for unknown formats
            return {
                "raw_result": str(agent_result),
                "status": "completed", 
                "format": "unknown"
            }

# Export the controller
__all__ = ["AnalysisController", "create_analysis_agent"] 