"""
Insight Agent - Recommendation Click Insights for Understanding User Behavior
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
    AgentTaskResult, AgentStatus, RecommendationInsight
)

class RecommendationInsightTool(BaseTool):
    """Tool for generating insights about recommendation click behavior"""
    
    name: str = "recommendation_insights"
    description: str = "Generate business insights about why users don't click recommendations"
    
    def _run(self, analysis_results: Dict[str, Any], analysis_type: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate insights based on recommendation analysis results"""
        try:
            if analysis_type == "recommendation_funnel":
                return self._generate_funnel_insights(analysis_results, context)
            elif analysis_type == "time_pattern_analysis":
                return self._generate_time_insights(analysis_results, context)
            elif analysis_type == "user_behavior_analysis":
                return self._generate_behavior_insights(analysis_results, context)
            else:
                return {"error": f"Unsupported analysis type for insight generation: {analysis_type}"}
                
        except Exception as e:
            return {"error": f"Insight generation failed: {str(e)}"}
    
    def _generate_funnel_insights(self, results: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate insights for recommendation funnel analysis"""
        insights = []
        recommendations = []
        
        funnel_metrics = results.get("funnel_metrics", {})
        
        # Extract key metrics
        click_through_rate = funnel_metrics.get("click_through_rate", 0)
        exposure_users = funnel_metrics.get("exposure_users", 0)
        no_click_users = funnel_metrics.get("no_click_users", 0)
        avg_exposures_per_user = funnel_metrics.get("avg_exposures_per_user", 0)
        
        # Generate CTR insights
        if click_through_rate < 0.05:  # Less than 5%
            insights.append(RecommendationInsight(
                insight_type="performance_issue",
                title="Very Low Click-Through Rate",
                description=f"CTR of {click_through_rate:.1%} is significantly below industry standards (8-12%). This indicates major issues with recommendation relevance or presentation.",
                confidence="high",
                impact="High - Major revenue opportunity loss",
                recommendation="Immediate review of recommendation algorithm and UI/UX design",
                supporting_metrics={"ctr": click_through_rate, "industry_benchmark": 0.10},
                affected_users=no_click_users
            ))
            
            recommendations.append({
                "priority": "critical",
                "category": "algorithm",
                "title": "Recommendation Algorithm Review",
                "description": "Audit recommendation algorithm for relevance and personalization issues",
                "expected_impact": f"Potential to increase CTR to 8-10% (+{(0.08-click_through_rate)*100:.1f}%)",
                "implementation_effort": "high"
            })
            
        elif click_through_rate < 0.08:  # Less than 8%
            insights.append(RecommendationInsight(
                insight_type="optimization_opportunity",
                title="Below Average Click-Through Rate",
                description=f"CTR of {click_through_rate:.1%} is below industry average. There's room for improvement in recommendation quality or presentation.",
                confidence="high",
                impact="Medium - Moderate improvement opportunity",
                recommendation="Optimize recommendation relevance and UI presentation",
                supporting_metrics={"ctr": click_through_rate, "benchmark": 0.10},
                affected_users=no_click_users
            ))
            
        else:
            insights.append(RecommendationInsight(
                insight_type="performance_positive",
                title="Good Click-Through Rate Performance",
                description=f"CTR of {click_through_rate:.1%} meets or exceeds industry standards. Focus on maintaining quality.",
                confidence="high", 
                impact="Positive - Good baseline performance",
                recommendation="Monitor performance and test incremental improvements",
                supporting_metrics={"ctr": click_through_rate}
            ))
        
        # Exposure frequency insights
        if avg_exposures_per_user > 5:
            insights.append(RecommendationInsight(
                insight_type="frequency_issue",
                title="High Exposure Frequency Without Clicks",
                description=f"Users see {avg_exposures_per_user:.1f} recommendations on average but don't click. This suggests recommendation fatigue or irrelevance.",
                confidence="medium",
                impact="Medium - User experience degradation",
                recommendation="Reduce recommendation frequency or improve targeting",
                supporting_metrics={"avg_exposures": avg_exposures_per_user}
            ))
            
            recommendations.append({
                "priority": "medium",
                "category": "frequency",
                "title": "Optimize Recommendation Frequency",
                "description": "Implement frequency capping or improve recommendation targeting to reduce over-exposure",
                "expected_impact": "Reduce recommendation fatigue, improve user experience",
                "implementation_effort": "medium"
            })
        
        # Data quality insights
        cleaned_stats = results.get("cleaned_data_stats", {})
        removed_events = cleaned_stats.get("removed_events", 0)
        original_events = cleaned_stats.get("original_events", 1)
        
        if removed_events / original_events > 0.1:  # More than 10% removed
            insights.append(RecommendationInsight(
                insight_type="data_quality_issue",
                title="High Invalid Data Rate",
                description=f"Removed {removed_events:,} invalid events ({removed_events/original_events:.1%}). Many users open app without proper recommendation exposure.",
                confidence="high",
                impact="High - Data integrity and tracking issues",
                recommendation="Fix recommendation tracking and exposure logging",
                supporting_metrics={"removed_ratio": removed_events/original_events}
            ))
        
        return {
            "insights": [insight.dict() for insight in insights],
            "recommendations": recommendations,
            "key_metrics": {
                "click_through_rate": click_through_rate,
                "total_exposure_users": exposure_users,
                "no_click_users": no_click_users,
                "avg_exposures_per_user": avg_exposures_per_user
            },
            "summary": self._generate_funnel_summary(funnel_metrics, insights),
            "status": "success"
        }
    
    def _generate_time_insights(self, results: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate insights for time pattern analysis"""
        insights = []
        recommendations = []
        
        hourly_patterns = results.get("hourly_patterns", {})
        daily_patterns = results.get("daily_patterns", {})
        
        peak_hour = hourly_patterns.get("peak_hour")
        low_hour = hourly_patterns.get("low_hour") 
        peak_day = daily_patterns.get("peak_day")
        low_day = daily_patterns.get("low_day")
        
        # Hourly insights
        if peak_hour is not None and low_hour is not None:
            insights.append(RecommendationInsight(
                insight_type="time_optimization",
                title="Clear Hourly Engagement Patterns",
                description=f"Users are most active at {peak_hour}:00 and least active at {low_hour}:00. This suggests optimal timing opportunities.",
                confidence="high",
                impact="Medium - Timing optimization opportunity",
                recommendation="Schedule recommendation delivery during peak hours",
                supporting_metrics={"peak_hour": peak_hour, "low_hour": low_hour}
            ))
            
            recommendations.append({
                "priority": "medium",
                "category": "timing",
                "title": "Optimize Recommendation Timing",
                "description": f"Increase recommendation frequency during peak hours ({peak_hour}:00) and reduce during low hours ({low_hour}:00)",
                "expected_impact": "15-25% improvement in engagement rates",
                "implementation_effort": "low"
            })
        
        # Daily insights  
        if peak_day and low_day:
            insights.append(RecommendationInsight(
                insight_type="weekly_pattern",
                title="Weekly Engagement Patterns Identified",
                description=f"{peak_day} shows highest engagement while {low_day} shows lowest. Consider different strategies for different days.",
                confidence="medium",
                impact="Medium - Day-specific optimization",
                recommendation="Implement day-specific recommendation strategies",
                supporting_metrics={"peak_day": peak_day, "low_day": low_day}
            ))
            
            if low_day == "Monday":
                recommendations.append({
                    "priority": "medium",
                    "category": "content",
                    "title": "Monday Engagement Strategy",
                    "description": "Develop specific content strategy for Monday to combat 'Monday blues' effect",
                    "expected_impact": "Improve Monday engagement rates",
                    "implementation_effort": "medium"
                })
        
        return {
            "insights": [insight.dict() for insight in insights],
            "recommendations": recommendations,
            "key_metrics": {
                "peak_hour": peak_hour,
                "low_hour": low_hour,
                "peak_day": peak_day,
                "low_day": low_day
            },
            "summary": self._generate_time_summary(hourly_patterns, daily_patterns, insights),
            "status": "success"
        }
    
    def _generate_behavior_insights(self, results: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate insights for user behavior analysis"""
        insights = []
        recommendations = []
        
        user_segments = results.get("user_segments", {})
        event_patterns = results.get("event_patterns", {})
        
        # Segment insights
        segment_distribution = user_segments.get("segment_distribution", [])
        
        for segment in segment_distribution:
            segment_name = segment.get("segment", "Unknown")
            user_count = segment.get("user_count", 0)
            avg_events = segment.get("avg_events", 0)
            
            if "High Activity" in segment_name and user_count > 0:
                insights.append(RecommendationInsight(
                    insight_type="segment_opportunity",
                    title="High-Activity User Segment Identified",
                    description=f"High-activity users ({user_count:,} users) show {avg_events:.1f} events on average. They may be good candidates for personalized recommendations.",
                    confidence="medium",
                    impact="Medium - Personalization opportunity",
                    recommendation="Develop specialized recommendation strategy for high-activity users",
                    supporting_metrics={"segment_size": user_count, "avg_events": avg_events},
                    affected_users=user_count
                ))
                
                recommendations.append({
                    "priority": "medium",
                    "category": "personalization",
                    "title": "High-Activity User Personalization",
                    "description": "Create premium or advanced recommendation features for high-activity users",
                    "expected_impact": "Increase engagement in high-value user segment",
                    "implementation_effort": "high"
                })
            
            elif "Low Activity" in segment_name and user_count > 0:
                insights.append(RecommendationInsight(
                    insight_type="retention_risk",
                    title="Large Low-Activity User Segment",
                    description=f"Low-activity users ({user_count:,} users) with {avg_events:.1f} events on average may be at risk of churn.",
                    confidence="medium",
                    impact="High - Retention risk",
                    recommendation="Implement re-engagement strategies for low-activity users",
                    supporting_metrics={"segment_size": user_count, "avg_events": avg_events},
                    affected_users=user_count
                ))
                
                recommendations.append({
                    "priority": "high",
                    "category": "retention",
                    "title": "Low-Activity User Re-engagement",
                    "description": "Develop onboarding and re-engagement campaigns for low-activity users",
                    "expected_impact": "Reduce churn risk, improve overall engagement",
                    "implementation_effort": "medium"
                })
        
        # Event pattern insights
        top_events = event_patterns.get("top_events", {})
        if top_events:
            most_common_event = max(top_events.items(), key=lambda x: x[1])
            insights.append(RecommendationInsight(
                insight_type="event_pattern",
                title="Dominant Event Pattern Identified",
                description=f"Most common event: '{most_common_event[0]}' ({most_common_event[1]:,} occurrences). Understanding this pattern can inform recommendation strategy.",
                confidence="high",
                impact="Low - Informational",
                recommendation="Analyze context around dominant events for recommendation opportunities",
                supporting_metrics={"top_event": most_common_event[0], "frequency": most_common_event[1]}
            ))
        
        return {
            "insights": [insight.dict() for insight in insights],
            "recommendations": recommendations,
            "key_metrics": {
                "total_segments": len(segment_distribution),
                "top_event": max(top_events.items(), key=lambda x: x[1])[0] if top_events else "Unknown"
            },
            "summary": self._generate_behavior_summary(user_segments, event_patterns, insights),
            "status": "success"
        }
    
    def _generate_funnel_summary(self, metrics: Dict[str, Any], insights: List[RecommendationInsight]) -> str:
        """Generate executive summary for funnel analysis"""
        ctr = metrics.get("click_through_rate", 0)
        exposure_users = metrics.get("exposure_users", 0)
        
        performance_level = "excellent" if ctr > 0.12 else "good" if ctr > 0.08 else "needs improvement"
        
        summary = f"""
## Executive Summary: Recommendation Funnel Analysis

**Key Metrics:**
- Click-Through Rate: {ctr:.1%}
- Users with Exposures: {exposure_users:,}
- Performance Level: {performance_level.title()}

**Key Insights:**
Overall recommendation performance {performance_level} with {len(insights)} actionable insights identified.
The analysis reveals specific opportunities to improve user engagement with recommended content.
        """.strip()
        
        return summary
    
    def _generate_time_summary(self, hourly: Dict[str, Any], daily: Dict[str, Any], insights: List[RecommendationInsight]) -> str:
        """Generate executive summary for time pattern analysis"""
        peak_hour = hourly.get("peak_hour", "N/A")
        peak_day = daily.get("peak_day", "N/A")
        
        summary = f"""
## Executive Summary: Time Pattern Analysis

**Optimal Timing:**
- Peak Hour: {peak_hour}:00
- Peak Day: {peak_day}
- Insights Generated: {len(insights)}

**Strategic Opportunity:**
Clear time-based patterns identified that can be leveraged to optimize recommendation delivery timing and improve engagement rates.
        """.strip()
        
        return summary
    
    def _generate_behavior_summary(self, segments: Dict[str, Any], patterns: Dict[str, Any], insights: List[RecommendationInsight]) -> str:
        """Generate executive summary for behavior analysis"""
        total_users = segments.get("total_users", 0)
        unique_events = patterns.get("unique_events", 0)
        
        summary = f"""
## Executive Summary: User Behavior Analysis

**Segmentation Results:**
- Total Users Analyzed: {total_users:,}
- Unique Event Types: {unique_events}
- Behavioral Insights: {len(insights)}

**Strategic Value:**
User segmentation reveals distinct behavioral patterns that can be leveraged for personalized recommendation strategies and improved user experience.
        """.strip()
        
        return summary

def create_insight_agent() -> Agent:
    """Create and configure the simplified Insight Agent for recommendation analysis"""
    
    # Initialize simplified tool
    insight_tool = RecommendationInsightTool()
    
    # Create agent
    insight_agent = Agent(
        role="Recommendation Insights Specialist",
        goal="""
        As a Recommendation Insights Specialist, I focus specifically on:
        1. Understanding why users don't click on recommended content
        2. Generating actionable insights from recommendation analysis results  
        3. Identifying optimization opportunities for click-through rates
        4. Providing specific, implementable recommendations for improvement
        5. Creating clear summaries focused on recommendation performance
        """,
        backstory="""
        I am a specialist in recommendation system optimization with deep expertise in 
        user engagement analysis. I focus specifically on understanding user behavior 
        around recommendations and translating data insights into actionable business 
        strategies. My expertise includes click-through rate optimization, user 
        segmentation for recommendations, and time-based engagement patterns.
        """,
        tools=[insight_tool],
        verbose=settings.agent.verbose,
        allow_delegation=False,
        max_execution_time=settings.agent.max_execution_time,
        llm_config={
            "model": settings.openai.model,
            "temperature": settings.openai.temperature,
            "max_tokens": settings.openai.max_tokens
        }
    )
    
    return insight_agent

class InsightController:
    """Simplified controller for Recommendation Insight operations"""
    
    def __init__(self):
        self.agent = create_insight_agent()
    
    def generate_insights(self, analysis_results: Dict[str, Any], analysis_type: str, context: Dict[str, Any] = None) -> AgentTaskResult:
        """Generate recommendation-specific insights and recommendations"""
        start_time = datetime.now()
        
        try:
            # Generate insights
            insight_tool = RecommendationInsightTool()
            insights_result = insight_tool._run(analysis_results, analysis_type, context)
            
            if "error" in insights_result:
                return AgentTaskResult(
                    agent_name="insight",
                    task_id="generate_insights",
                    status=AgentStatus.FAILED,
                    start_time=start_time,
                    end_time=datetime.now(),
                    error_message=insights_result["error"]
                )
            
            # Save insights to file
            output_files = self._save_insights(insights_result, analysis_type)
            insights_result["output_files"] = output_files
            
            return AgentTaskResult(
                agent_name="insight",
                task_id="generate_insights",
                status=AgentStatus.COMPLETED,
                start_time=start_time,
                end_time=datetime.now(),
                output_data=insights_result,
                output_files=output_files,
                metadata={
                    "analysis_type": analysis_type,
                    "insights_count": len(insights_result.get("insights", [])),
                    "recommendations_count": len(insights_result.get("recommendations", []))
                }
            )
            
        except Exception as e:
            return AgentTaskResult(
                agent_name="insight",
                task_id="generate_insights",
                status=AgentStatus.FAILED,
                start_time=start_time,
                end_time=datetime.now(),
                error_message=f"Insight generation failed: {str(e)}"
            )
    
    def _save_insights(self, insights_result: Dict[str, Any], analysis_type: str) -> List[str]:
        """Save insights to files"""
        output_dir = Path(settings.paths.output_data_path)
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_files = []
        
        # Save insights as JSON
        insights_file = output_dir / f"recommendation_insights_{analysis_type}_{timestamp}.json"
        with open(insights_file, 'w') as f:
            json.dump(insights_result, f, indent=2, default=str)
        output_files.append(str(insights_file))
        
        # Save summary as markdown
        if "summary" in insights_result:
            summary_file = output_dir / f"recommendation_summary_{analysis_type}_{timestamp}.md"
            with open(summary_file, 'w') as f:
                f.write(insights_result["summary"])
            output_files.append(str(summary_file))
        
        return output_files

# Export the controller
__all__ = ["InsightController", "create_insight_agent"] 