"""
Dynamic Multi-Agent System - Agent Registry and Management
Built with CrewAI framework for flexible question-answering
"""

from typing import Dict, List, Any, Optional

# Import agent controllers
from .manager_agent import DynamicManagerController
from .data_pulling_agent import DataPullingController  
from .preprocessing_agent import PreprocessingController
from .analysis_agent import AnalysisController
from .qa_agent import QAController
from .insight_agent import InsightController

# Agent Information Registry for Dynamic Q&A System
AGENT_INFO = {
    "manager": {
        "name": "Dynamic Manager",
        "role": "Question-Answering Orchestrator", 
        "description": "Understands natural language questions and coordinates specialized agents to provide answers",
        "controller": DynamicManagerController,
        "capabilities": [
            "Natural language understanding",
            "Database schema inspection", 
            "Dynamic workflow planning",
            "Agent coordination",
            "Question type detection"
        ],
        "input_types": ["natural_language_questions"],
        "output_types": ["execution_plans", "analysis_workflows"],
        "priority": 1
    },
    "data_pulling": {
        "name": "Data Extraction Specialist",
        "role": "Dynamic Data Retrieval",
        "description": "Extracts relevant data from various sources based on question requirements", 
        "controller": DataPullingController,
        "capabilities": [
            "Database querying",
            "Multi-source data extraction",
            "Schema-aware data retrieval",
            "Selective column extraction",
            "Time-based filtering"
        ],
        "input_types": ["database_queries", "extraction_parameters"],
        "output_types": ["structured_data", "data_samples"],
        "priority": 2
    },
    "preprocessing": {
        "name": "Data Preparation Specialist", 
        "role": "Adaptive Data Processing",
        "description": "Prepares and cleans data based on analysis requirements",
        "controller": PreprocessingController,
        "capabilities": [
            "Data cleaning",
            "Feature engineering", 
            "Data type conversion",
            "Missing value handling",
            "Data validation"
        ],
        "input_types": ["raw_data", "processing_requirements"],
        "output_types": ["clean_data", "feature_sets"],
        "priority": 3
    },
    "analysis": {
        "name": "Dynamic Analysis Specialist",
        "role": "Question-Driven Analytics",
        "description": "Performs various types of analysis based on question type and requirements",
        "controller": AnalysisController,
        "capabilities": [
            "Statistical analysis",
            "Trend analysis",
            "Comparative analysis", 
            "Correlation analysis",
            "Predictive modeling",
            "Custom analytics"
        ],
        "input_types": ["prepared_data", "analysis_specifications"],
        "output_types": ["analysis_results", "statistical_summaries", "predictions"],
        "priority": 4
    },
    "qa": {
        "name": "Quality Assurance Specialist",
        "role": "Result Validation Expert", 
        "description": "Validates analysis results and ensures quality of answers",
        "controller": QAController,
        "capabilities": [
            "Data quality assessment",
            "Result validation",
            "Statistical significance testing",
            "Business rule validation",
            "Quality scoring"
        ],
        "input_types": ["analysis_results", "validation_rules"],
        "output_types": ["quality_reports", "validation_scores"],
        "priority": 5
    },
    "insight": {
        "name": "Insight Generation Specialist",
        "role": "Answer Synthesis Expert",
        "description": "Generates human-readable insights and answers from analysis results",
        "controller": InsightController,
        "capabilities": [
            "Natural language generation",
            "Insight synthesis", 
            "Recommendation formulation",
            "Report generation",
            "Visualization suggestions"
        ],
        "input_types": ["validated_results", "question_context"],
        "output_types": ["insights", "recommendations", "summaries", "reports"],
        "priority": 6
    }
}

# Dynamic Workflow Configuration
DYNAMIC_WORKFLOWS = {
    "data_exploration": {
        "name": "Data Exploration Workflow",
        "description": "Understand what data is available and its structure",
        "required_agents": ["manager", "data_pulling", "insight"],
        "optional_agents": ["qa"],
        "estimated_time": 60
    },
    "statistical_summary": {
        "name": "Statistical Summary Workflow", 
        "description": "Generate statistical summaries and descriptive analytics",
        "required_agents": ["manager", "data_pulling", "preprocessing", "analysis", "insight"],
        "optional_agents": ["qa"],
        "estimated_time": 120
    },
    "trend_analysis": {
        "name": "Trend Analysis Workflow",
        "description": "Analyze changes and trends over time",
        "required_agents": ["manager", "data_pulling", "preprocessing", "analysis", "insight"],
        "optional_agents": ["qa"],
        "estimated_time": 180
    },
    "comparison": {
        "name": "Comparative Analysis Workflow",
        "description": "Compare different segments or groups",
        "required_agents": ["manager", "data_pulling", "preprocessing", "analysis", "insight"],
        "optional_agents": ["qa"],
        "estimated_time": 150
    },
    "correlation": {
        "name": "Correlation Analysis Workflow",
        "description": "Find relationships and correlations between variables",
        "required_agents": ["manager", "data_pulling", "preprocessing", "analysis", "qa", "insight"],
        "optional_agents": [],
        "estimated_time": 200
    },
    "prediction": {
        "name": "Predictive Modeling Workflow",
        "description": "Build predictive models and forecasts",
        "required_agents": ["manager", "data_pulling", "preprocessing", "analysis", "qa", "insight"],
        "optional_agents": [],
        "estimated_time": 300
    }
}

# Question Type Mappings
QUESTION_TYPE_WORKFLOWS = {
    "data_exploration": "data_exploration",
    "statistical_summary": "statistical_summary", 
    "trend_analysis": "trend_analysis",
    "comparison": "comparison",
    "correlation": "correlation",
    "prediction": "prediction",
    "custom_query": "statistical_summary"  # Default fallback
}

def get_agent_info(agent_id: Optional[str] = None) -> Dict[str, Any]:
    """Get information about agents in the system"""
    if agent_id:
        return AGENT_INFO.get(agent_id, {})
    return AGENT_INFO

def list_agents() -> List[str]:
    """List all available agent IDs"""
    return list(AGENT_INFO.keys())

def get_workflow_info(workflow_id: Optional[str] = None) -> Dict[str, Any]:
    """Get information about available workflows"""
    if workflow_id:
        return DYNAMIC_WORKFLOWS.get(workflow_id, {})
    return DYNAMIC_WORKFLOWS

def get_workflow_for_question_type(question_type: str) -> str:
    """Get recommended workflow for a question type"""
    return QUESTION_TYPE_WORKFLOWS.get(question_type, "statistical_summary")

def validate_workflow_agents(workflow_id: str, available_agents: List[str]) -> Dict[str, Any]:
    """Validate if required agents are available for a workflow"""
    workflow = DYNAMIC_WORKFLOWS.get(workflow_id)
    if not workflow:
        return {"valid": False, "error": f"Unknown workflow: {workflow_id}"}
    
    required_agents = workflow["required_agents"]
    missing_agents = [agent for agent in required_agents if agent not in available_agents]
    
    return {
        "valid": len(missing_agents) == 0,
        "missing_agents": missing_agents,
        "required_agents": required_agents,
        "optional_agents": workflow.get("optional_agents", [])
    }

def create_agent_controller(agent_id: str):
    """Create and return an agent controller instance"""
    agent_info = AGENT_INFO.get(agent_id)
    if not agent_info:
        raise ValueError(f"Unknown agent: {agent_id}")
    
    controller_class = agent_info["controller"]
    return controller_class()

def get_system_capabilities() -> Dict[str, List[str]]:
    """Get all system capabilities grouped by agent"""
    capabilities = {}
    for agent_id, info in AGENT_INFO.items():
        capabilities[agent_id] = info["capabilities"]
    return capabilities

def suggest_workflow(question_keywords: List[str]) -> str:
    """Suggest a workflow based on question keywords"""
    keyword_workflow_mapping = {
        # Data exploration keywords
        "data": "data_exploration",
        "available": "data_exploration", 
        "tables": "data_exploration",
        "columns": "data_exploration",
        "schema": "data_exploration",
        
        # Summary keywords
        "summary": "statistical_summary",
        "count": "statistical_summary",
        "total": "statistical_summary",
        "average": "statistical_summary", 
        "mean": "statistical_summary",
        
        # Trend keywords
        "trend": "trend_analysis",
        "time": "trend_analysis",
        "over": "trend_analysis",
        "change": "trend_analysis",
        "growth": "trend_analysis",
        
        # Comparison keywords
        "compare": "comparison",
        "vs": "comparison",
        "between": "comparison",
        "difference": "comparison",
        
        # Correlation keywords
        "correlate": "correlation",
        "relationship": "correlation",
        "factors": "correlation",
        "influence": "correlation",
        
        # Prediction keywords
        "predict": "prediction",
        "forecast": "prediction",
        "future": "prediction",
        "churn": "prediction"
    }
    
    # Score workflows based on keyword matches
    workflow_scores = {}
    for keyword in question_keywords:
        keyword_lower = keyword.lower()
        for word, workflow in keyword_workflow_mapping.items():
            if word in keyword_lower:
                workflow_scores[workflow] = workflow_scores.get(workflow, 0) + 1
    
    # Return highest scoring workflow or default
    if workflow_scores:
        return max(workflow_scores, key=workflow_scores.get)
    else:
        return "statistical_summary"  # Default fallback

# Export all public functions and constants
__all__ = [
    "AGENT_INFO", 
    "DYNAMIC_WORKFLOWS",
    "QUESTION_TYPE_WORKFLOWS",
    "get_agent_info", 
    "list_agents",
    "get_workflow_info",
    "get_workflow_for_question_type", 
    "validate_workflow_agents",
    "create_agent_controller",
    "get_system_capabilities",
    "suggest_workflow",
    "DynamicManagerController"
] 