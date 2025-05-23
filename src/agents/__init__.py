"""
Sekai Data Analysis Multi-Agent System - Agent Package
Six specialized agents for comprehensive data analysis workflow
"""

from .manager_agent import ManagerAgentController, create_manager_agent
from .data_pulling_agent import DataPullingController, create_data_pulling_agent
from .preprocessing_agent import PreprocessingController, create_preprocessing_agent
from .analysis_agent import AnalysisController, create_analysis_agent
from .qa_agent import QAController, create_qa_agent
from .insight_agent import InsightController, create_insight_agent

# Controllers for easy access
__all__ = [
    # Controllers
    "ManagerAgentController",
    "DataPullingController", 
    "PreprocessingController",
    "AnalysisController",
    "QAController",
    "InsightController",
    
    # Agent creation functions
    "create_manager_agent",
    "create_data_pulling_agent",
    "create_preprocessing_agent", 
    "create_analysis_agent",
    "create_qa_agent",
    "create_insight_agent"
]

# Agent information for documentation
AGENT_INFO = {
    "manager": {
        "name": "Manager Agent",
        "role": "Data Analysis Manager", 
        "description": "Orchestrates the entire analysis workflow and coordinates sub-agents",
        "controller": ManagerAgentController,
        "creator": create_manager_agent
    },
    "data_pulling": {
        "name": "Data Pulling Agent",
        "role": "Data Extraction Specialist",
        "description": "Extracts data from various sources with quality validation",
        "controller": DataPullingController,
        "creator": create_data_pulling_agent
    },
    "preprocessing": {
        "name": "Preprocessing Agent", 
        "role": "Data Preprocessing Specialist",
        "description": "Cleans, transforms, and prepares data for analysis",
        "controller": PreprocessingController,
        "creator": create_preprocessing_agent
    },
    "analysis": {
        "name": "Analysis Agent",
        "role": "Data Analysis Specialist", 
        "description": "Performs core data analysis algorithms and statistical methods",
        "controller": AnalysisController,
        "creator": create_analysis_agent
    },
    "qa": {
        "name": "QA Agent",
        "role": "Quality Assurance Specialist",
        "description": "Validates data quality and analysis results",
        "controller": QAController,
        "creator": create_qa_agent
    },
    "insight": {
        "name": "Insight Agent",
        "role": "Business Intelligence Specialist",
        "description": "Generates business insights and actionable recommendations",
        "controller": InsightController,
        "creator": create_insight_agent
    }
}

def get_agent_info(agent_name: str = None):
    """Get information about agents"""
    if agent_name:
        return AGENT_INFO.get(agent_name)
    return AGENT_INFO

def list_agents():
    """List all available agents"""
    return list(AGENT_INFO.keys())

# Workflow sequence for reference
STANDARD_WORKFLOW = [
    "manager",      # 1. Orchestrate and plan
    "data_pulling", # 2. Extract data
    "preprocessing",# 3. Clean and prepare
    "analysis",     # 4. Perform analysis
    "qa",          # 5. Validate results
    "insight"      # 6. Generate insights
] 