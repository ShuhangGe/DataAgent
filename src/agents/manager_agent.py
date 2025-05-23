"""
Manager Agent - Orchestrator for Sekai Data Analysis Multi-Agent System
Built with CrewAI framework for robust agent coordination
"""

from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool
from typing import Dict, List, Any, Optional
import yaml
import json
from datetime import datetime, timedelta
from jinja2 import Template

from src.config.settings import settings, ANALYSIS_TEMPLATES
from src.models.data_models import (
    AnalysisRequest, AnalysisResult, TemplateConfig, SekaiContext,
    AnalysisType, ValidationResult, ValidationLevel
)

class ContextRetrievalTool(BaseTool):
    """Tool for retrieving Sekai product context and templates"""
    
    name: str = "context_retrieval"
    description: str = "Retrieve Sekai product context, event definitions, and analysis templates"
    
    def _run(self, query: str) -> Dict[str, Any]:
        """Load context based on query type"""
        try:
            if "events" in query.lower():
                return self._load_event_dictionary()
            elif "template" in query.lower():
                return self._load_analysis_templates()
            elif "kpi" in query.lower():
                return self._load_kpi_definitions()
            else:
                return self._load_full_context()
        except Exception as e:
            return {"error": f"Context retrieval failed: {str(e)}"}
    
    def _load_event_dictionary(self) -> Dict[str, Any]:
        """Load Sekai event definitions"""
        try:
            with open(settings.sekai.event_dictionary_path, 'r', encoding='utf-8') as f:
                events = yaml.safe_load(f)
            return {"events": events, "status": "success"}
        except FileNotFoundError:
            # Provide default Sekai events if file not found
            return {
                "events": {
                    "user_login": {"category": "auth", "impact": "high", "frequency": "daily"},
                    "character_gacha": {"category": "monetization", "impact": "critical", "frequency": "high"},
                    "story_complete": {"category": "engagement", "impact": "medium", "frequency": "medium"},
                    "battle_start": {"category": "gameplay", "impact": "high", "frequency": "high"},
                    "purchase_complete": {"category": "monetization", "impact": "critical", "frequency": "low"}
                },
                "status": "default_loaded"
            }
    
    def _load_analysis_templates(self) -> Dict[str, Any]:
        """Load available analysis templates"""
        return {
            "templates": ANALYSIS_TEMPLATES,
            "available_types": list(ANALYSIS_TEMPLATES.keys()),
            "status": "success"
        }
    
    def _load_kpi_definitions(self) -> Dict[str, Any]:
        """Load KPI definitions"""
        try:
            with open(settings.sekai.kpi_definitions_path, 'r', encoding='utf-8') as f:
                kpis = yaml.safe_load(f)
            return {"kpis": kpis, "status": "success"}
        except FileNotFoundError:
            return {
                "kpis": {
                    "DAU": "Daily Active Users",
                    "ARPU": "Average Revenue Per User", 
                    "Retention_D1": "Day 1 Retention Rate",
                    "Retention_D7": "Day 7 Retention Rate",
                    "LTV": "Lifetime Value",
                    "ROAS": "Return on Ad Spend"
                },
                "status": "default_loaded"
            }
    
    def _load_full_context(self) -> Dict[str, Any]:
        """Load complete Sekai context"""
        events = self._load_event_dictionary()
        templates = self._load_analysis_templates()
        kpis = self._load_kpi_definitions()
        
        return {
            "sekai_context": {
                "domain": settings.sekai.domain,
                "timezone": settings.sekai.timezone,
                "events": events.get("events", {}),
                "templates": templates.get("templates", {}),
                "kpis": kpis.get("kpis", {})
            },
            "status": "success"
        }

class TemplateParameterizationTool(BaseTool):
    """Tool for template selection and parameter rendering"""
    
    name: str = "template_parameterization"
    description: str = "Select appropriate analysis template and render parameters based on user requirements"
    
    def _run(self, user_query: str, analysis_type: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Parameterize analysis template"""
        try:
            if analysis_type not in ANALYSIS_TEMPLATES:
                return {"error": f"Unknown analysis type: {analysis_type}"}
            
            template_config = ANALYSIS_TEMPLATES[analysis_type]
            
            # Render template with parameters
            rendered_config = self._render_template(template_config, parameters)
            
            # Validate required parameters
            validation_result = self._validate_parameters(rendered_config, parameters)
            
            return {
                "template_config": rendered_config,
                "validation": validation_result,
                "status": "success"
            }
            
        except Exception as e:
            return {"error": f"Template parameterization failed: {str(e)}"}
    
    def _render_template(self, template_config: Dict[str, Any], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Render template with Jinja2"""
        rendered = template_config.copy()
        
        # Add default date range if not specified
        if "date_range" not in parameters:
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=30)
            parameters["date_range"] = {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat()
            }
        
        # Render string templates
        for key, value in rendered.items():
            if isinstance(value, str) and "{{" in value:
                template = Template(value)
                rendered[key] = template.render(**parameters)
        
        return rendered
    
    def _validate_parameters(self, config: Dict[str, Any], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Validate template parameters"""
        missing_required = []
        for field in config.get("required_fields", []):
            if field not in parameters and field not in ["user_id", "timestamp", "event_name"]:
                missing_required.append(field)
        
        warnings = []
        if "date_range" in parameters:
            date_range = parameters["date_range"]
            if isinstance(date_range, dict):
                start = datetime.fromisoformat(date_range["start_date"])
                end = datetime.fromisoformat(date_range["end_date"])
                days_diff = (end - start).days
                
                if days_diff > 365:
                    warnings.append("Large date range may impact performance")
                elif days_diff < 1:
                    warnings.append("Date range too small for meaningful analysis")
        
        return {
            "missing_required": missing_required,
            "warnings": warnings,
            "is_valid": len(missing_required) == 0
        }

class TaskOrchestrationTool(BaseTool):
    """Tool for coordinating sub-agent tasks"""
    
    name: str = "task_orchestration"
    description: str = "Generate and coordinate task sequences for sub-agents"
    
    def _run(self, analysis_config: Dict[str, Any]) -> Dict[str, Any]:
        """Generate task sequence for analysis"""
        try:
            analysis_type = analysis_config.get("analysis_type")
            
            # Define standard task sequence
            task_sequence = [
                {
                    "agent": "data_pulling",
                    "task": "extract_data",
                    "parameters": {
                        "data_source": analysis_config.get("data_source", {}),
                        "filters": analysis_config.get("filters", {}),
                        "date_range": analysis_config.get("date_range", {}),
                        "sample_size": settings.analysis.sample_size
                    }
                },
                {
                    "agent": "preprocessing",
                    "task": "clean_and_prepare",
                    "parameters": {
                        "timezone": settings.sekai.timezone,
                        "derive_features": True,
                        "quality_threshold": settings.analysis.min_data_quality_score
                    }
                },
                {
                    "agent": "analysis",
                    "task": f"execute_{analysis_type}",
                    "parameters": analysis_config.get("analysis_parameters", {})
                },
                {
                    "agent": "qa",
                    "task": "validate_results",
                    "parameters": {
                        "quality_rules": analysis_config.get("quality_rules", []),
                        "business_rules": analysis_config.get("business_rules", [])
                    }
                },
                {
                    "agent": "insight",
                    "task": "generate_summary",
                    "parameters": {
                        "include_recommendations": True,
                        "output_format": "markdown"
                    }
                }
            ]
            
            return {
                "task_sequence": task_sequence,
                "estimated_time": len(task_sequence) * 60,  # 1 minute per task estimate
                "status": "success"
            }
            
        except Exception as e:
            return {"error": f"Task orchestration failed: {str(e)}"}

def create_manager_agent() -> Agent:
    """Create and configure the Manager Agent using CrewAI"""
    
    # Initialize tools
    context_tool = ContextRetrievalTool()
    template_tool = TemplateParameterizationTool()
    orchestration_tool = TaskOrchestrationTool()
    
    # Create agent with latest OpenAI model
    manager_agent = Agent(
        role="Data Analysis Manager",
        goal="""
        As the Data Analysis Manager for the Sekai system, I am responsible for:
        1. Understanding user data analysis requirements
        2. Loading Sekai product context and templates
        3. Selecting appropriate analysis templates and configuring parameters
        4. Coordinating execution sequence across sub-agents
        5. Ensuring final result quality and completeness
        """,
        backstory="""
        I am a professional data analysis manager with deep understanding of the Sekai product
        and extensive experience in data analysis. I can accurately understand user requirements,
        select the most appropriate analysis methods, and ensure efficient execution of the entire
        analysis workflow. I specialize in handling complex scenarios in game data analysis,
        such as user retention, conversion funnels, user segmentation, and more.
        """,
        tools=[context_tool, template_tool, orchestration_tool],
        verbose=settings.agent.verbose,
        allow_delegation=False,
        max_execution_time=settings.agent.max_execution_time,
        llm_config={
            "model": settings.openai.model,
            "temperature": settings.openai.temperature,
            "max_tokens": settings.openai.max_tokens
        }
    )
    
    return manager_agent

class ManagerAgentController:
    """Controller class for Manager Agent operations"""
    
    def __init__(self):
        self.agent = create_manager_agent()
        self.context_cache = {}
        self.session_state = {}
    
    def process_analysis_request(self, request: AnalysisRequest) -> Dict[str, Any]:
        """Process a complete analysis request"""
        try:
            # Step 1: Load context
            context_result = self._load_context()
            if "error" in context_result:
                return {"status": "failed", "error": context_result["error"]}
            
            # Step 2: Clarify requirements and parameterize template
            template_result = self._parameterize_template(request)
            if "error" in template_result:
                return {"status": "failed", "error": template_result["error"]}
            
            # Step 3: Generate task sequence
            orchestration_result = self._orchestrate_tasks(request, template_result)
            if "error" in orchestration_result:
                return {"status": "failed", "error": orchestration_result["error"]}
            
            return {
                "status": "success",
                "context": context_result,
                "template_config": template_result,
                "task_sequence": orchestration_result["task_sequence"],
                "estimated_execution_time": orchestration_result["estimated_time"]
            }
            
        except Exception as e:
            return {"status": "failed", "error": f"Manager processing failed: {str(e)}"}
    
    def _load_context(self) -> Dict[str, Any]:
        """Load Sekai product context"""
        if "sekai_context" not in self.context_cache:
            context_tool = ContextRetrievalTool()
            result = context_tool._run("full_context")
            self.context_cache["sekai_context"] = result
        
        return self.context_cache["sekai_context"]
    
    def _parameterize_template(self, request: AnalysisRequest) -> Dict[str, Any]:
        """Parameterize analysis template based on request"""
        template_tool = TemplateParameterizationTool()
        
        parameters = {
            "user_query": request.user_query,
            "date_range": request.date_range,
            "filters": request.filters,
            **request.custom_parameters
        }
        
        return template_tool._run(
            request.user_query,
            request.analysis_type.value,
            parameters
        )
    
    def _orchestrate_tasks(self, request: AnalysisRequest, template_result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate orchestrated task sequence"""
        orchestration_tool = TaskOrchestrationTool()
        
        analysis_config = {
            "analysis_type": request.analysis_type.value,
            "data_source": request.data_source_config,
            "filters": request.filters,
            "date_range": request.date_range,
            "analysis_parameters": request.custom_parameters,
            "quality_rules": template_result.get("template_config", {}).get("quality_rules", []),
            "business_rules": template_result.get("template_config", {}).get("business_rules", [])
        }
        
        return orchestration_tool._run(analysis_config)
    
    def handle_agent_failure(self, failed_agent: str, error_details: Dict[str, Any]) -> Dict[str, Any]:
        """Handle sub-agent failures with recovery strategies"""
        recovery_strategies = {
            "data_pulling": ["retry_with_smaller_chunk", "switch_to_sample_data", "adjust_query_parameters"],
            "preprocessing": ["skip_problematic_columns", "use_alternative_cleaning", "reduce_quality_threshold"],
            "analysis": ["use_simpler_algorithm", "reduce_complexity", "fall_back_to_basic_stats"],
            "qa": ["lower_validation_threshold", "skip_non_critical_checks", "manual_review_mode"],
            "insight": ["use_template_summary", "basic_statistical_summary", "raw_data_summary"]
        }
        
        strategies = recovery_strategies.get(failed_agent, ["manual_intervention_required"])
        
        return {
            "failed_agent": failed_agent,
            "error": error_details,
            "recovery_strategies": strategies,
            "recommended_action": strategies[0] if strategies else "manual_intervention_required"
        }

# Export the controller
__all__ = ["ManagerAgentController", "create_manager_agent"] 