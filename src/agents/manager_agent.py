"""
Dynamic Manager Agent - Question-Answering Orchestrator
Built with CrewAI framework for flexible multi-agent coordination
"""

from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool
from typing import Dict, List, Any, Optional
import yaml
import json
import re
from datetime import datetime, timedelta
from jinja2 import Template
import sqlalchemy as sa
from sqlalchemy import inspect

from src.config.settings import settings
from src.models.data_models import (
    UserQuestion, QuestionType, DatabaseSchema, DatabaseContext, 
    DynamicAnalysisResult, ValidationResult, ValidationLevel
)

class QuestionUnderstandingTool(BaseTool):
    """Tool for parsing and understanding natural language questions"""
    
    name: str = "question_understanding"
    description: str = "Parse natural language questions to extract intent, entities, and analysis requirements"
    
    def _run(self, question_text: str) -> Dict[str, Any]:
        """Parse and understand user question"""
        try:
            # Clean and normalize question
            question = question_text.strip().lower()
            
            # Detect question type
            question_type = self._detect_question_type(question)
            
            # Extract entities (tables, columns, metrics)
            entities = self._extract_entities(question)
            
            # Extract time filters
            time_filters = self._extract_time_filters(question)
            
            # Extract conditions and groupings
            conditions = self._extract_conditions(question)
            grouping = self._extract_grouping(question)
            
            # Determine analysis methods needed
            analysis_methods = self._determine_analysis_methods(question_type, question)
            
            # Determine output format
            output_format = self._determine_output_format(question)
            
            return {
                "question_type": question_type,
                "entities": entities,
                "time_filters": time_filters,
                "conditions": conditions,
                "grouping": grouping,
                "analysis_methods": analysis_methods,
                "output_format": output_format,
                "confidence": self._calculate_confidence(question_type, entities),
                "status": "success"
            }
            
        except Exception as e:
            return {"error": f"Question understanding failed: {str(e)}"}
    
    def _detect_question_type(self, question: str) -> QuestionType:
        """Detect the type of question being asked"""
        
        # Data exploration patterns
        if any(pattern in question for pattern in [
            "what data", "show me data", "available data", "tables", "columns"
        ]):
            return QuestionType.DATA_EXPLORATION
        
        # Statistical summary patterns
        elif any(pattern in question for pattern in [
            "summary", "average", "total", "count", "mean", "statistics", "overview"
        ]):
            return QuestionType.STATISTICAL_SUMMARY
        
        # Trend analysis patterns
        elif any(pattern in question for pattern in [
            "trend", "over time", "change", "growth", "decline", "monthly", "daily", "yearly"
        ]):
            return QuestionType.TREND_ANALYSIS
        
        # Comparison patterns
        elif any(pattern in question for pattern in [
            "compare", "vs", "versus", "difference", "between", "against"
        ]):
            return QuestionType.COMPARISON
        
        # Correlation patterns
        elif any(pattern in question for pattern in [
            "correlate", "relationship", "factors", "influence", "affect", "impact"
        ]):
            return QuestionType.CORRELATION
        
        # Prediction patterns
        elif any(pattern in question for pattern in [
            "predict", "forecast", "future", "churn", "likely to", "probability"
        ]):
            return QuestionType.PREDICTION
        
        # Default to custom query
        else:
            return QuestionType.CUSTOM_QUERY
    
    def _extract_entities(self, question: str) -> List[str]:
        """Extract relevant entities from question"""
        entities = []
        
        # Common database entities
        entity_patterns = {
            "users": ["user", "users", "customer", "customers"],
            "events": ["event", "events", "activity", "activities", "action", "actions"],
            "sessions": ["session", "sessions", "visit", "visits"],
            "revenue": ["revenue", "sales", "purchase", "purchases", "money"],
            "time": ["time", "date", "timestamp", "when", "day", "month", "year"],
            "retention": ["retention", "return", "comeback", "stay"],
            "engagement": ["engagement", "activity", "usage", "behavior"],
            "conversion": ["conversion", "convert", "funnel", "complete"],
            "churn": ["churn", "leave", "quit", "uninstall"]
        }
        
        for entity, patterns in entity_patterns.items():
            if any(pattern in question for pattern in patterns):
                entities.append(entity)
        
        return entities
    
    def _extract_time_filters(self, question: str) -> Dict[str, Any]:
        """Extract time-based filters from question"""
        time_filters = {}
        
        # Common time patterns
        if "last week" in question:
            time_filters["period"] = "last_week"
        elif "last month" in question:
            time_filters["period"] = "last_month"
        elif "last year" in question:
            time_filters["period"] = "last_year"
        elif "today" in question:
            time_filters["period"] = "today"
        elif "yesterday" in question:
            time_filters["period"] = "yesterday"
        
        # Date range patterns
        date_pattern = r'(\d{4}-\d{2}-\d{2})'
        dates = re.findall(date_pattern, question)
        if len(dates) >= 2:
            time_filters["start_date"] = dates[0]
            time_filters["end_date"] = dates[1]
        elif len(dates) == 1:
            time_filters["date"] = dates[0]
        
        return time_filters
    
    def _extract_conditions(self, question: str) -> List[str]:
        """Extract WHERE conditions from question"""
        conditions = []
        
        # Common condition patterns
        if "new users" in question or "new customers" in question:
            conditions.append("user_type = 'new'")
        elif "returning users" in question:
            conditions.append("user_type = 'returning'")
        
        if "premium" in question:
            conditions.append("user_tier = 'premium'")
        elif "free" in question:
            conditions.append("user_tier = 'free'")
        
        if "mobile" in question:
            conditions.append("platform = 'mobile'")
        elif "web" in question:
            conditions.append("platform = 'web'")
        
        return conditions
    
    def _extract_grouping(self, question: str) -> List[str]:
        """Extract GROUP BY requirements from question"""
        grouping = []
        
        if any(word in question for word in ["by day", "daily", "per day"]):
            grouping.append("date")
        elif any(word in question for word in ["by week", "weekly", "per week"]):
            grouping.append("week")
        elif any(word in question for word in ["by month", "monthly", "per month"]):
            grouping.append("month")
        
        if any(word in question for word in ["by country", "per country"]):
            grouping.append("country")
        elif any(word in question for word in ["by device", "per device"]):
            grouping.append("device_type")
        elif any(word in question for word in ["by platform", "per platform"]):
            grouping.append("platform")
        
        return grouping
    
    def _determine_analysis_methods(self, question_type: QuestionType, question: str) -> List[str]:
        """Determine what analysis methods are needed"""
        methods = []
        
        if question_type == QuestionType.DATA_EXPLORATION:
            methods = ["schema_inspection", "data_profiling"]
        elif question_type == QuestionType.STATISTICAL_SUMMARY:
            methods = ["descriptive_statistics", "aggregation"]
        elif question_type == QuestionType.TREND_ANALYSIS:
            methods = ["time_series_analysis", "trend_calculation"]
        elif question_type == QuestionType.COMPARISON:
            methods = ["comparative_analysis", "segmentation"]
        elif question_type == QuestionType.CORRELATION:
            methods = ["correlation_analysis", "statistical_modeling"]
        elif question_type == QuestionType.PREDICTION:
            methods = ["predictive_modeling", "machine_learning"]
        else:
            methods = ["custom_analysis"]
        
        return methods
    
    def _determine_output_format(self, question: str) -> str:
        """Determine desired output format"""
        if any(word in question for word in ["chart", "graph", "plot", "visualize"]):
            return "visualization"
        elif any(word in question for word in ["table", "list", "rows"]):
            return "table"
        elif any(word in question for word in ["summary", "brief", "overview"]):
            return "summary"
        else:
            return "detailed"
    
    def _calculate_confidence(self, question_type: QuestionType, entities: List[str]) -> float:
        """Calculate confidence in question understanding"""
        confidence = 0.5  # Base confidence
        
        # Higher confidence if we detected specific entities
        confidence += len(entities) * 0.1
        
        # Higher confidence for clear question types
        if question_type != QuestionType.CUSTOM_QUERY:
            confidence += 0.2
        
        return min(confidence, 1.0)

class DatabaseSchemaInspectionTool(BaseTool):
    """Tool for inspecting database schema and understanding available data"""
    
    name: str = "database_schema_inspection"
    description: str = "Inspect database schema to understand available tables, columns, and data"
    
    def _run(self, entities: List[str] = None) -> Dict[str, Any]:
        """Inspect database schema and return relevant information"""
        try:
            # Connect to database
            engine = sa.create_engine(settings.database.url)
            inspector = inspect(engine)
            
            # Get all table names
            table_names = inspector.get_table_names()
            
            # Build schema information
            schemas = []
            for table_name in table_names:
                columns = inspector.get_columns(table_name)
                
                # Get sample data
                sample_data = self._get_sample_data(engine, table_name)
                
                schema = DatabaseSchema(
                    table_name=table_name,
                    columns=[{
                        "name": col["name"],
                        "type": str(col["type"]),
                        "nullable": col["nullable"]
                    } for col in columns],
                    sample_data=sample_data
                )
                schemas.append(schema)
            
            # Filter relevant schemas based on entities
            if entities:
                relevant_schemas = self._filter_relevant_schemas(schemas, entities)
            else:
                relevant_schemas = schemas
            
            # Generate database context
            db_context = DatabaseContext(
                schemas=relevant_schemas,
                available_metrics=self._identify_metrics(relevant_schemas),
                common_queries=self._suggest_common_queries(relevant_schemas)
            )
            
            return {
                "database_context": db_context.dict(),
                "total_tables": len(table_names),
                "relevant_tables": len(relevant_schemas),
                "status": "success"
            }
            
        except Exception as e:
            return {"error": f"Database inspection failed: {str(e)}"}
    
    def _get_sample_data(self, engine, table_name: str, limit: int = 5) -> Dict[str, Any]:
        """Get sample data from table"""
        try:
            with engine.connect() as conn:
                result = conn.execute(sa.text(f"SELECT * FROM {table_name} LIMIT {limit}"))
                rows = result.fetchall()
                columns = result.keys()
                
                return {
                    "columns": list(columns),
                    "sample_rows": [dict(zip(columns, row)) for row in rows],
                    "row_count": len(rows)
                }
        except Exception:
            return {"error": "Could not fetch sample data"}
    
    def _filter_relevant_schemas(self, schemas: List[DatabaseSchema], entities: List[str]) -> List[DatabaseSchema]:
        """Filter schemas relevant to the user's question"""
        relevant = []
        
        for schema in schemas:
            # Check if table name matches entities
            table_relevant = any(entity in schema.table_name.lower() for entity in entities)
            
            # Check if columns match entities
            column_relevant = any(
                any(entity in col["name"].lower() for entity in entities)
                for col in schema.columns
            )
            
            if table_relevant or column_relevant:
                relevant.append(schema)
        
        # If no specific matches, return all schemas for data exploration
        return relevant if relevant else schemas
    
    def _identify_metrics(self, schemas: List[DatabaseSchema]) -> List[str]:
        """Identify available metrics from schema"""
        metrics = []
        
        for schema in schemas:
            for col in schema.columns:
                col_name = col["name"].lower()
                col_type = col["type"].lower()
                
                # Identify common metric patterns
                if "count" in col_name or "total" in col_name:
                    metrics.append(f"{schema.table_name}.{col['name']}")
                elif "revenue" in col_name or "amount" in col_name:
                    metrics.append(f"{schema.table_name}.{col['name']}")
                elif "rate" in col_name or "ratio" in col_name:
                    metrics.append(f"{schema.table_name}.{col['name']}")
                elif col_type in ["integer", "float", "decimal", "numeric"]:
                    metrics.append(f"{schema.table_name}.{col['name']}")
        
        return metrics
    
    def _suggest_common_queries(self, schemas: List[DatabaseSchema]) -> List[str]:
        """Suggest common queries based on schema"""
        queries = []
        
        for schema in schemas:
            table_name = schema.table_name
            
            # Basic count query
            queries.append(f"SELECT COUNT(*) FROM {table_name}")
            
            # Date-based queries if timestamp column exists
            timestamp_cols = [col["name"] for col in schema.columns 
                            if "timestamp" in col["name"].lower() or "date" in col["name"].lower()]
            
            if timestamp_cols:
                ts_col = timestamp_cols[0]
                queries.append(f"SELECT DATE({ts_col}) as date, COUNT(*) FROM {table_name} GROUP BY DATE({ts_col})")
        
        return queries

class DynamicTaskPlanningTool(BaseTool):
    """Tool for creating dynamic task plans based on user questions"""
    
    name: str = "dynamic_task_planning"
    description: str = "Create custom agent workflows based on user questions and database context"
    
    def _run(self, user_question: UserQuestion, db_context: DatabaseContext) -> Dict[str, Any]:
        """Create dynamic task plan for answering user question"""
        try:
            # Determine required agents and tasks
            task_plan = self._create_task_plan(user_question, db_context)
            
            # Estimate execution time
            estimated_time = len(task_plan) * 30  # 30 seconds per task
            
            # Add quality checks
            quality_checks = self._determine_quality_checks(user_question)
            
            return {
                "task_plan": task_plan,
                "estimated_time": estimated_time,
                "quality_checks": quality_checks,
                "plan_confidence": self._calculate_plan_confidence(task_plan, db_context),
                "status": "success"
            }
            
        except Exception as e:
            return {"error": f"Task planning failed: {str(e)}"}
    
    def _create_task_plan(self, question: UserQuestion, db_context: DatabaseContext) -> List[Dict[str, Any]]:
        """Create task plan based on question type and database context"""
        tasks = []
        
        # Always start with data pulling if we need database data
        if question.target_tables or question.question_type != QuestionType.DATA_EXPLORATION:
            tasks.append({
                "agent": "data_pulling",
                "task": "extract_relevant_data",
                "parameters": {
                    "target_tables": question.target_tables,
                    "columns": question.required_columns,
                    "time_filters": question.time_filters,
                    "conditions": question.conditions,
                    "sample_size": 50000
                }
            })
        
        # Add preprocessing if needed
        if question.question_type not in [QuestionType.DATA_EXPLORATION]:
            tasks.append({
                "agent": "preprocessing", 
                "task": "prepare_for_analysis",
                "parameters": {
                    "grouping": question.grouping,
                    "time_aggregation": "auto",
                    "missing_data_strategy": "auto"
                }
            })
        
        # Add analysis tasks based on question type
        if question.question_type == QuestionType.STATISTICAL_SUMMARY:
            tasks.append({
                "agent": "analysis",
                "task": "calculate_summary_statistics",
                "parameters": {
                    "metrics": ["count", "mean", "median", "std"],
                    "grouping": question.grouping
                }
            })
        
        elif question.question_type == QuestionType.TREND_ANALYSIS:
            tasks.append({
                "agent": "analysis",
                "task": "analyze_trends",
                "parameters": {
                    "time_column": "timestamp",
                    "metrics": question.entities,
                    "trend_method": "linear"
                }
            })
        
        elif question.question_type == QuestionType.COMPARISON:
            tasks.append({
                "agent": "analysis", 
                "task": "comparative_analysis",
                "parameters": {
                    "comparison_groups": question.grouping,
                    "metrics": question.entities
                }
            })
        
        elif question.question_type == QuestionType.CORRELATION:
            tasks.append({
                "agent": "analysis",
                "task": "correlation_analysis", 
                "parameters": {
                    "variables": question.entities,
                    "method": "pearson"
                }
            })
        
        elif question.question_type == QuestionType.PREDICTION:
            tasks.append({
                "agent": "analysis",
                "task": "predictive_modeling",
                "parameters": {
                    "target_variable": question.entities[0] if question.entities else "churn",
                    "features": question.entities[1:] if len(question.entities) > 1 else "auto"
                }
            })
        
        # Add QA validation
        tasks.append({
            "agent": "qa",
            "task": "validate_analysis_results",
            "parameters": {
                "question_context": question.question_text,
                "expected_output": question.output_format
            }
        })
        
        # Add insight generation
        tasks.append({
            "agent": "insight",
            "task": "generate_answer",
            "parameters": {
                "question": question.question_text,
                "question_type": question.question_type,
                "output_format": question.output_format
            }
        })
        
        return tasks
    
    def _determine_quality_checks(self, question: UserQuestion) -> List[str]:
        """Determine quality checks needed for this question"""
        checks = ["data_completeness", "result_validity"]
        
        if question.question_type == QuestionType.TREND_ANALYSIS:
            checks.append("temporal_consistency")
        elif question.question_type == QuestionType.PREDICTION:
            checks.append("model_accuracy")
        elif question.question_type == QuestionType.CORRELATION:
            checks.append("statistical_significance")
        
        return checks
    
    def _calculate_plan_confidence(self, task_plan: List[Dict], db_context: DatabaseContext) -> float:
        """Calculate confidence in the task plan"""
        confidence = 0.7  # Base confidence
        
        # Higher confidence if we have relevant data
        if db_context.schemas:
            confidence += 0.2
        
        # Lower confidence for complex plans
        if len(task_plan) > 5:
            confidence -= 0.1
        
        return max(0.0, min(1.0, confidence))

def create_manager_agent() -> Agent:
    """Create and configure the Dynamic Manager Agent"""
    
    # Initialize dynamic tools
    question_tool = QuestionUnderstandingTool()
    schema_tool = DatabaseSchemaInspectionTool()
    planning_tool = DynamicTaskPlanningTool()
    
    # Create agent
    manager_agent = Agent(
        role="Dynamic Question-Answering Manager",
        goal="""
        As a Dynamic Question-Answering Manager, I am responsible for:
        1. Understanding natural language questions from users
        2. Inspecting database schema to understand available data
        3. Creating custom workflows to answer specific questions
        4. Coordinating multi-agent execution based on question requirements
        5. Ensuring accurate and relevant answers to user questions
        """,
        backstory="""
        I am an intelligent data analysis orchestrator with expertise in natural language 
        understanding and dynamic workflow planning. I can interpret various types of questions 
        about data and coordinate specialized agents to provide accurate, insightful answers. 
        I adapt my approach based on the specific question and available data sources.
        """,
        tools=[question_tool, schema_tool, planning_tool],
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

class DynamicManagerController:
    """Controller for Dynamic Manager Agent operations"""
    
    def __init__(self):
        self.agent = create_manager_agent()
        self.question_cache = {}
        self.schema_cache = {}
    
    def process_user_question(self, question_text: str) -> Dict[str, Any]:
        """Process a natural language question and create execution plan"""
        try:
            # Step 1: Understand the question
            question_tool = QuestionUnderstandingTool()
            understanding_result = question_tool._run(question_text)
            
            if "error" in understanding_result:
                return {"status": "failed", "error": understanding_result["error"]}
            
            # Create UserQuestion object
            user_question = UserQuestion(
                question_text=question_text,
                question_type=understanding_result["question_type"],
                entities=understanding_result["entities"],
                time_filters=understanding_result["time_filters"],
                conditions=understanding_result["conditions"],
                grouping=understanding_result["grouping"],
                analysis_methods=understanding_result["analysis_methods"],
                output_format=understanding_result["output_format"]
            )
            
            # Step 2: Inspect database schema
            schema_tool = DatabaseSchemaInspectionTool()
            schema_result = schema_tool._run(user_question.entities)
            
            if "error" in schema_result:
                return {"status": "failed", "error": schema_result["error"]}
            
            db_context = DatabaseContext(**schema_result["database_context"])
            
            # Update question with relevant tables and columns
            user_question.target_tables = [schema.table_name for schema in db_context.schemas]
            user_question.required_columns = [
                col["name"] for schema in db_context.schemas for col in schema.columns
                if any(entity in col["name"].lower() for entity in user_question.entities)
            ]
            
            # Step 3: Create dynamic task plan
            planning_tool = DynamicTaskPlanningTool()
            plan_result = planning_tool._run(user_question, db_context)
            
            if "error" in plan_result:
                return {"status": "failed", "error": plan_result["error"]}
            
            return {
                "status": "success",
                "question": user_question.dict(),
                "database_context": db_context.dict(),
                "task_plan": plan_result["task_plan"],
                "estimated_time": plan_result["estimated_time"],
                "confidence": understanding_result["confidence"]
            }
            
        except Exception as e:
            return {"status": "failed", "error": f"Question processing failed: {str(e)}"}
    
    def suggest_questions(self, db_context: DatabaseContext) -> List[str]:
        """Suggest relevant questions based on available data"""
        suggestions = []
        
        for schema in db_context.schemas:
            table_name = schema.table_name
            
            # Basic exploration questions
            suggestions.append(f"What data is available in {table_name}?")
            suggestions.append(f"Show me a summary of {table_name}")
            
            # Time-based questions if timestamp columns exist
            timestamp_cols = [col["name"] for col in schema.columns 
                            if "timestamp" in col["name"].lower() or "date" in col["name"].lower()]
            if timestamp_cols:
                suggestions.append(f"How has {table_name} activity changed over time?")
                suggestions.append(f"What are the trends in {table_name}?")
            
            # User-related questions
            if "user" in table_name.lower():
                suggestions.append(f"How many users are active?")
                suggestions.append(f"What is user retention like?")
                suggestions.append(f"Compare user behavior by segment")
        
        return suggestions[:10]  # Limit to top 10 suggestions

# Export the controller
__all__ = ["DynamicManagerController", "create_manager_agent"] 