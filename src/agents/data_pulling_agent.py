"""
Data Pulling Agent - Data Extraction for Sekai Data Analysis Multi-Agent System
Built with CrewAI framework for robust data extraction
"""

from crewai import Agent, Task
from crewai.tools import BaseTool
from typing import Dict, List, Any, Optional
import pandas as pd
import duckdb
import sqlalchemy
from pathlib import Path
import yaml
from datetime import datetime, timedelta

from src.config.settings import settings
from src.models.data_models import (
    DataSourceType, ValidationResult, ValidationLevel, 
    DataQualityMetrics, AgentTaskResult, AgentStatus
)

class DataExtractionTool(BaseTool):
    """Tool for extracting data from various sources"""
    
    name: str = "data_extraction"
    description: str = "Extract data from databases, CSV files, Parquet files, or APIs"
    
    def _run(self, source_config: Dict[str, Any]) -> Dict[str, Any]:
        """Extract data based on source configuration"""
        try:
            source_type = source_config.get("type")
            
            if source_type == DataSourceType.DATABASE:
                return self._extract_from_database(source_config)
            elif source_type == DataSourceType.CSV:
                return self._extract_from_csv(source_config)
            elif source_type == DataSourceType.DUCKDB:
                return self._extract_from_duckdb(source_config)
            elif source_type == DataSourceType.PARQUET:
                return self._extract_from_parquet(source_config)
            else:
                return {"error": f"Unsupported data source type: {source_type}"}
                
        except Exception as e:
            return {"error": f"Data extraction failed: {str(e)}"}
    
    def _extract_from_database(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Extract data from PostgreSQL database"""
        try:
            connection_string = config.get("connection_string", settings.database.url)
            table_name = config.get("table_name", "sekai_events")
            schema = config.get("schema", "public")
            
            # Build query with filters
            filters = config.get("filters", {})
            date_range = config.get("date_range", {})
            sample_size = config.get("sample_size", settings.analysis.sample_size)
            
            query = f"SELECT * FROM {schema}.{table_name}"
            where_conditions = []
            
            # Add date range filter
            if date_range.get("start_date") and date_range.get("end_date"):
                where_conditions.append(f"event_date BETWEEN '{date_range['start_date']}' AND '{date_range['end_date']}'")
            
            # Add other filters
            for key, value in filters.items():
                if value:
                    where_conditions.append(f"{key} = '{value}'")
            
            if where_conditions:
                query += " WHERE " + " AND ".join(where_conditions)
            
            # Add sampling
            if sample_size:
                query += f" ORDER BY RANDOM() LIMIT {sample_size}"
            
            # Execute query
            engine = sqlalchemy.create_engine(connection_string)
            df = pd.read_sql(query, engine)
            
            return {
                "data": df,
                "row_count": len(df),
                "columns": df.columns.tolist(),
                "source_info": {
                    "type": "database",
                    "table": table_name,
                    "query": query
                },
                "status": "success"
            }
            
        except Exception as e:
            return {"error": f"Database extraction failed: {str(e)}"}
    
    def _extract_from_csv(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Extract data from CSV file"""
        try:
            file_path = config.get("file_path")
            encoding = config.get("encoding", "utf-8")
            delimiter = config.get("delimiter", ",")
            
            if not Path(file_path).exists():
                return {"error": f"CSV file not found: {file_path}"}
            
            # Read CSV with error handling
            df = pd.read_csv(
                file_path, 
                encoding=encoding, 
                delimiter=delimiter,
                parse_dates=True,
                infer_datetime_format=True
            )
            
            # Apply filters
            df = self._apply_filters(df, config.get("filters", {}), config.get("date_range", {}))
            
            # Apply sampling
            sample_size = config.get("sample_size", settings.analysis.sample_size)
            if sample_size and len(df) > sample_size:
                df = df.sample(n=sample_size, random_state=42)
            
            return {
                "data": df,
                "row_count": len(df),
                "columns": df.columns.tolist(),
                "source_info": {
                    "type": "csv",
                    "file_path": file_path,
                    "encoding": encoding
                },
                "status": "success"
            }
            
        except Exception as e:
            return {"error": f"CSV extraction failed: {str(e)}"}
    
    def _extract_from_duckdb(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Extract data from DuckDB"""
        try:
            database_path = config.get("database_path", "data/sekai_analytics.duckdb")
            table_name = config.get("table_name", "events")
            
            # Connect to DuckDB
            conn = duckdb.connect(database_path)
            
            # Build query
            filters = config.get("filters", {})
            date_range = config.get("date_range", {})
            sample_size = config.get("sample_size", settings.analysis.sample_size)
            
            query = f"SELECT * FROM {table_name}"
            where_conditions = []
            
            # Add filters
            if date_range.get("start_date") and date_range.get("end_date"):
                where_conditions.append(f"event_date BETWEEN '{date_range['start_date']}' AND '{date_range['end_date']}'")
            
            for key, value in filters.items():
                if value:
                    where_conditions.append(f"{key} = '{value}'")
            
            if where_conditions:
                query += " WHERE " + " AND ".join(where_conditions)
            
            if sample_size:
                query += f" USING SAMPLE {min(sample_size, 100000)}"
            
            # Execute query
            df = conn.execute(query).df()
            conn.close()
            
            return {
                "data": df,
                "row_count": len(df),
                "columns": df.columns.tolist(),
                "source_info": {
                    "type": "duckdb",
                    "database_path": database_path,
                    "table": table_name
                },
                "status": "success"
            }
            
        except Exception as e:
            return {"error": f"DuckDB extraction failed: {str(e)}"}
    
    def _extract_from_parquet(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Extract data from Parquet file"""
        try:
            file_path = config.get("file_path")
            
            if not Path(file_path).exists():
                return {"error": f"Parquet file not found: {file_path}"}
            
            # Read Parquet file
            df = pd.read_parquet(file_path)
            
            # Apply filters
            df = self._apply_filters(df, config.get("filters", {}), config.get("date_range", {}))
            
            # Apply sampling
            sample_size = config.get("sample_size", settings.analysis.sample_size)
            if sample_size and len(df) > sample_size:
                df = df.sample(n=sample_size, random_state=42)
            
            return {
                "data": df,
                "row_count": len(df),
                "columns": df.columns.tolist(),
                "source_info": {
                    "type": "parquet",
                    "file_path": file_path
                },
                "status": "success"
            }
            
        except Exception as e:
            return {"error": f"Parquet extraction failed: {str(e)}"}
    
    def _apply_filters(self, df: pd.DataFrame, filters: Dict[str, Any], date_range: Dict[str, Any]) -> pd.DataFrame:
        """Apply filters to DataFrame"""
        # Apply date range filter
        if date_range.get("start_date") and date_range.get("end_date"):
            if "event_date" in df.columns:
                df = df[
                    (pd.to_datetime(df["event_date"]) >= pd.to_datetime(date_range["start_date"])) &
                    (pd.to_datetime(df["event_date"]) <= pd.to_datetime(date_range["end_date"]))
                ]
        
        # Apply other filters
        for key, value in filters.items():
            if value and key in df.columns:
                df = df[df[key] == value]
        
        return df

class SchemaValidationTool(BaseTool):
    """Tool for validating data schema and quality"""
    
    name: str = "schema_validation"
    description: str = "Validate data schema, types, and quality metrics"
    
    def _run(self, data: pd.DataFrame, expected_schema: Dict[str, Any] = None) -> Dict[str, Any]:
        """Validate data schema and calculate quality metrics"""
        try:
            validations = []
            
            # Basic data validation
            if data.empty:
                validations.append(ValidationResult(
                    level=ValidationLevel.CRITICAL,
                    message="No data extracted",
                    component="data_pulling_agent"
                ))
                return {"validations": validations, "quality_metrics": None}
            
            # Check required columns
            required_columns = ["user_id", "timestamp", "event_name"]
            missing_columns = [col for col in required_columns if col not in data.columns]
            
            if missing_columns:
                validations.append(ValidationResult(
                    level=ValidationLevel.ERROR,
                    message=f"Missing required columns: {missing_columns}",
                    component="data_pulling_agent"
                ))
            
            # Data quality metrics
            quality_metrics = self._calculate_quality_metrics(data)
            
            # Quality threshold checks
            if quality_metrics.completeness_score < settings.analysis.min_data_quality_score:
                validations.append(ValidationResult(
                    level=ValidationLevel.WARNING,
                    message=f"Data completeness below threshold: {quality_metrics.completeness_score:.2f}",
                    component="data_pulling_agent"
                ))
            
            return {
                "validations": validations,
                "quality_metrics": quality_metrics,
                "schema_info": {
                    "columns": data.columns.tolist(),
                    "dtypes": data.dtypes.to_dict(),
                    "shape": data.shape
                },
                "status": "success"
            }
            
        except Exception as e:
            return {"error": f"Schema validation failed: {str(e)}"}
    
    def _calculate_quality_metrics(self, df: pd.DataFrame) -> DataQualityMetrics:
        """Calculate data quality metrics"""
        # Completeness score
        completeness_score = 1 - (df.isnull().sum().sum() / (len(df) * len(df.columns)))
        
        # Consistency score (basic type consistency)
        consistency_score = 0.9  # Simplified for demo
        
        # Accuracy score (basic validation)
        accuracy_score = 0.85  # Simplified for demo
        
        # Timeliness score (based on most recent data)
        timeliness_score = 0.9  # Simplified for demo
        
        return DataQualityMetrics(
            completeness_score=completeness_score,
            consistency_score=consistency_score,
            accuracy_score=accuracy_score,
            timeliness_score=timeliness_score
        )

def create_data_pulling_agent() -> Agent:
    """Create and configure the Data Pulling Agent"""
    
    # Initialize tools
    extraction_tool = DataExtractionTool()
    validation_tool = SchemaValidationTool()
    
    # Create agent
    data_pulling_agent = Agent(
        role="Data Extraction Specialist",
        goal="""
        As a Data Extraction Specialist, I am responsible for:
        1. Extracting data from various sources (databases, files, APIs)
        2. Performing initial schema validation and volume checks
        3. Implementing retries and error handling for data access
        4. Ensuring data quality meets minimum thresholds
        5. Providing detailed extraction metadata and quality metrics
        """,
        backstory="""
        I am an expert in data extraction and integration with extensive experience in
        handling diverse data sources and formats. I specialize in ensuring data quality
        and reliability from the very first step of the analysis pipeline. I'm skilled
        in working with databases, file systems, and APIs, and I always validate data
        integrity before passing it to downstream processes.
        """,
        tools=[extraction_tool, validation_tool],
        verbose=settings.agent.verbose,
        allow_delegation=False,
        max_execution_time=settings.agent.max_execution_time,
        llm_config={
            "model": settings.openai.model,
            "temperature": settings.openai.temperature,
            "max_tokens": settings.openai.max_tokens
        }
    )
    
    return data_pulling_agent

class DataPullingController:
    """Controller for Data Pulling Agent operations"""
    
    def __init__(self):
        self.agent = create_data_pulling_agent()
    
    def extract_data(self, source_config: Dict[str, Any]) -> AgentTaskResult:
        """Extract data with full validation and quality checks"""
        start_time = datetime.now()
        
        try:
            # Extract data
            extraction_tool = DataExtractionTool()
            extraction_result = extraction_tool._run(source_config)
            
            if "error" in extraction_result:
                return AgentTaskResult(
                    agent_name="data_pulling",
                    task_id="extract_data",
                    status=AgentStatus.FAILED,
                    start_time=start_time,
                    end_time=datetime.now(),
                    error_message=extraction_result["error"]
                )
            
            # Validate schema and quality
            validation_tool = SchemaValidationTool()
            validation_result = validation_tool._run(extraction_result["data"])
            
            if "error" in validation_result:
                return AgentTaskResult(
                    agent_name="data_pulling",
                    task_id="extract_data",
                    status=AgentStatus.FAILED,
                    start_time=start_time,
                    end_time=datetime.now(),
                    error_message=validation_result["error"]
                )
            
            # Determine status based on validations
            critical_issues = [v for v in validation_result["validations"] 
                             if v.level == ValidationLevel.CRITICAL]
            
            status = AgentStatus.FAILED if critical_issues else AgentStatus.COMPLETED
            
            return AgentTaskResult(
                agent_name="data_pulling",
                task_id="extract_data",
                status=status,
                start_time=start_time,
                end_time=datetime.now(),
                output_data=extraction_result["data"],
                validations=validation_result["validations"],
                quality_metrics=validation_result["quality_metrics"],
                metadata={
                    "source_info": extraction_result["source_info"],
                    "schema_info": validation_result["schema_info"]
                }
            )
            
        except Exception as e:
            return AgentTaskResult(
                agent_name="data_pulling",
                task_id="extract_data",
                status=AgentStatus.FAILED,
                start_time=start_time,
                end_time=datetime.now(),
                error_message=f"Data pulling failed: {str(e)}"
            )

# Export the controller
__all__ = ["DataPullingController", "create_data_pulling_agent"] 