"""
Data models for Sekai Multi-Agent Data Analysis System
Using latest Pydantic v2 for robust data validation and serialization
"""

from datetime import datetime, date
from typing import Dict, List, Optional, Any, Union, Literal
from enum import Enum
from pydantic import BaseModel, Field, validator, model_validator
import pandas as pd
import json

class AnalysisType(str, Enum):
    """Available analysis types for recommendation click analysis"""
    RECOMMENDATION_FUNNEL = "recommendation_funnel"
    TIME_PATTERN = "time_pattern_analysis"
    USER_BEHAVIOR = "user_behavior_analysis"

class DataSourceType(str, Enum):
    """Supported data source types"""
    DATABASE = "database"
    CSV = "csv"
    PARQUET = "parquet"
    API = "api"
    DUCKDB = "duckdb"  # Latest high-performance analytics DB

class ValidationLevel(str, Enum):
    """Validation result severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class AgentStatus(str, Enum):
    """Agent execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"

class SekaiEvent(BaseModel):
    """Sekai product event definition"""
    event_name: str
    event_category: str
    description: str
    required_properties: List[str] = []
    optional_properties: List[str] = []
    business_impact: str
    tracking_frequency: str

class DataQualityMetrics(BaseModel):
    """Data quality assessment metrics"""
    completeness_score: float = Field(ge=0.0, le=1.0, description="Ratio of non-null values")
    consistency_score: float = Field(ge=0.0, le=1.0, description="Data format consistency")
    accuracy_score: float = Field(ge=0.0, le=1.0, description="Data accuracy estimation")
    timeliness_score: float = Field(ge=0.0, le=1.0, description="Data freshness score")
    overall_score: float = Field(ge=0.0, le=1.0, description="Weighted overall quality score")
    
    @model_validator(mode='before')
    def calculate_overall_score(cls, values):
        if isinstance(values, dict):
            scores = [
                values.get('completeness_score', 0),
                values.get('consistency_score', 0),
                values.get('accuracy_score', 0),
                values.get('timeliness_score', 0)
            ]
            values['overall_score'] = sum(scores) / len(scores)
        return values

class ValidationResult(BaseModel):
    """Validation result from QA Agent"""
    level: ValidationLevel
    message: str
    component: str = Field(description="Which component/agent generated this validation")
    column: Optional[str] = None
    row_count: Optional[int] = None
    timestamp: datetime = Field(default_factory=datetime.now)
    details: Dict[str, Any] = {}
    auto_fix_applied: bool = False
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class DataSchema(BaseModel):
    """Data schema definition"""
    column_name: str
    data_type: str
    is_required: bool = True
    description: Optional[str] = None
    constraints: Dict[str, Any] = {}
    business_meaning: Optional[str] = None

class AnalysisRequest(BaseModel):
    """Request for data analysis"""
    request_id: str = Field(default_factory=lambda: f"req_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}")
    analysis_type: AnalysisType
    user_query: str = Field(description="Natural language description of analysis needs")
    
    # Data source configuration
    data_source_type: DataSourceType
    data_source_config: Dict[str, Any]
    
    # Analysis parameters
    date_range: Dict[str, Union[str, date]] = {}
    filters: Dict[str, Any] = {}
    custom_parameters: Dict[str, Any] = {}
    
    # Output preferences
    output_format: List[str] = ["csv", "markdown"]
    include_visualizations: bool = True
    
    # Quality requirements
    min_data_quality_score: float = Field(default=0.8, ge=0.0, le=1.0)
    
    created_at: datetime = Field(default_factory=datetime.now)
    requested_by: Optional[str] = None

class AgentTaskResult(BaseModel):
    """Result from individual agent execution"""
    agent_name: str
    task_id: str
    status: AgentStatus
    
    # Execution details
    start_time: datetime
    end_time: Optional[datetime] = None
    execution_duration: Optional[float] = None  # seconds
    
    # Results
    output_data: Optional[Any] = None  # Can be DataFrame, dict, etc.
    output_files: List[str] = []
    validations: List[ValidationResult] = []
    quality_metrics: Optional[DataQualityMetrics] = None
    
    # Error handling
    error_message: Optional[str] = None
    retry_count: int = 0
    auto_fixes_applied: List[str] = []
    
    # Metadata
    metadata: Dict[str, Any] = {}
    
    @validator('execution_duration', always=True)
    def calculate_duration(cls, v, values):
        if 'start_time' in values and 'end_time' in values and values['end_time']:
            return (values['end_time'] - values['start_time']).total_seconds()
        return v

class AnalysisResult(BaseModel):
    """Complete analysis result"""
    request_id: str
    analysis_type: AnalysisType
    status: Literal["success", "partial_success", "failed"]
    
    # Results
    output_files: List[str] = []
    summary_markdown: Optional[str] = None
    visualization_files: List[str] = []
    
    # Quality and validation
    overall_quality_score: float = Field(ge=0.0, le=1.0)
    all_validations: List[ValidationResult] = []
    critical_issues: List[str] = []
    
    # Agent execution summary
    agent_results: List[AgentTaskResult] = []
    total_execution_time: float = 0.0  # seconds
    
    # Recommendations
    next_analysis_suggestions: List[str] = []
    data_improvement_suggestions: List[str] = []
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.now)
    data_source_info: Dict[str, Any] = {}
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            pd.DataFrame: lambda v: v.to_dict('records')
        }

class TemplateConfig(BaseModel):
    """Analysis template configuration"""
    template_id: str
    name: str
    description: str
    analysis_type: AnalysisType
    
    # Data requirements
    required_columns: List[str]
    optional_columns: List[str] = []
    expected_data_volume: Optional[int] = None
    
    # Analysis parameters
    default_parameters: Dict[str, Any] = {}
    parameter_constraints: Dict[str, Any] = {}
    
    # Output specification
    output_schema: List[DataSchema] = []
    visualization_types: List[str] = []
    
    # Quality rules
    quality_checks: List[Dict[str, Any]] = []
    business_rules: List[Dict[str, Any]] = []
    
    # Performance hints
    estimated_execution_time: int = 300  # seconds
    memory_requirements: str = "medium"  # low, medium, high
    
    created_at: datetime = Field(default_factory=datetime.now)
    version: str = "1.0"

class SekaiContext(BaseModel):
    """Sekai product domain context"""
    product_domain: str
    available_events: List[SekaiEvent] = []
    kpi_definitions: Dict[str, Any] = {}
    user_segments: List[str] = []
    business_metrics: Dict[str, Any] = {}
    data_dictionary: Dict[str, str] = {}
    
    # Temporal context
    business_timezone: str = "Asia/Tokyo"
    business_hours: Dict[str, str] = {"start": "09:00", "end": "18:00"}
    
    # Analysis context
    typical_analysis_patterns: List[str] = []
    common_data_issues: List[str] = []
    
    last_updated: datetime = Field(default_factory=datetime.now)

class SystemMetrics(BaseModel):
    """System performance and health metrics"""
    total_requests_processed: int = 0
    average_execution_time: float = 0.0
    success_rate: float = 1.0
    
    # Agent performance
    agent_performance: Dict[str, Dict[str, float]] = {}
    
    # Data quality trends
    average_quality_score: float = 0.0
    quality_trend: List[float] = []
    
    # Resource usage
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    
    # Error tracking
    common_errors: Dict[str, int] = {}
    resolved_issues: int = 0
    
    last_updated: datetime = Field(default_factory=datetime.now)

class RecommendationEvent(BaseModel):
    """Recommendation event data structure"""
    event: str = Field(description="Event type (e.g., 'recommendation_view', 'app_open')")
    timestamp: datetime = Field(description="Event timestamp in UTC")
    device_id: str = Field(description="Device identifier (used as user ID)")
    uuid: Optional[str] = Field(default=None, description="Event UUID")
    distinct_id: Optional[str] = Field(default=None, description="Distinct user ID")
    country: Optional[str] = Field(default=None, description="User country")
    timezone: Optional[str] = Field(default=None, description="User timezone")
    newDevice: Optional[bool] = Field(default=None, description="Whether device is new")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class RecommendationAnalysisResult(BaseModel):
    """Results from recommendation click analysis"""
    analysis_type: AnalysisType
    
    # Funnel Analysis Results
    exposure_users: int = Field(default=0, description="Users who saw recommendations")
    click_users: int = Field(default=0, description="Users who clicked recommendations") 
    no_click_users: int = Field(default=0, description="Users who didn't click")
    click_through_rate: float = Field(default=0.0, description="Overall click-through rate")
    
    # Time Pattern Results
    hourly_patterns: Dict[int, float] = Field(default_factory=dict, description="Click rates by hour")
    daily_patterns: Dict[str, float] = Field(default_factory=dict, description="Click rates by day")
    
    # User Behavior Results
    avg_exposures_per_user: float = Field(default=0.0, description="Average exposures per user")
    users_with_no_exposure: int = Field(default=0, description="Users without valid exposure")
    
    # Data Quality
    total_events: int = Field(default=0, description="Total events processed")
    valid_events: int = Field(default=0, description="Valid events after cleaning")
    data_quality_score: float = Field(default=0.0, description="Data quality score")
    
    created_at: datetime = Field(default_factory=datetime.now)

class RecommendationInsight(BaseModel):
    """Business insights for recommendation non-clicks"""
    insight_type: str = Field(description="Type of insight (time_based, user_based, etc.)")
    title: str = Field(description="Insight title")
    description: str = Field(description="Detailed insight description")
    confidence: str = Field(description="Confidence level: high, medium, low")
    impact: str = Field(description="Potential business impact")
    recommendation: str = Field(description="Actionable recommendation")
    
    # Supporting data
    supporting_metrics: Dict[str, Any] = Field(default_factory=dict)
    affected_users: Optional[int] = Field(default=None)
    
    created_at: datetime = Field(default_factory=datetime.now)

# Custom serialization for complex types
def serialize_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
    """Serialize DataFrame for JSON storage"""
    return {
        "data": df.to_dict('records'),
        "columns": df.columns.tolist(),
        "shape": df.shape,
        "dtypes": df.dtypes.to_dict()
    }

def deserialize_dataframe(data: Dict[str, Any]) -> pd.DataFrame:
    """Deserialize DataFrame from JSON storage"""
    return pd.DataFrame(data['data'])

class QuestionType(str, Enum):
    """Dynamic question types the system can answer"""
    DATA_EXPLORATION = "data_exploration"      # "What data do we have?"
    STATISTICAL_SUMMARY = "statistical_summary"  # "Show me summary of user activity"
    TREND_ANALYSIS = "trend_analysis"         # "How has engagement changed over time?"
    COMPARISON = "comparison"                 # "Compare user behavior between segments"
    CORRELATION = "correlation"               # "What factors correlate with retention?"
    PREDICTION = "prediction"                 # "Predict user churn"
    CUSTOM_QUERY = "custom_query"            # Direct SQL or complex analysis

class UserQuestion(BaseModel):
    """Dynamic user question model"""
    question_id: str = Field(default_factory=lambda: f"q_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}")
    question_text: str = Field(description="Natural language question from user")
    question_type: Optional[QuestionType] = Field(default=None, description="Detected question type")
    
    # Context extraction
    entities: List[str] = Field(default_factory=list, description="Extracted entities (tables, columns, metrics)")
    time_filters: Dict[str, Any] = Field(default_factory=dict, description="Time-based filters")
    conditions: List[str] = Field(default_factory=list, description="WHERE conditions")
    grouping: List[str] = Field(default_factory=list, description="GROUP BY fields")
    
    # Database context
    target_tables: List[str] = Field(default_factory=list, description="Relevant database tables")
    required_columns: List[str] = Field(default_factory=list, description="Required columns for analysis")
    
    # Analysis requirements
    analysis_methods: List[str] = Field(default_factory=list, description="Required analysis methods")
    output_format: str = Field(default="summary", description="Desired output format")
    
    created_at: datetime = Field(default_factory=datetime.now)
    
class DatabaseSchema(BaseModel):
    """Database schema information"""
    table_name: str
    columns: List[Dict[str, Any]]  # column name, type, description
    row_count: Optional[int] = None
    sample_data: Optional[Dict[str, Any]] = None
    relationships: List[str] = Field(default_factory=list)
    
class DatabaseContext(BaseModel):
    """Complete database context"""
    schemas: List[DatabaseSchema]
    available_metrics: List[str] = Field(default_factory=list)
    common_queries: List[str] = Field(default_factory=list)
    business_context: Dict[str, Any] = Field(default_factory=dict)

class DynamicAnalysisResult(BaseModel):
    """Results from dynamic question answering"""
    question_id: str
    question_text: str
    question_type: QuestionType
    
    # Analysis results
    data_summary: Dict[str, Any] = Field(default_factory=dict)
    key_findings: List[str] = Field(default_factory=list)
    charts_data: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Query information
    sql_queries: List[str] = Field(default_factory=list)
    data_sources: List[str] = Field(default_factory=list)
    
    # Insights
    insights: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    
    # Metadata
    execution_time: float = 0.0
    data_quality_score: float = 0.0
    confidence_score: float = 0.0
    
    created_at: datetime = Field(default_factory=datetime.now)

# Export all models
__all__ = [
    "AnalysisType", "DataSourceType", "ValidationLevel", "AgentStatus",
    "SekaiEvent", "DataQualityMetrics", "ValidationResult", "DataSchema",
    "AnalysisRequest", "AgentTaskResult", "AnalysisResult", "TemplateConfig",
    "SekaiContext", "SystemMetrics", "serialize_dataframe", "deserialize_dataframe",
    "RecommendationEvent", "RecommendationAnalysisResult", "RecommendationInsight",
    "QuestionType", "UserQuestion", "DatabaseSchema", "DatabaseContext", "DynamicAnalysisResult"
] 