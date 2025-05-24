"""
Configuration settings for Sekai Data Analysis Multi-Agent System
Using latest Pydantic for robust configuration management
"""

import os
from typing import Optional, List, Dict, Any
from pydantic import Field, validator
from pydantic_settings import BaseSettings
from dotenv import load_dotenv
import pytz

load_dotenv()

class OpenAISettings(BaseSettings):
    """OpenAI API configuration for latest models"""
    api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    model: str = Field(default="gpt-4o", env="OPENAI_MODEL")  # Latest model
    temperature: float = Field(default=0.1, env="OPENAI_TEMPERATURE")
    max_tokens: int = Field(default=4000, env="OPENAI_MAX_TOKENS")
    
    @validator("api_key", pre=True)
    def validate_api_key(cls, v):
        if v is None or v == "your_openai_api_key_here":
            print("⚠️  Warning: No OpenAI API key configured. Set OPENAI_API_KEY environment variable.")
            return None
        return v
    
    class Config:
        env_prefix = "OPENAI_"

class DatabaseSettings(BaseSettings):
    """Database configuration"""
    url: str = Field(default="sqlite:///data/sekai_analysis.db", env="DATABASE_URL")
    max_connections: int = Field(default=20, env="DB_MAX_CONNECTIONS")
    
    class Config:
        env_prefix = "DATABASE_"

class SekaiProductSettings(BaseSettings):
    """Sekai product domain context"""
    domain: str = Field(default="gaming", env="SEKAI_PRODUCT_DOMAIN")
    timezone: str = Field(default="Asia/Tokyo", env="SEKAI_TIMEZONE")
    event_dictionary_path: str = Field(default="templates/sekai_events.yaml")
    kpi_definitions_path: str = Field(default="templates/sekai_kpis.yaml")
    
    @validator("timezone")
    def validate_timezone(cls, v):
        try:
            pytz.timezone(v)
            return v
        except Exception:
            raise ValueError(f"Invalid timezone: {v}")

class AnalysisSettings(BaseSettings):
    """Analysis execution configuration"""
    max_retries: int = Field(default=3, env="MAX_RETRIES")
    chunk_size: int = Field(default=10000, env="CHUNK_SIZE")
    sample_size: int = Field(default=100000, env="SAMPLE_SIZE")
    timeout_seconds: int = Field(default=300, env="TIMEOUT_SECONDS")
    
    # Data quality thresholds
    min_data_quality_score: float = Field(default=0.8)
    max_missing_ratio: float = Field(default=0.1)
    outlier_threshold_std: float = Field(default=3.0)
    
    class Config:
        env_prefix = "ANALYSIS_"

class AgentSettings(BaseSettings):
    """CrewAI agent configuration"""
    verbose: bool = Field(default=True, env="AGENT_VERBOSE")
    allow_delegation: bool = Field(default=False, env="AGENT_ALLOW_DELEGATION")
    max_execution_time: int = Field(default=300, env="AGENT_MAX_EXECUTION_TIME")
    memory: bool = Field(default=True, env="AGENT_MEMORY")
    
    class Config:
        env_prefix = "AGENT_"

class PathSettings(BaseSettings):
    """File system paths"""
    input_data_path: str = Field(default="data/input")
    output_data_path: str = Field(default="data/output")
    templates_path: str = Field(default="src/templates")
    logs_path: str = Field(default="logs")
    
    def __post_init__(self):
        # Ensure directories exist
        for path in [self.input_data_path, self.output_data_path, self.logs_path]:
            os.makedirs(path, exist_ok=True)

class LoggingSettings(BaseSettings):
    """Logging configuration"""
    level: str = Field(default="INFO", env="LOG_LEVEL")
    enable_monitoring: bool = Field(default=True, env="ENABLE_MONITORING")
    log_format: str = "{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}"
    
    class Config:
        env_prefix = "LOG_"

class Settings(BaseSettings):
    """Main settings class combining all configurations"""
    openai: OpenAISettings = OpenAISettings()
    database: DatabaseSettings = DatabaseSettings()
    sekai: SekaiProductSettings = SekaiProductSettings()
    analysis: AnalysisSettings = AnalysisSettings()
    agent: AgentSettings = AgentSettings()
    paths: PathSettings = PathSettings()
    logging: LoggingSettings = LoggingSettings()
    
    # Analysis templates registry
    available_templates: List[str] = [
        "recommendation_funnel",
        "time_pattern_analysis",
        "user_behavior_analysis"
    ]
    
    class Config:
        case_sensitive = False
        env_file = ".env"

# Global settings instance
settings = Settings()

# Template configurations for recommendation click analysis (simplified for MVP)
ANALYSIS_TEMPLATES = {
    "recommendation_funnel": {
        "name": "Recommendation Click Funnel Analysis",
        "description": "Analyze user journey from viewing to not clicking recommendations",
        "required_fields": ["device_id", "timestamp", "event"],
        "optional_fields": ["country", "timezone", "newDevice"],
        "output_format": ["funnel_analysis.csv", "user_behavior_report.md"]
    },
    "time_pattern_analysis": {
        "name": "Time Pattern Analysis",
        "description": "Analyze when users are most/least likely to click recommendations",
        "required_fields": ["device_id", "timestamp", "event"],
        "optional_fields": [],
        "output_format": ["time_patterns.csv", "hourly_behavior.png"]
    },
    "user_behavior_analysis": {
        "name": "User Behavior Analysis", 
        "description": "Analyze patterns in recommendation viewing vs clicking behavior",
        "required_fields": ["device_id", "timestamp", "event"],
        "optional_fields": ["newDevice"],
        "output_format": ["behavior_patterns.csv", "user_segments.csv"]
    }
} 