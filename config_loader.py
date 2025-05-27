#!/usr/bin/env python3
"""
Configuration Loader for Event-Based User Analysis System
Handles loading and validation of configuration parameters from YAML file
"""

import os
import yaml
import logging
from typing import Dict, Any, Optional
from pathlib import Path

class ConfigLoader:
    """Configuration loader with validation and defaults"""
    
    DEFAULT_CONFIG = {
        'database': {
            'path': 'DataProcess/event_analysis.db',
            'table_name': 'device_event_dictionaries'
        },
        'llm': {
            'model': 'gpt-4o-mini',
            'temperature': 0.3,
            'api_key': '',
            'max_tokens': None,
            'timeout': 60
        },
        'analysis': {
            'print_details': False,
            'enable_ai_insights': True,
            'enable_recommendations': True
        },
        'output': {
            'default_file': 'analysis_report.md',
            'format': 'report',
            'auto_save': True,
            'include_raw_data': False,
            'report': {
                'include_summary': True,
                'include_statistics': True,
                'include_visualizations': True,
                'include_ai_insights': True,
                'include_recommendations': True,
                'include_technical_details': False
            }
        },
        'logging': {
            'level': 'INFO',
            'log_to_file': False,
            'log_file': 'analysis.log'
        },
        'performance': {
            'max_sessions': 0,
            'batch_size': 1000,
            'enable_parallel': False
        }
    }
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize configuration loader
        
        Args:
            config_path: Path to the YAML configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        self._validate_config()
        self._apply_environment_overrides()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file with fallback to defaults"""
        if not os.path.exists(self.config_path):
            print(f"⚠️  Configuration file {self.config_path} not found. Using default configuration.")
            return self.DEFAULT_CONFIG.copy()
        
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                loaded_config = yaml.safe_load(f) or {}
            
            # Merge with defaults (defaults take precedence for missing keys)
            config = self._deep_merge(self.DEFAULT_CONFIG.copy(), loaded_config)
            print(f"✅ Configuration loaded from {self.config_path}")
            return config
            
        except yaml.YAMLError as e:
            print(f"❌ Error parsing YAML configuration file: {e}")
            print("Using default configuration.")
            return self.DEFAULT_CONFIG.copy()
        except Exception as e:
            print(f"❌ Error loading configuration file: {e}")
            print("Using default configuration.")
            return self.DEFAULT_CONFIG.copy()
    
    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries, with override taking precedence"""
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        return result
    
    def _validate_config(self):
        """Validate configuration parameters"""
        # Validate LLM temperature
        temp = self.config['llm']['temperature']
        if not (0.0 <= temp <= 2.0):
            print(f"⚠️  Invalid LLM temperature {temp}. Must be between 0.0 and 2.0. Using default 0.3.")
            self.config['llm']['temperature'] = 0.3
        
        # Validate LLM model
        valid_models = ['gpt-4', 'gpt-4o', 'gpt-4o-mini', 'gpt-3.5-turbo', 'gpt-3.5-turbo-16k']
        model = self.config['llm']['model']
        if model not in valid_models:
            print(f"⚠️  Unknown LLM model {model}. Using default gpt-4o-mini.")
            self.config['llm']['model'] = 'gpt-4o-mini'
        
        # Validate database path exists (if not default)
        db_path = self.config['database']['path']
        if db_path != self.DEFAULT_CONFIG['database']['path'] and not os.path.exists(db_path):
            print(f"⚠️  Database file {db_path} not found. Please ensure it exists before running analysis.")
        
        # Validate logging level
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        log_level = self.config['logging']['level'].upper()
        if log_level not in valid_levels:
            print(f"⚠️  Invalid logging level {log_level}. Using INFO.")
            self.config['logging']['level'] = 'INFO'
        else:
            self.config['logging']['level'] = log_level
    
    def _apply_environment_overrides(self):
        """Apply environment variable overrides"""
        # OpenAI API Key - environment variable takes precedence
        env_api_key = os.getenv('OPENAI_API_KEY')
        if env_api_key:
            self.config['llm']['api_key'] = env_api_key
        
        # Database path override
        env_db_path = os.getenv('DB_PATH')
        if env_db_path:
            self.config['database']['path'] = env_db_path
        
        # Output file override
        env_output_file = os.getenv('OUTPUT_FILE')
        if env_output_file:
            self.config['output']['default_file'] = env_output_file
        
        # Print details override
        env_print_details = os.getenv('PRINT_DETAILS')
        if env_print_details and env_print_details.lower() in ['true', '1', 'yes']:
            self.config['analysis']['print_details'] = True
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """Get configuration value using dot notation
        
        Args:
            key_path: Dot-separated path to the configuration key (e.g., 'llm.model')
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        keys = key_path.split('.')
        value = self.config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
    
    def get_database_config(self) -> Dict[str, Any]:
        """Get database configuration"""
        return self.config['database']
    
    def get_llm_config(self) -> Dict[str, Any]:
        """Get LLM configuration"""
        llm_config = self.config['llm'].copy()
        
        # Ensure API key is available
        if not llm_config['api_key']:
            llm_config['api_key'] = os.getenv('OPENAI_API_KEY')
        
        return llm_config
    
    def get_analysis_config(self) -> Dict[str, Any]:
        """Get analysis configuration"""
        return self.config['analysis']
    
    def get_output_config(self) -> Dict[str, Any]:
        """Get output configuration"""
        return self.config['output']
    
    def get_logging_config(self) -> Dict[str, Any]:
        """Get logging configuration"""
        return self.config['logging']
    
    def get_performance_config(self) -> Dict[str, Any]:
        """Get performance configuration"""
        return self.config['performance']
    
    def print_config_summary(self):
        """Print a summary of the current configuration"""
        print("\n📋 Configuration Summary:")
        print("=" * 50)
        
        # Database
        db_config = self.get_database_config()
        print(f"🗄️  Database: {db_config['path']}")
        print(f"   Table: {db_config['table_name']}")
        
        # LLM
        llm_config = self.get_llm_config()
        api_key_status = "✅ Configured" if llm_config.get('api_key') else "❌ Missing"
        print(f"🤖 LLM: {llm_config['model']} (temp: {llm_config['temperature']})")
        print(f"   API Key: {api_key_status}")
        
        # Analysis
        analysis_config = self.get_analysis_config()
        print(f"🔍 Analysis: Details={analysis_config['print_details']}, AI={analysis_config['enable_ai_insights']}")
        
        # Output
        output_config = self.get_output_config()
        print(f"💾 Output: {output_config['default_file']} (auto_save={output_config['auto_save']})")
        
        print("=" * 50)
    
    def save_config(self, output_path: Optional[str] = None):
        """Save current configuration to YAML file
        
        Args:
            output_path: Path to save configuration (defaults to original config_path)
        """
        save_path = output_path or self.config_path
        
        try:
            with open(save_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.config, f, default_flow_style=False, indent=2, sort_keys=False)
            print(f"✅ Configuration saved to {save_path}")
        except Exception as e:
            print(f"❌ Error saving configuration: {e}")

def load_config(config_path: str = "config.yaml") -> ConfigLoader:
    """Convenience function to load configuration
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        ConfigLoader instance
    """
    return ConfigLoader(config_path)

if __name__ == "__main__":
    # Test the configuration loader
    print("Testing Configuration Loader")
    print("=" * 40)
    
    config = load_config()
    config.print_config_summary()
    
    # Test getting specific values
    print(f"\nTest get() method:")
    print(f"LLM Model: {config.get('llm.model')}")
    print(f"Database Path: {config.get('database.path')}")
    print(f"Non-existent Key: {config.get('non.existent.key', 'DEFAULT_VALUE')}") 