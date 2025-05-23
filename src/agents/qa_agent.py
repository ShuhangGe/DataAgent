"""
QA Agent - Quality Assurance and Validation for Sekai Data Analysis Multi-Agent System
Built with CrewAI framework for robust quality assurance
"""

from crewai import Agent
from crewai.tools import BaseTool
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from pathlib import Path
from scipy import stats

from src.config.settings import settings
from src.models.data_models import (
    ValidationResult, ValidationLevel, DataQualityMetrics, 
    AgentTaskResult, AgentStatus
)

class DataQualityValidationTool(BaseTool):
    """Tool for validating data quality and integrity"""
    
    name: str = "data_quality_validation"
    description: str = "Perform comprehensive data quality validation and generate quality metrics"
    
    def _run(self, data: pd.DataFrame, quality_rules: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute data quality validation"""
        try:
            validations = []
            
            # Basic data integrity checks
            basic_validations = self._perform_basic_validations(data)
            validations.extend(basic_validations)
            
            # Statistical quality checks
            statistical_validations = self._perform_statistical_validations(data)
            validations.extend(statistical_validations)
            
            # Business rule validations
            if quality_rules:
                business_validations = self._perform_business_rule_validations(data, quality_rules)
                validations.extend(business_validations)
            
            # Calculate overall quality metrics
            quality_metrics = self._calculate_comprehensive_quality_metrics(data, validations)
            
            # Generate quality report
            quality_report = self._generate_quality_report(data, validations, quality_metrics)
            
            # Determine overall quality status
            critical_issues = [v for v in validations if v.level == ValidationLevel.CRITICAL]
            error_issues = [v for v in validations if v.level == ValidationLevel.ERROR]
            
            overall_status = "failed" if critical_issues else ("warning" if error_issues else "passed")
            
            return {
                "validations": validations,
                "quality_metrics": quality_metrics,
                "quality_report": quality_report,
                "overall_status": overall_status,
                "critical_issues_count": len(critical_issues),
                "error_issues_count": len(error_issues),
                "status": "success"
            }
            
        except Exception as e:
            return {"error": f"Data quality validation failed: {str(e)}"}
    
    def _perform_basic_validations(self, data: pd.DataFrame) -> List[ValidationResult]:
        """Perform basic data integrity validations"""
        validations = []
        
        # Check for empty dataset
        if data.empty:
            validations.append(ValidationResult(
                level=ValidationLevel.CRITICAL,
                message="Dataset is empty",
                component="qa_agent",
                row_count=0
            ))
            return validations
        
        # Check for required columns
        required_columns = ["user_id", "timestamp", "event_name"]
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            validations.append(ValidationResult(
                level=ValidationLevel.ERROR,
                message=f"Missing required columns: {missing_columns}",
                component="qa_agent",
                details={"missing_columns": missing_columns}
            ))
        
        # Check for completely null columns
        null_columns = data.columns[data.isnull().all()].tolist()
        if null_columns:
            validations.append(ValidationResult(
                level=ValidationLevel.WARNING,
                message=f"Columns with all null values: {null_columns}",
                component="qa_agent",
                details={"null_columns": null_columns}
            ))
        
        # Check for duplicate rows
        duplicate_count = data.duplicated().sum()
        if duplicate_count > 0:
            duplicate_ratio = duplicate_count / len(data)
            level = ValidationLevel.ERROR if duplicate_ratio > 0.1 else ValidationLevel.WARNING
            
            validations.append(ValidationResult(
                level=level,
                message=f"Found {duplicate_count} duplicate rows ({duplicate_ratio:.1%})",
                component="qa_agent",
                row_count=duplicate_count,
                details={"duplicate_ratio": duplicate_ratio}
            ))
        
        # Check data types consistency
        for column in data.columns:
            if column in ["user_id", "event_name"] and not data[column].dtype == 'object':
                validations.append(ValidationResult(
                    level=ValidationLevel.WARNING,
                    message=f"Column {column} should be string/object type, found {data[column].dtype}",
                    component="qa_agent",
                    column=column
                ))
        
        return validations
    
    def _perform_statistical_validations(self, data: pd.DataFrame) -> List[ValidationResult]:
        """Perform statistical quality validations"""
        validations = []
        
        # Check for outliers in numeric columns
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        
        for column in numeric_columns:
            if column in data.columns:
                # Z-score based outlier detection
                z_scores = np.abs(stats.zscore(data[column].dropna()))
                outlier_count = (z_scores > settings.analysis.outlier_threshold_std).sum()
                
                if outlier_count > 0:
                    outlier_ratio = outlier_count / len(data[column].dropna())
                    level = ValidationLevel.WARNING if outlier_ratio < 0.05 else ValidationLevel.ERROR
                    
                    validations.append(ValidationResult(
                        level=level,
                        message=f"Column {column}: {outlier_count} outliers detected ({outlier_ratio:.1%})",
                        component="qa_agent",
                        column=column,
                        details={
                            "outlier_count": int(outlier_count),
                            "outlier_ratio": outlier_ratio,
                            "threshold_std": settings.analysis.outlier_threshold_std
                        }
                    ))
        
        # Check data distribution patterns
        if "timestamp" in data.columns:
            timestamp_validations = self._validate_timestamp_distribution(data)
            validations.extend(timestamp_validations)
        
        return validations
    
    def _validate_timestamp_distribution(self, data: pd.DataFrame) -> List[ValidationResult]:
        """Validate timestamp distribution patterns"""
        validations = []
        
        try:
            data["timestamp"] = pd.to_datetime(data["timestamp"])
            
            # Check for data gaps
            data_by_date = data.groupby(data["timestamp"].dt.date).size()
            
            # Find dates with unusually low activity (less than 10% of median)
            median_activity = data_by_date.median()
            low_activity_dates = data_by_date[data_by_date < median_activity * 0.1]
            
            if len(low_activity_dates) > 0:
                validations.append(ValidationResult(
                    level=ValidationLevel.WARNING,
                    message=f"Found {len(low_activity_dates)} dates with unusually low activity",
                    component="qa_agent",
                    details={
                        "low_activity_dates": low_activity_dates.index.astype(str).tolist(),
                        "median_daily_activity": median_activity
                    }
                ))
            
            # Check for future timestamps
            future_timestamps = data[data["timestamp"] > datetime.now()]
            if len(future_timestamps) > 0:
                validations.append(ValidationResult(
                    level=ValidationLevel.ERROR,
                    message=f"Found {len(future_timestamps)} records with future timestamps",
                    component="qa_agent",
                    row_count=len(future_timestamps)
                ))
            
        except Exception as e:
            validations.append(ValidationResult(
                level=ValidationLevel.ERROR,
                message=f"Timestamp validation failed: {str(e)}",
                component="qa_agent",
                column="timestamp"
            ))
        
        return validations
    
    def _perform_business_rule_validations(self, data: pd.DataFrame, rules: List[Dict[str, Any]]) -> List[ValidationResult]:
        """Perform business rule validations"""
        validations = []
        
        for rule in rules:
            try:
                rule_name = rule.get("name", "Unknown Rule")
                rule_type = rule.get("type", "custom")
                
                if rule_type == "value_range":
                    validation = self._validate_value_range(data, rule)
                    if validation:
                        validations.append(validation)
                
                elif rule_type == "allowed_values":
                    validation = self._validate_allowed_values(data, rule)
                    if validation:
                        validations.append(validation)
                
                elif rule_type == "pattern_match":
                    validation = self._validate_pattern_match(data, rule)
                    if validation:
                        validations.append(validation)
                
                elif rule_type == "cross_field":
                    validation = self._validate_cross_field(data, rule)
                    if validation:
                        validations.append(validation)
                
            except Exception as e:
                validations.append(ValidationResult(
                    level=ValidationLevel.ERROR,
                    message=f"Business rule validation failed for {rule_name}: {str(e)}",
                    component="qa_agent",
                    details={"rule": rule}
                ))
        
        return validations
    
    def _validate_value_range(self, data: pd.DataFrame, rule: Dict[str, Any]) -> Optional[ValidationResult]:
        """Validate value range constraints"""
        column = rule.get("column")
        min_value = rule.get("min_value")
        max_value = rule.get("max_value")
        
        if column not in data.columns:
            return None
        
        violations = 0
        if min_value is not None:
            violations += (data[column] < min_value).sum()
        if max_value is not None:
            violations += (data[column] > max_value).sum()
        
        if violations > 0:
            return ValidationResult(
                level=ValidationLevel.WARNING,
                message=f"Column {column}: {violations} values outside range [{min_value}, {max_value}]",
                component="qa_agent",
                column=column,
                row_count=violations
            )
        
        return None
    
    def _validate_allowed_values(self, data: pd.DataFrame, rule: Dict[str, Any]) -> Optional[ValidationResult]:
        """Validate allowed values constraints"""
        column = rule.get("column")
        allowed_values = rule.get("allowed_values", [])
        
        if column not in data.columns or not allowed_values:
            return None
        
        invalid_values = ~data[column].isin(allowed_values)
        violation_count = invalid_values.sum()
        
        if violation_count > 0:
            unique_invalid = data[column][invalid_values].unique()
            return ValidationResult(
                level=ValidationLevel.WARNING,
                message=f"Column {column}: {violation_count} values not in allowed list",
                component="qa_agent",
                column=column,
                row_count=violation_count,
                details={
                    "invalid_values": unique_invalid.tolist(),
                    "allowed_values": allowed_values
                }
            )
        
        return None
    
    def _validate_pattern_match(self, data: pd.DataFrame, rule: Dict[str, Any]) -> Optional[ValidationResult]:
        """Validate pattern matching constraints"""
        column = rule.get("column")
        pattern = rule.get("pattern")
        
        if column not in data.columns or not pattern:
            return None
        
        import re
        pattern_violations = ~data[column].astype(str).str.match(pattern, na=False)
        violation_count = pattern_violations.sum()
        
        if violation_count > 0:
            return ValidationResult(
                level=ValidationLevel.WARNING,
                message=f"Column {column}: {violation_count} values don't match pattern {pattern}",
                component="qa_agent",
                column=column,
                row_count=violation_count
            )
        
        return None
    
    def _validate_cross_field(self, data: pd.DataFrame, rule: Dict[str, Any]) -> Optional[ValidationResult]:
        """Validate cross-field constraints"""
        # Simplified cross-field validation
        field1 = rule.get("field1")
        field2 = rule.get("field2")
        constraint = rule.get("constraint", "field1 > field2")
        
        if field1 not in data.columns or field2 not in data.columns:
            return None
        
        # Simple constraint evaluation (in production, use safer evaluation)
        try:
            violations = 0
            if constraint == "field1 > field2":
                violations = (data[field1] <= data[field2]).sum()
            elif constraint == "field1 >= field2":
                violations = (data[field1] < data[field2]).sum()
            
            if violations > 0:
                return ValidationResult(
                    level=ValidationLevel.WARNING,
                    message=f"Cross-field constraint violated: {field1} {constraint} {field2} ({violations} violations)",
                    component="qa_agent",
                    row_count=violations
                )
        except Exception:
            pass
        
        return None
    
    def _calculate_comprehensive_quality_metrics(self, data: pd.DataFrame, validations: List[ValidationResult]) -> DataQualityMetrics:
        """Calculate comprehensive data quality metrics"""
        # Completeness score
        total_cells = len(data) * len(data.columns)
        null_cells = data.isnull().sum().sum()
        completeness_score = 1 - (null_cells / total_cells) if total_cells > 0 else 0
        
        # Consistency score based on validations
        error_count = sum(1 for v in validations if v.level in [ValidationLevel.ERROR, ValidationLevel.CRITICAL])
        warning_count = sum(1 for v in validations if v.level == ValidationLevel.WARNING)
        total_checks = len(validations) if validations else 1
        
        consistency_score = max(0, 1 - (error_count * 0.3 + warning_count * 0.1) / total_checks)
        
        # Accuracy score (based on outliers and constraint violations)
        accuracy_score = 0.9  # Simplified - in production, calculate based on actual accuracy tests
        
        # Timeliness score (based on data freshness)
        timeliness_score = self._calculate_timeliness_score(data)
        
        return DataQualityMetrics(
            completeness_score=completeness_score,
            consistency_score=consistency_score,
            accuracy_score=accuracy_score,
            timeliness_score=timeliness_score
        )
    
    def _calculate_timeliness_score(self, data: pd.DataFrame) -> float:
        """Calculate timeliness score based on data freshness"""
        if "timestamp" not in data.columns:
            return 0.8  # Default score if no timestamp
        
        try:
            data["timestamp"] = pd.to_datetime(data["timestamp"])
            most_recent = data["timestamp"].max()
            days_old = (datetime.now() - most_recent).days
            
            # Score decreases as data gets older
            if days_old <= 1:
                return 1.0
            elif days_old <= 7:
                return 0.9
            elif days_old <= 30:
                return 0.7
            else:
                return 0.5
                
        except Exception:
            return 0.8
    
    def _generate_quality_report(self, data: pd.DataFrame, validations: List[ValidationResult], quality_metrics: DataQualityMetrics) -> Dict[str, Any]:
        """Generate comprehensive quality report"""
        return {
            "dataset_summary": {
                "total_rows": len(data),
                "total_columns": len(data.columns),
                "data_types": data.dtypes.to_dict(),
                "memory_usage_mb": data.memory_usage(deep=True).sum() / 1024 / 1024
            },
            "quality_scores": {
                "overall_score": quality_metrics.overall_score,
                "completeness": quality_metrics.completeness_score,
                "consistency": quality_metrics.consistency_score,
                "accuracy": quality_metrics.accuracy_score,
                "timeliness": quality_metrics.timeliness_score
            },
            "validation_summary": {
                "total_validations": len(validations),
                "critical_issues": sum(1 for v in validations if v.level == ValidationLevel.CRITICAL),
                "errors": sum(1 for v in validations if v.level == ValidationLevel.ERROR),
                "warnings": sum(1 for v in validations if v.level == ValidationLevel.WARNING),
                "info": sum(1 for v in validations if v.level == ValidationLevel.INFO)
            },
            "recommendations": self._generate_recommendations(validations, quality_metrics)
        }
    
    def _generate_recommendations(self, validations: List[ValidationResult], quality_metrics: DataQualityMetrics) -> List[str]:
        """Generate improvement recommendations"""
        recommendations = []
        
        if quality_metrics.completeness_score < 0.9:
            recommendations.append("Improve data completeness by addressing missing values")
        
        if quality_metrics.consistency_score < 0.8:
            recommendations.append("Address data consistency issues and validation errors")
        
        critical_issues = [v for v in validations if v.level == ValidationLevel.CRITICAL]
        if critical_issues:
            recommendations.append("Resolve critical data quality issues before proceeding with analysis")
        
        if quality_metrics.timeliness_score < 0.7:
            recommendations.append("Consider using more recent data for better insights")
        
        duplicate_issues = [v for v in validations if "duplicate" in v.message.lower()]
        if duplicate_issues:
            recommendations.append("Remove or investigate duplicate records")
        
        return recommendations

class ResultValidationTool(BaseTool):
    """Tool for validating analysis results and outputs"""
    
    name: str = "result_validation"
    description: str = "Validate analysis results for accuracy, completeness, and business logic"
    
    def _run(self, analysis_results: Dict[str, Any], analysis_type: str, input_data_info: Dict[str, Any]) -> Dict[str, Any]:
        """Validate analysis results"""
        try:
            validations = []
            
            # Check result completeness
            completeness_validations = self._validate_result_completeness(analysis_results, analysis_type)
            validations.extend(completeness_validations)
            
            # Check result accuracy
            accuracy_validations = self._validate_result_accuracy(analysis_results, analysis_type, input_data_info)
            validations.extend(accuracy_validations)
            
            # Check business logic
            business_validations = self._validate_business_logic(analysis_results, analysis_type)
            validations.extend(business_validations)
            
            # Generate validation summary
            validation_summary = self._generate_result_validation_summary(validations)
            
            return {
                "validations": validations,
                "validation_summary": validation_summary,
                "overall_result_quality": validation_summary["overall_quality"],
                "status": "success"
            }
            
        except Exception as e:
            return {"error": f"Result validation failed: {str(e)}"}
    
    def _validate_result_completeness(self, results: Dict[str, Any], analysis_type: str) -> List[ValidationResult]:
        """Validate completeness of analysis results"""
        validations = []
        
        # Check for required output components
        required_components = {
            "retention_analysis": ["retention_table", "cohort_summary"],
            "funnel_analysis": ["funnel_results", "user_journeys"],
            "segmentation_analysis": ["segments", "user_metrics"]
        }
        
        required = required_components.get(analysis_type, [])
        missing_components = [comp for comp in required if comp not in results]
        
        if missing_components:
            validations.append(ValidationResult(
                level=ValidationLevel.ERROR,
                message=f"Missing required result components: {missing_components}",
                component="qa_agent",
                details={"missing_components": missing_components}
            ))
        
        # Check for empty results
        for key, value in results.items():
            if value is None or (isinstance(value, (list, dict)) and len(value) == 0):
                validations.append(ValidationResult(
                    level=ValidationLevel.WARNING,
                    message=f"Result component '{key}' is empty",
                    component="qa_agent"
                ))
        
        return validations
    
    def _validate_result_accuracy(self, results: Dict[str, Any], analysis_type: str, input_info: Dict[str, Any]) -> List[ValidationResult]:
        """Validate accuracy of analysis results"""
        validations = []
        
        if analysis_type == "retention_analysis" and "retention_table" in results:
            retention_validations = self._validate_retention_accuracy(results["retention_table"], input_info)
            validations.extend(retention_validations)
        
        elif analysis_type == "funnel_analysis" and "funnel_results" in results:
            funnel_validations = self._validate_funnel_accuracy(results["funnel_results"], input_info)
            validations.extend(funnel_validations)
        
        return validations
    
    def _validate_retention_accuracy(self, retention_table: Any, input_info: Dict[str, Any]) -> List[ValidationResult]:
        """Validate retention analysis accuracy"""
        validations = []
        
        if isinstance(retention_table, pd.DataFrame):
            # Check retention rates are between 0 and 1
            numeric_cols = retention_table.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if col.startswith("day_"):
                    invalid_rates = (retention_table[col] < 0) | (retention_table[col] > 1)
                    if invalid_rates.any():
                        validations.append(ValidationResult(
                            level=ValidationLevel.ERROR,
                            message=f"Invalid retention rates in {col}: rates should be between 0 and 1",
                            component="qa_agent",
                            column=col
                        ))
            
            # Check retention rates are monotonically decreasing (generally expected)
            for idx, row in retention_table.iterrows():
                day_cols = [col for col in row.index if col.startswith("day_")]
                day_cols.sort(key=lambda x: int(x.split("_")[1]))
                
                for i in range(1, len(day_cols)):
                    if row[day_cols[i]] > row[day_cols[i-1]]:
                        validations.append(ValidationResult(
                            level=ValidationLevel.WARNING,
                            message=f"Retention rate increased from {day_cols[i-1]} to {day_cols[i]} in cohort {idx}",
                            component="qa_agent"
                        ))
        
        return validations
    
    def _validate_funnel_accuracy(self, funnel_results: Any, input_info: Dict[str, Any]) -> List[ValidationResult]:
        """Validate funnel analysis accuracy"""
        validations = []
        
        if isinstance(funnel_results, list):
            # Check conversion rates are monotonically decreasing
            prev_users = None
            for step_result in funnel_results:
                current_users = step_result.get("users_count", 0)
                
                if prev_users is not None and current_users > prev_users:
                    validations.append(ValidationResult(
                        level=ValidationLevel.WARNING,
                        message=f"Funnel step {step_result.get('step')} has more users than previous step",
                        component="qa_agent"
                    ))
                
                # Check conversion rates are valid
                conv_rate = step_result.get("conversion_rate", 0)
                if conv_rate < 0 or conv_rate > 1:
                    validations.append(ValidationResult(
                        level=ValidationLevel.ERROR,
                        message=f"Invalid conversion rate {conv_rate} for step {step_result.get('step')}",
                        component="qa_agent"
                    ))
                
                prev_users = current_users
        
        return validations
    
    def _validate_business_logic(self, results: Dict[str, Any], analysis_type: str) -> List[ValidationResult]:
        """Validate business logic constraints"""
        validations = []
        
        # Add business-specific validations based on Sekai domain knowledge
        if analysis_type == "retention_analysis":
            # Check if retention patterns make business sense
            if "cohort_summary" in results:
                summary = results["cohort_summary"]
                d1_retention = summary.get("avg_d1_retention", 0)
                d7_retention = summary.get("avg_d7_retention", 0)
                
                if d1_retention < 0.3:
                    validations.append(ValidationResult(
                        level=ValidationLevel.WARNING,
                        message=f"Day 1 retention ({d1_retention:.1%}) is unusually low for gaming products",
                        component="qa_agent"
                    ))
                
                if d7_retention > d1_retention:
                    validations.append(ValidationResult(
                        level=ValidationLevel.WARNING,
                        message="Day 7 retention higher than Day 1 retention - verify calculation",
                        component="qa_agent"
                    ))
        
        return validations
    
    def _generate_result_validation_summary(self, validations: List[ValidationResult]) -> Dict[str, Any]:
        """Generate result validation summary"""
        critical_count = sum(1 for v in validations if v.level == ValidationLevel.CRITICAL)
        error_count = sum(1 for v in validations if v.level == ValidationLevel.ERROR)
        warning_count = sum(1 for v in validations if v.level == ValidationLevel.WARNING)
        
        # Calculate overall quality score
        total_issues = critical_count + error_count + warning_count
        if total_issues == 0:
            overall_quality = 1.0
        else:
            # Penalize critical and error issues more heavily
            penalty = critical_count * 0.5 + error_count * 0.3 + warning_count * 0.1
            overall_quality = max(0, 1 - penalty / 10)  # Scale penalty
        
        return {
            "total_validations": len(validations),
            "critical_issues": critical_count,
            "errors": error_count,
            "warnings": warning_count,
            "overall_quality": overall_quality,
            "quality_status": "passed" if critical_count == 0 and error_count == 0 else "failed"
        }

def create_qa_agent() -> Agent:
    """Create and configure the QA Agent"""
    
    # Initialize tools
    data_quality_tool = DataQualityValidationTool()
    result_validation_tool = ResultValidationTool()
    
    # Create agent
    qa_agent = Agent(
        role="Quality Assurance Specialist",
        goal="""
        As a Quality Assurance Specialist, I am responsible for:
        1. Validating data quality and integrity throughout the analysis pipeline
        2. Performing comprehensive quality checks on input data and results
        3. Identifying potential issues, anomalies, and inconsistencies
        4. Ensuring analysis results meet quality standards and business logic
        5. Providing actionable recommendations for quality improvements
        """,
        backstory="""
        I am a meticulous Quality Assurance professional with extensive experience in data
        validation, testing methodologies, and quality control processes. I specialize in
        ensuring data reliability and analysis accuracy in complex analytical systems.
        My expertise includes statistical validation, business rule verification, and
        automated quality testing frameworks.
        """,
        tools=[data_quality_tool, result_validation_tool],
        verbose=settings.agent.verbose,
        allow_delegation=False,
        max_execution_time=settings.agent.max_execution_time,
        llm_config={
            "model": settings.openai.model,
            "temperature": settings.openai.temperature,
            "max_tokens": settings.openai.max_tokens
        }
    )
    
    return qa_agent

class QAController:
    """Controller for QA Agent operations"""
    
    def __init__(self):
        self.agent = create_qa_agent()
    
    def validate_data_quality(self, data: pd.DataFrame, quality_rules: List[Dict[str, Any]] = None) -> AgentTaskResult:
        """Validate data quality with comprehensive checks"""
        start_time = datetime.now()
        
        try:
            quality_tool = DataQualityValidationTool()
            result = quality_tool._run(data, quality_rules)
            
            if "error" in result:
                return AgentTaskResult(
                    agent_name="qa",
                    task_id="validate_data_quality",
                    status=AgentStatus.FAILED,
                    start_time=start_time,
                    end_time=datetime.now(),
                    error_message=result["error"]
                )
            
            # Determine task status based on validation results
            status = AgentStatus.FAILED if result["overall_status"] == "failed" else AgentStatus.COMPLETED
            
            return AgentTaskResult(
                agent_name="qa",
                task_id="validate_data_quality",
                status=status,
                start_time=start_time,
                end_time=datetime.now(),
                output_data=result,
                validations=result["validations"],
                quality_metrics=result["quality_metrics"],
                metadata={
                    "validation_type": "data_quality",
                    "overall_status": result["overall_status"],
                    "critical_issues_count": result["critical_issues_count"],
                    "error_issues_count": result["error_issues_count"]
                }
            )
            
        except Exception as e:
            return AgentTaskResult(
                agent_name="qa",
                task_id="validate_data_quality",
                status=AgentStatus.FAILED,
                start_time=start_time,
                end_time=datetime.now(),
                error_message=f"Data quality validation failed: {str(e)}"
            )
    
    def validate_results(self, analysis_results: Dict[str, Any], analysis_type: str, input_data_info: Dict[str, Any]) -> AgentTaskResult:
        """Validate analysis results for accuracy and completeness"""
        start_time = datetime.now()
        
        try:
            result_tool = ResultValidationTool()
            result = result_tool._run(analysis_results, analysis_type, input_data_info)
            
            if "error" in result:
                return AgentTaskResult(
                    agent_name="qa",
                    task_id="validate_results",
                    status=AgentStatus.FAILED,
                    start_time=start_time,
                    end_time=datetime.now(),
                    error_message=result["error"]
                )
            
            # Determine task status based on validation quality
            quality_status = result["validation_summary"]["quality_status"]
            status = AgentStatus.COMPLETED if quality_status == "passed" else AgentStatus.FAILED
            
            return AgentTaskResult(
                agent_name="qa",
                task_id="validate_results",
                status=status,
                start_time=start_time,
                end_time=datetime.now(),
                output_data=result,
                validations=result["validations"],
                metadata={
                    "validation_type": "result_validation",
                    "analysis_type": analysis_type,
                    "overall_result_quality": result["overall_result_quality"],
                    "quality_status": quality_status
                }
            )
            
        except Exception as e:
            return AgentTaskResult(
                agent_name="qa",
                task_id="validate_results",
                status=AgentStatus.FAILED,
                start_time=start_time,
                end_time=datetime.now(),
                error_message=f"Result validation failed: {str(e)}"
            )

# Export the controller
__all__ = ["QAController", "create_qa_agent"] 