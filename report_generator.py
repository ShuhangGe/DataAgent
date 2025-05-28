#!/usr/bin/env python3
"""
Report Generator for Event-Based User Analysis System
Generates comprehensive markdown reports from analysis results
"""

import json
from datetime import datetime
from typing import Dict, Any, List
import pandas as pd

class AnalysisReportGenerator:
    """Generates comprehensive analysis reports in markdown format"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize report generator with configuration
        
        Args:
            config: Output configuration dictionary
        """
        self.config = config
        self.report_config = config.get('report', {})
    
    def generate_report(self, analysis_results: Dict[str, Any], output_file: str) -> bool:
        """Generate a comprehensive analysis report
        
        Args:
            analysis_results: Results from the LangGraph analysis
            output_file: Path to save the report
            
        Returns:
            True if report generated successfully, False otherwise
        """
        try:
            report_content = self._build_report_content(analysis_results)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report_content)
            
            return True
        except Exception as e:
            print(f"❌ Error generating report: {e}")
            return False
    
    def _build_report_content(self, results: Dict[str, Any]) -> str:
        """Build the complete report content"""
        sections = []
        
        # Header
        sections.append(self._generate_header())
        
        # Executive Summary
        if self.report_config.get('include_summary', True):
            sections.append(self._generate_executive_summary(results))
        
        # Key Statistics
        if self.report_config.get('include_statistics', True):
            sections.append(self._generate_statistics_section(results))
        
        # Event Pattern Analysis
        if self.report_config.get('include_visualizations', True):
            sections.append(self._generate_pattern_analysis(results))
        
        # AI Insights
        if self.report_config.get('include_ai_insights', True):
            sections.append(self._generate_ai_insights_section(results))
        
        # Recommendations
        if self.report_config.get('include_recommendations', True):
            sections.append(self._generate_recommendations_section(results))
        
        # Technical Details
        if self.report_config.get('include_technical_details', False):
            sections.append(self._generate_technical_details(results))
        
        # Footer
        sections.append(self._generate_footer(results))
        
        return '\n\n'.join(sections)
    
    def _generate_header(self) -> str:
        """Generate report header"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return f"""# Event-Based User Analysis Report

**Generated:** {timestamp}  
**Analysis System:** LangGraph-powered Event Relationship Analysis  
**Report Version:** 1.0

---"""
    
    def _generate_executive_summary(self, results: Dict[str, Any]) -> str:
        """Generate executive summary section"""
        raw_data = results.get('raw_data', pd.DataFrame())
        total_users = len(raw_data) if not raw_data.empty else 0
        
        success_status = "✅ Completed Successfully" if results.get('success') else "❌ Completed with Errors"
        
        return f"""## Executive Summary

**Analysis Status:** {success_status}  
**Total User Sessions Analyzed:** {total_users:,}  
**Analysis Completion:** {results.get('current_step', 'Unknown')}

This report provides a comprehensive analysis of user event patterns and behaviors based on processed event data. The analysis identifies key behavioral trends, event relationships, and provides actionable recommendations for improving user engagement."""
    
    def _generate_statistics_section(self, results: Dict[str, Any]) -> str:
        """Generate key statistics section"""
        event_patterns = results.get('event_patterns', {})
        
        # Extract key metrics
        sequences = event_patterns.get('sequences', {})
        temporal = event_patterns.get('temporal', {})
        relationships = event_patterns.get('relationships', {})
        
        total_sessions = sequences.get('total_sessions', 0)
        
        # Event frequency stats
        event_freq = relationships.get('event_frequency', {})
        top_events = sorted(event_freq.items(), key=lambda x: x[1], reverse=True)[:5] if event_freq else []
        
        # Session duration stats
        session_durations = temporal.get('session_durations', [])
        avg_duration = sum(session_durations) / len(session_durations) if session_durations else 0
        
        # Peak activity
        peak_hours = temporal.get('peak_activity_hours', [])
        peak_hour_text = f"Hour {peak_hours[0][0]} ({peak_hours[0][1]} events)" if peak_hours else "N/A"
        
        stats_content = f"""## Key Statistics

### Session Overview
- **Total Sessions:** {total_sessions:,}
- **Average Session Duration:** {avg_duration:.1f} minutes
- **Peak Activity Time:** {peak_hour_text}

### Top Events
"""
        
        if top_events:
            for i, (event, count) in enumerate(top_events, 1):
                stats_content += f"{i}. **{event}:** {count:,} occurrences\n"
        else:
            stats_content += "No event data available\n"
        
        # Conversion rates
        conversion_patterns = relationships.get('conversion_patterns', {})
        if conversion_patterns:
            stats_content += "\n### Conversion Rates\n"
            for key, value in conversion_patterns.items():
                if key.endswith('_rate'):
                    clean_name = key.replace('_rate', '').replace('_', ' ').title()
                    stats_content += f"- **{clean_name}:** {value:.1%}\n"
        
        return stats_content
    
    def _generate_pattern_analysis(self, results: Dict[str, Any]) -> str:
        """Generate event pattern analysis section"""
        event_patterns = results.get('event_patterns', {})
        sequences = event_patterns.get('sequences', {})
        temporal = event_patterns.get('temporal', {})
        relationships = event_patterns.get('relationships', {})
        
        content = "## Event Pattern Analysis\n\n"
        
        # Event transitions
        transitions = sequences.get('transitions', {})
        if transitions:
            content += "### Most Common Event Transitions\n\n"
            content += "| From Event | To Event | Frequency |\n"
            content += "|------------|----------|----------|\n"
            
            # Get top transitions
            all_transitions = []
            for from_event, to_events in transitions.items():
                for to_event, count in to_events.items():
                    all_transitions.append((from_event, to_event, count))
            
            top_transitions = sorted(all_transitions, key=lambda x: x[2], reverse=True)[:10]
            for from_event, to_event, count in top_transitions:
                content += f"| {from_event} | {to_event} | {count} |\n"
            content += "\n"
        
        # Temporal patterns
        hourly_dist = temporal.get('hourly_distribution', {})
        if hourly_dist:
            content += "### Activity by Hour\n\n"
            sorted_hours = sorted(hourly_dist.items(), key=lambda x: x[1], reverse=True)[:12]
            
            content += "| Hour | Events | Activity Level |\n"
            content += "|------|--------|---------------|\n"
            
            max_events = max(hourly_dist.values()) if hourly_dist else 1
            for hour, events in sorted_hours:
                activity_level = "🔥 High" if events > max_events * 0.7 else "📈 Medium" if events > max_events * 0.3 else "📉 Low"
                content += f"| {hour:02d}:00 | {events:,} | {activity_level} |\n"
            content += "\n"
        
        # Common sequences
        funnel_analysis = relationships.get('funnel_analysis', {})
        top_sequences = funnel_analysis.get('top_sequences', [])
        if top_sequences:
            content += "### Most Common Event Sequences\n\n"
            for i, (sequence, count) in enumerate(top_sequences[:5], 1):
                sequence_str = " → ".join(sequence)
                content += f"{i}. **{sequence_str}** ({count} occurrences)\n"
            content += "\n"
        
        return content
    
    def _generate_ai_insights_section(self, results: Dict[str, Any]) -> str:
        """Generate AI insights section"""
        behavioral_insights = results.get('behavioral_insights', {})
        ai_analysis = behavioral_insights.get('ai_analysis', '')
        
        if not ai_analysis:
            return "## AI-Generated Insights\n\n*No AI insights available.*"
        
        return f"""## AI-Generated Insights

{ai_analysis}"""
    
    def _generate_recommendations_section(self, results: Dict[str, Any]) -> str:
        """Generate recommendations section"""
        recommendations = results.get('recommendations', [])
        
        if not recommendations:
            return "## Recommendations\n\n*No recommendations generated.*"
        
        content = "## Recommendations\n\n"
        content += "Based on the analysis of user behavior patterns, here are actionable recommendations:\n\n"
        
        for i, rec in enumerate(recommendations, 1):
            content += f"### {i}. {rec}\n\n"
        
        return content
    
    def _generate_technical_details(self, results: Dict[str, Any]) -> str:
        """Generate technical details section"""
        content = "## Technical Details\n\n"
        
        # Analysis configuration
        content += "### Analysis Configuration\n"
        content += f"- **Current Step:** {results.get('current_step', 'N/A')}\n"
        content += f"- **Success Status:** {results.get('success', False)}\n"
        
        if results.get('error_message'):
            content += f"- **Error Message:** {results.get('error_message')}\n"
        
        content += "\n"
        
        # Workflow log
        messages = results.get('messages', [])
        if messages:
            content += "### Workflow Log\n\n"
            for message in messages[-10:]:  # Show last 10 messages
                content += f"- {message}\n"
            content += "\n"
        
        return content
    
    def _generate_footer(self, results: Dict[str, Any]) -> str:
        """Generate report footer"""
        return f"""---

## Report Information

**Generated by:** Event-Based User Analysis System  
**Powered by:** LangGraph Agent Framework  
**Analysis Engine:** OpenAI GPT Models  
**Report Format:** Markdown  

*This report was automatically generated from event data analysis. For questions about methodology or findings, please refer to the system documentation.*"""

def generate_analysis_report(results: Dict[str, Any], output_config: Dict[str, Any], output_file: str) -> bool:
    """Convenience function to generate analysis report
    
    Args:
        results: Analysis results from LangGraph workflow
        output_config: Output configuration
        output_file: Path to save the report
        
    Returns:
        True if successful, False otherwise
    """
    generator = AnalysisReportGenerator(output_config)
    return generator.generate_report(results, output_file)

if __name__ == "__main__":
    # Test report generation with sample data
    sample_results = {
        'success': True,
        'current_step': 'completed',
        'raw_data': pd.DataFrame({'device_id': range(100)}),  # Sample data
        'event_patterns': {
            'sequences': {'total_sessions': 100},
            'temporal': {'session_durations': [5.2, 3.1, 7.8]},
            'relationships': {'event_frequency': {'app_open': 150, 'screen_view': 120}}
        },
        'behavioral_insights': {'ai_analysis': 'Sample AI insights...'},
        'recommendations': ['Improve onboarding', 'Optimize peak hours']
    }
    
    sample_config = {
        'report': {
            'include_summary': True,
            'include_statistics': True,
            'include_visualizations': True,
            'include_ai_insights': True,
            'include_recommendations': True,
            'include_technical_details': True
        }
    }
    
    success = generate_analysis_report(sample_results, sample_config, 'test_report.md')
    print(f"Test report generation: {'✅ Success' if success else '❌ Failed'}") 