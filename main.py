#!/usr/bin/env python3
"""
Sekai Data Analysis Multi-Agent System
Main Application Entry Point

Built with CrewAI framework and latest AI technologies
"""

import asyncio
import sys
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime, date
import json

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt, Confirm
from rich.markdown import Markdown
import yaml

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from src.config.settings import settings, ANALYSIS_TEMPLATES
from src.models.data_models import (
    AnalysisRequest, AnalysisType, DataSourceType, 
    ValidationLevel, AgentStatus
)
from src.agents import ManagerAgentController, get_agent_info, list_agents, AGENT_INFO

# Initialize Rich console and Typer app
console = Console()
app = typer.Typer(
    name="sekai-data-agent",
    help="🎮 Sekai Data Analysis Multi-Agent System",
    add_completion=False,
    rich_markup_mode="rich"
)

class SekaiDataAnalysisApp:
    """Main application class for Sekai Data Analysis System"""
    
    def __init__(self):
        self.manager = ManagerAgentController()
        self.console = Console()
        
    def display_welcome(self):
        """Display welcome message and system info"""
        welcome_panel = Panel.fit(
            """[bold blue]🎮 Recommendation Click Analysis System[/bold blue]

[green]✨ Simplified for MVP: Understanding User Behavior[/green]
[yellow]🤖 Multi-Agent Analysis for Recommendation Insights[/yellow]

[white]📊 Focus: Why users don't click on recommended content[/white]
[white]📁 Data: Users who viewed but didn't click recommendations[/white]
[white]🔧 MVP: Using only timestamp and event columns[/white]

Available Analysis Types:
• Recommendation Funnel Analysis
• Time Pattern Analysis  
• User Behavior Analysis
            """,
            title="Welcome to Recommendation Analysis",
            border_style="blue"
        )
        self.console.print(welcome_panel)
    
    def display_analysis_options(self) -> str:
        """Display and get analysis type selection"""
        table = Table(title="📊 Recommendation Analysis Options", show_header=True, header_style="bold magenta")
        table.add_column("ID", style="cyan", width=10)
        table.add_column("Analysis Type", style="green", width=30)
        table.add_column("Description", style="white", width=50)
        table.add_column("Focus", style="yellow", width=30)
        
        analysis_options = {
            "1": ("recommendation_funnel", "Funnel Analysis", "Analyze user journey from viewing to not clicking recommendations", "Click-through rates"),
            "2": ("time_pattern_analysis", "Time Pattern Analysis", "When users are most/least likely to engage with recommendations", "Time-based patterns"),
            "3": ("user_behavior_analysis", "User Behavior Analysis", "Analyze user engagement patterns and segments", "Behavioral insights")
        }
        
        for option_id, (analysis_type, name, description, focus) in analysis_options.items():
            table.add_row(option_id, name, description, focus)
        
        self.console.print(table)
        
        choice = Prompt.ask(
            "\n[bold cyan]Please select analysis type[/bold cyan]",
            choices=list(analysis_options.keys()),
            default="1"
        )
        
        return analysis_options[choice][0]
    
    def get_data_source_config(self) -> Dict[str, Any]:
        """Get data source configuration for recommendation data"""
        self.console.print("\n[bold yellow]📁 Recommendation Data Source Configuration[/bold yellow]")
        self.console.print("[dim]Expected columns: event, timestamp, device_id, uuid, distinct_id, country, timezone, newDevice[/dim]")
        self.console.print("[dim]MVP focus: timestamp and event columns only[/dim]")
        
        # Simplified data source options for recommendation data
        source_types = {
            "1": ("csv", "CSV File (Recommended)"),
            "2": ("database", "Database Table"),
            "3": ("parquet", "Parquet File")
        }
        
        source_table = Table(show_header=True, header_style="bold magenta")
        source_table.add_column("ID", style="cyan")
        source_table.add_column("Type", style="green")
        source_table.add_column("Description", style="white")
        
        for source_id, (source_type, description) in source_types.items():
            source_table.add_row(source_id, source_type.upper(), description)
        
        self.console.print(source_table)
        
        source_choice = Prompt.ask(
            "[bold cyan]Select data source type[/bold cyan]",
            choices=list(source_types.keys()),
            default="1"
        )
        
        source_type = source_types[source_choice][0]
        
        if source_type == "csv":
            return {
                "type": DataSourceType.CSV,
                "file_path": Prompt.ask(
                    "CSV file path",
                    default="data/input/recommendation_data.csv"
                ),
                "encoding": Prompt.ask("File encoding", default="utf-8"),
                "delimiter": Prompt.ask("CSV delimiter", default=",")
            }
        elif source_type == "database":
            return {
                "type": DataSourceType.DATABASE,
                "connection_string": Prompt.ask(
                    "Database connection string",
                    default=settings.database.url
                ),
                "table_name": Prompt.ask("Table name", default="recommendation_events"),
                "schema": Prompt.ask("Schema name", default="public")
            }
        else:  # parquet
            return {
                "type": DataSourceType.PARQUET,
                "file_path": Prompt.ask(
                    "Parquet file path",
                    default="data/input/recommendation_data.parquet"
                )
            }
    
    def get_analysis_parameters(self, analysis_type: str) -> Dict[str, Any]:
        """Get analysis-specific parameters for recommendation analysis"""
        self.console.print(f"\n[bold yellow]⚙️ Parameters for {analysis_type.replace('_', ' ').title()}[/bold yellow]")
        
        template_config = ANALYSIS_TEMPLATES.get(analysis_type, {})
        required_fields = template_config.get("required_fields", [])
        
        self.console.print(f"[dim]Required fields: {', '.join(required_fields)}[/dim]")
        self.console.print(f"[dim]User identifier: device_id[/dim]")
        self.console.print(f"[dim]Timezone: UTC (default)[/dim]")
        
        # Date range for filtering
        end_date_str = Prompt.ask(
            "End date (YYYY-MM-DD) - leave empty for all data",
            default=""
        )
        start_date_str = Prompt.ask(
            "Start date (YYYY-MM-DD) - leave empty for all data", 
            default=""
        )
        
        # Analysis-specific parameters
        custom_params = {}
        
        if analysis_type == "recommendation_funnel":
            self.console.print("[dim]Funnel analysis will identify exposure and click events automatically[/dim]")
            custom_params["auto_detect_events"] = True
            
        elif analysis_type == "time_pattern_analysis":
            custom_params["timezone"] = Prompt.ask(
                "Analysis timezone (for hour-of-day patterns)",
                default="UTC"
            )
            
        elif analysis_type == "user_behavior_analysis":
            custom_params["segment_method"] = Prompt.ask(
                "Segmentation method (activity_based/time_based)",
                default="activity_based"
            )
        
        # Build parameters
        params = {
            "custom_parameters": custom_params
        }
        
        # Add date range if specified
        if start_date_str or end_date_str:
            params["date_range"] = {}
            if start_date_str:
                params["date_range"]["start_date"] = start_date_str
            if end_date_str:
                params["date_range"]["end_date"] = end_date_str
        
        return params
    
    async def execute_analysis(self, request: AnalysisRequest) -> Dict[str, Any]:
        """Execute the analysis with progress tracking"""
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
        ) as progress:
            
            # Step 1: Manager processing
            task1 = progress.add_task("🤖 Manager: Processing request and loading context...", total=None)
            manager_result = self.manager.process_analysis_request(request)
            progress.update(task1, completed=True)
            
            if manager_result["status"] != "success":
                self.console.print(f"[red]❌ Manager failed: {manager_result['error']}[/red]")
                return manager_result
            
            # Step 2: Execute task sequence
            task_sequence = manager_result["task_sequence"]
            results = []
            
            for i, task in enumerate(task_sequence):
                agent_name = task["agent"]
                task_name = task["task"]
                
                task_id = progress.add_task(
                    f"🔄 {agent_name.title()}: {task_name}...",
                    total=None
                )
                
                # Simulate agent execution (in real implementation, call actual agents)
                await asyncio.sleep(2)  # Simulate processing time
                
                # Simulate success/failure
                import random
                success = random.random() > 0.1  # 90% success rate
                
                if success:
                    progress.update(task_id, completed=True)
                    results.append({
                        "agent": agent_name,
                        "task": task_name,
                        "status": "completed",
                        "output": f"✅ {task_name} completed successfully"
                    })
                else:
                    progress.update(task_id, completed=True)
                    self.console.print(f"[yellow]⚠️ {agent_name} failed, attempting recovery...[/yellow]")
                    
                    # Simulate recovery
                    recovery_task = progress.add_task(f"🔧 Recovering {agent_name}...", total=None)
                    await asyncio.sleep(1)
                    progress.update(recovery_task, completed=True)
                    
                    results.append({
                        "agent": agent_name,
                        "task": task_name,
                        "status": "recovered",
                        "output": f"⚠️ {task_name} completed with recovery"
                    })
            
            # Final step: Generate results
            final_task = progress.add_task("📊 Generating final results and summary...", total=None)
            await asyncio.sleep(1)
            progress.update(final_task, completed=True)
        
        return {
            "status": "success",
            "request_id": request.request_id,
            "results": results,
            "output_files": [
                f"data/output/{request.analysis_type.value}_{request.request_id}.csv",
                f"data/output/{request.analysis_type.value}_{request.request_id}_summary.md"
            ]
        }
    
    def display_results(self, analysis_result: Dict[str, Any]):
        """Display recommendation analysis results"""
        if analysis_result["status"] == "success":
            # Success panel
            success_panel = Panel.fit(
                f"""[bold green]✅ Recommendation Analysis Completed Successfully![/bold green]

[cyan]Request ID:[/cyan] {analysis_result['request_id']}
[cyan]Analysis Focus:[/cyan] Understanding why users don't click recommendations
[cyan]Output Files:[/cyan]
{chr(10).join(f"  • {file}" for file in analysis_result['output_files'])}

[yellow]Agent Execution Summary:[/yellow]
{chr(10).join(f"  • {result['agent']}: {result['output']}" for result in analysis_result['results'])}
                """,
                title="🎉 Analysis Results",
                border_style="green"
            )
            self.console.print(success_panel)
            
            # Display sample recommendation analysis summary
            sample_summary = """
# 📊 Recommendation Click Analysis Report

## Executive Summary
- **Analysis Focus**: Why users don't click on recommended content
- **Data Period**: Last 30 days  
- **Users Analyzed**: 45,670 users who viewed recommendations

## Key Findings

### 📈 Funnel Metrics
- **Exposure Users**: 45,670 (users who saw recommendations)
- **Click Users**: 3,421 (users who clicked)  
- **No-Click Users**: 42,249 (users who didn't click)
- **Click-Through Rate**: 7.5% (industry average: 8-12%)

### ⏰ Time Pattern Insights
- **Peak Engagement Hour**: 8 PM (15.2% higher CTR)
- **Low Engagement Hour**: 3 AM (4.1% CTR)
- **Best Day**: Friday (11.8% CTR)
- **Worst Day**: Monday (5.2% CTR)

### 👥 User Behavior Patterns
- **High Activity Users**: 8,234 users (18.1%) - 12.4% CTR
- **Medium Activity Users**: 22,145 users (48.5%) - 7.8% CTR  
- **Low Activity Users**: 15,291 users (33.4%) - 4.2% CTR

## Data Quality
- **Valid Events**: 98.2% (removed users without proper exposure)
- **Invalid Data Removed**: 847 users who opened app without exposure cards

## Key Recommendations
1. **Optimize Timing**: Focus recommendation delivery during 6-10 PM
2. **Improve Weekend Strategy**: Leverage Friday engagement patterns
3. **Target High-Activity Users**: Personalize for engaged user segments
4. **Investigate Low CTR**: Monday performance needs attention

*Generated by Recommendation Analysis Multi-Agent System*
            """
            
            self.console.print("\n[bold blue]📄 Sample Recommendation Analysis Summary:[/bold blue]")
            self.console.print(Markdown(sample_summary))
            
        else:
            # Error panel
            error_panel = Panel.fit(
                f"""[bold red]❌ Recommendation Analysis Failed[/bold red]

[yellow]Error Details:[/yellow]
{analysis_result.get('error', 'Unknown error occurred')}

[dim]Please check your data format and try again.[/dim]
[dim]Expected columns: event, timestamp, device_id[/dim]
                """,
                title="💥 Error",
                border_style="red"
            )
            self.console.print(error_panel)

# CLI Commands
@app.command()
def analyze(
    interactive: bool = typer.Option(True, "--interactive/--batch", help="Interactive mode or batch processing"),
    config_file: Optional[str] = typer.Option(None, "--config", help="Configuration file path"),
):
    """🚀 Start data analysis with interactive or batch mode"""
    
    app_instance = SekaiDataAnalysisApp()
    
    if interactive:
        # Interactive mode
        app_instance.display_welcome()
        
        try:
            # Get analysis configuration from user
            analysis_type = app_instance.display_analysis_options()
            data_source_config = app_instance.get_data_source_config()
            analysis_params = app_instance.get_analysis_parameters(analysis_type)
            
            # Create analysis request
            request = AnalysisRequest(
                analysis_type=AnalysisType(analysis_type),
                user_query=f"Perform {analysis_type} analysis for Sekai product data",
                data_source_type=data_source_config["type"],
                data_source_config=data_source_config,
                **analysis_params
            )
            
            console.print(f"\n[bold green]🚀 Starting analysis...[/bold green]")
            console.print(f"[dim]Request ID: {request.request_id}[/dim]")
            
            # Execute analysis
            result = asyncio.run(app_instance.execute_analysis(request))
            
            # Display results
            app_instance.display_results(result)
            
        except KeyboardInterrupt:
            console.print("\n[yellow]⏹️ Analysis cancelled by user[/yellow]")
        except Exception as e:
            console.print(f"\n[red]💥 Unexpected error: {str(e)}[/red]")
    
    else:
        # Batch mode with config file
        if not config_file:
            console.print("[red]❌ Config file required for batch mode[/red]")
            raise typer.Exit(1)
        
        try:
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            
            console.print(f"[green]📁 Loading configuration from {config_file}[/green]")
            # Process batch configuration
            console.print("[yellow]🔄 Batch processing not implemented yet[/yellow]")
            
        except FileNotFoundError:
            console.print(f"[red]❌ Config file not found: {config_file}[/red]")
            raise typer.Exit(1)

@app.command()
def list_templates():
    """📋 List available analysis templates"""
    table = Table(title="📊 Analysis Templates", show_header=True, header_style="bold magenta")
    table.add_column("Template ID", style="cyan")
    table.add_column("Name", style="green")
    table.add_column("Description", style="white")
    table.add_column("Required Fields", style="yellow")
    
    for template_id, template_config in ANALYSIS_TEMPLATES.items():
        table.add_row(
            template_id,
            template_config["name"],
            template_config["description"],
            ", ".join(template_config["required_fields"])
        )
    
    console.print(table)

@app.command()
def check_system():
    """🔍 Check system status and configuration"""
    
    status_panel = Panel.fit(
        f"""[bold blue]🔧 System Status[/bold blue]

[green]✅ Configuration:[/green]
  • OpenAI Model: {settings.openai.model}
  • Database URL: {settings.database.url[:50]}...
  • Analysis Focus: Recommendation Click Analysis
  • User Identifier: device_id
  • Timezone: UTC (default)

[green]✅ Paths:[/green]
  • Input Data: {settings.paths.input_data_path}
  • Output Data: {settings.paths.output_data_path}
  • Templates: {settings.paths.templates_path}
  • Logs: {settings.paths.logs_path}

[green]✅ Analysis Settings:[/green]
  • Max Retries: {settings.analysis.max_retries}
  • Chunk Size: {settings.analysis.chunk_size:,}
  • Sample Size: {settings.analysis.sample_size:,}
  • Min Quality Score: {settings.analysis.min_data_quality_score}

[green]✅ Available Analysis Types:[/green]
  • {len(ANALYSIS_TEMPLATES)} recommendation analysis types configured
  • Focus: Funnel, Time Patterns, User Behavior

[yellow]💡 Expected Data Format:[/yellow]
  • Required: event, timestamp, device_id
  • Optional: uuid, distinct_id, country, timezone, newDevice
  • MVP: Using timestamp and event columns only
        """,
        title="System Check",
        border_style="blue"
    )
    
    console.print(status_panel)

@app.command()
def version():
    """📋 Show version information"""
    version_info = Panel.fit(
        """[bold blue]🎮 Recommendation Click Analysis System[/bold blue]

[green]Version:[/green] 1.0.0 (Simplified MVP)
[green]Framework:[/green] CrewAI 0.70.1
[green]AI Model:[/green] OpenAI GPT-4o
[green]Python:[/green] 3.11+

[yellow]🎯 Specialized for understanding why users don't click recommendations[/yellow]
[white]📁 Data: Users who viewed but didn't click recommendations[/white]
[white]🔧 MVP: Timestamp and event analysis focus[/white]
        """,
        title="Version Information",
        border_style="blue"
    )
    console.print(version_info)

@app.command()
def list_agents():
    """🤖 List all available agents in the system"""
    table = Table(title="🤖 Recommendation Analysis Multi-Agent System", show_header=True, header_style="bold magenta")
    table.add_column("Agent", style="cyan", width=20)
    table.add_column("Role", style="green", width=25)
    table.add_column("Description", style="white", width=50)
    table.add_column("Status", style="yellow", width=10)
    
    agent_info = get_agent_info()
    
    for agent_id, info in agent_info.items():
        # Try to instantiate controller to check if agent is working
        try:
            controller_class = info["controller"]
            controller = controller_class()
            status = "✅ Ready"
        except Exception as e:
            status = "❌ Error"
        
        table.add_row(
            info["name"],
            info["role"],
            info["description"],
            status
        )
    
    console.print(table)
    
    # Show simplified workflow sequence for recommendation analysis
    workflow_panel = Panel.fit(
        """[bold cyan]Recommendation Analysis Workflow:[/bold cyan]

1. 🎯 Manager Agent - Orchestrates recommendation analysis workflow
2. 📥 Data Pulling Agent - Loads recommendation event data (MVP: timestamp + event)
3. 🔧 Preprocessing Agent - Cleans invalid exposures, prepares data for analysis
4. 📊 Analysis Agent - Performs funnel, time pattern, and behavior analysis
5. ✅ QA Agent - Validates data quality and analysis accuracy  
6. 💡 Insight Agent - Generates insights on why users don't click recommendations

[bold yellow]🎯 Focus: Understanding User Behavior Around Recommendations[/bold yellow]
[white]• Data: Users who viewed but didn't click recommendations[/white]
[white]• Goal: Identify patterns and improve click-through rates[/white]
[white]• MVP: Analyzing timestamp and event patterns only[/white]
        """,
        title="Simplified Workflow",
        border_style="cyan"
    )
    
    console.print(workflow_panel)

if __name__ == "__main__":
    app() 