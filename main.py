#!/usr/bin/env python3
"""
Dynamic Multi-Agent Question-Answering System
Built with CrewAI framework for flexible data analysis

Usage:
    python main.py ask "What data do we have?"
    python main.py ask "Show me user activity trends over the last month"
    python main.py ask "Compare new vs returning users"
    python main.py suggest-questions
    python main.py version
    python main.py check-system
"""

import typer
import yaml
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.text import Text
from rich.syntax import Syntax

from src.agents.manager_agent import DynamicManagerController
from src.agents import get_agent_info, list_agents
from src.models.data_models import QuestionType, DynamicAnalysisResult
from src.config.settings import settings

# Initialize Rich console and Typer app
console = Console()
app = typer.Typer(help="🤖 Dynamic Multi-Agent Question-Answering System")

class QuestionAnsweringInterface:
    """Main interface for the dynamic question-answering system"""
    
    def __init__(self):
        self.manager = DynamicManagerController()
        self.session_history = []
        
    def display_welcome(self):
        """Display welcome message and system info"""
        welcome_panel = Panel.fit(
            """[bold blue]🤖 Dynamic Multi-Agent Question-Answering System[/bold blue]

[green]✨ Ask questions about your data in natural language![/green]
[yellow]🔍 The system will understand your questions and coordinate specialized agents[/yellow]

[white]💬 Examples of questions you can ask:[/white]
[cyan]• "What data do we have?"[/cyan]
[cyan]• "Show me user activity trends over the last month"[/cyan]
[cyan]• "Compare new vs returning users"[/cyan]
[cyan]• "How many users are active daily?"[/cyan]
[cyan]• "What factors correlate with user retention?"[/cyan]

[white]📊 Supported Question Types:[/white]
• Data Exploration • Statistical Summary • Trend Analysis
• Comparison • Correlation • Prediction • Custom Queries
            """,
            title="Welcome to Dynamic Data Analysis",
            border_style="blue"
        )
        console.print(welcome_panel)
    
    def process_question(self, question: str) -> Dict[str, Any]:
        """Process a natural language question"""
        
        # Display processing status
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            
            task1 = progress.add_task("🧠 Understanding your question...", total=None)
            task2 = progress.add_task("🔍 Inspecting database schema...", total=None)
            task3 = progress.add_task("📋 Planning analysis workflow...", total=None)
            
            # Process the question
            result = self.manager.process_user_question(question)
            
            progress.update(task1, completed=True)
            progress.update(task2, completed=True) 
            progress.update(task3, completed=True)
        
        # Store in session history
        self.session_history.append({
            "question": question,
            "timestamp": datetime.now(),
            "result": result
        })
        
        return result
    
    def display_question_understanding(self, result: Dict[str, Any]):
        """Display how the system understood the question"""
        if result["status"] != "success":
            return
            
        question_info = result["question"]
        
        understanding_panel = Panel.fit(
            f"""[bold green]✅ Question Understanding[/bold green]

[cyan]Question Type:[/cyan] {question_info['question_type'].replace('_', ' ').title()}
[cyan]Detected Entities:[/cyan] {', '.join(question_info['entities']) if question_info['entities'] else 'None'}
[cyan]Time Filters:[/cyan] {question_info['time_filters'] if question_info['time_filters'] else 'None'}
[cyan]Grouping:[/cyan] {', '.join(question_info['grouping']) if question_info['grouping'] else 'None'}
[cyan]Output Format:[/cyan] {question_info['output_format']}
[cyan]Confidence:[/cyan] {result['confidence']:.1%}

[yellow]Analysis Methods:[/yellow] {', '.join(question_info['analysis_methods'])}
            """,
            title="Question Analysis",
            border_style="green"
        )
        console.print(understanding_panel)
    
    def display_database_context(self, result: Dict[str, Any]):
        """Display database context information"""
        if result["status"] != "success":
            return
            
        db_context = result["database_context"]
        
        # Create table showing available data
        table = Table(title="📊 Available Data Sources", show_header=True, header_style="bold magenta")
        table.add_column("Table", style="cyan", width=20)
        table.add_column("Columns", style="white", width=40)
        table.add_column("Sample Data", style="green", width=30)
        
        for schema in db_context["schemas"][:5]:  # Show top 5 tables
            columns = [col["name"] for col in schema["columns"][:5]]  # Show top 5 columns
            column_text = ", ".join(columns)
            if len(schema["columns"]) > 5:
                column_text += f" (+{len(schema['columns'])-5} more)"
            
            sample_info = "Available" if schema.get("sample_data") else "No sample"
            
            table.add_row(
                schema["table_name"],
                column_text,
                sample_info
            )
        
        console.print(table)
        
        # Show metrics
        if db_context.get("available_metrics"):
            metrics_text = ", ".join(db_context["available_metrics"][:10])
            if len(db_context["available_metrics"]) > 10:
                metrics_text += f" (+{len(db_context['available_metrics'])-10} more)"
            
            metrics_panel = Panel(
                f"[yellow]Available Metrics:[/yellow] {metrics_text}",
                title="📈 Detected Metrics",
                border_style="yellow"
            )
            console.print(metrics_panel)
    
    def display_task_plan(self, result: Dict[str, Any]):
        """Display the planned task execution"""
        if result["status"] != "success":
            return
            
        task_plan = result["task_plan"]
        
        # Create execution plan table
        plan_table = Table(title="🔄 Execution Plan", show_header=True, header_style="bold blue")
        plan_table.add_column("Step", style="cyan", width=5)
        plan_table.add_column("Agent", style="green", width=15)
        plan_table.add_column("Task", style="white", width=25)
        plan_table.add_column("Parameters", style="yellow", width=35)
        
        for i, task in enumerate(task_plan, 1):
            # Format parameters for display
            params = task.get("parameters", {})
            param_text = ", ".join([f"{k}: {v}" for k, v in list(params.items())[:3]])
            if len(params) > 3:
                param_text += f" (+{len(params)-3} more)"
            
            plan_table.add_row(
                str(i),
                task["agent"].replace("_", " ").title(),
                task["task"].replace("_", " ").title(),
                param_text
            )
        
        console.print(plan_table)
        
        # Show estimated time
        estimated_time = result.get("estimated_time", 0)
        time_panel = Panel(
            f"[green]Estimated Execution Time:[/green] {estimated_time} seconds",
            title="⏱️ Time Estimate",
            border_style="green"
        )
        console.print(time_panel)
    
    def suggest_questions(self) -> List[str]:
        """Get question suggestions based on available data"""
        try:
            # Get database schema first
            from src.agents.manager_agent import DatabaseSchemaInspectionTool
            schema_tool = DatabaseSchemaInspectionTool()
            schema_result = schema_tool._run()
            
            if "error" in schema_result:
                return [
                    "What data do we have?",
                    "Show me a summary of user activity",
                    "How has engagement changed over time?",
                    "Compare user behavior by segment"
                ]
            
            from src.models.data_models import DatabaseContext
            db_context = DatabaseContext(**schema_result["database_context"])
            
            return self.manager.suggest_questions(db_context)
            
        except Exception as e:
            console.print(f"[red]Error getting suggestions: {str(e)}[/red]")
            return []

# Initialize the interface
qa_interface = QuestionAnsweringInterface()

@app.command()
def ask(question: str):
    """💬 Ask a question about your data in natural language"""
    
    qa_interface.display_welcome()
    
    console.print(f"\n[bold white]Your Question:[/bold white] [italic]{question}[/italic]\n")
    
    # Process the question
    result = qa_interface.process_question(question)
    
    if result["status"] == "failed":
        error_panel = Panel.fit(
            f"[bold red]❌ Error Processing Question[/bold red]\n\n[red]{result['error']}[/red]",
            title="Error",
            border_style="red"
        )
        console.print(error_panel)
        return
    
    # Display results
    qa_interface.display_question_understanding(result)
    console.print()
    qa_interface.display_database_context(result)
    console.print()
    qa_interface.display_task_plan(result)
    
    # Show next steps
    next_steps_panel = Panel.fit(
        """[bold yellow]🚀 Next Steps[/bold yellow]

[white]The system has analyzed your question and created an execution plan.[/white]
[white]To actually execute the analysis, the individual agents would now:[/white]

[cyan]1. Pull relevant data from the database[/cyan]
[cyan]2. Preprocess and clean the data[/cyan]
[cyan]3. Perform the requested analysis[/cyan]
[cyan]4. Validate the results[/cyan]
[cyan]5. Generate insights and answers[/cyan]

[green]Try asking another question or use 'suggest-questions' for ideas![/green]
        """,
        title="Execution Ready",
        border_style="yellow"
    )
    console.print(next_steps_panel)

@app.command()
def suggest_questions():
    """💡 Get suggestions for questions you can ask"""
    
    console.print("[bold blue]🤖 Generating Question Suggestions...[/bold blue]\n")
    
    suggestions = qa_interface.suggest_questions()
    
    if not suggestions:
        console.print("[yellow]No specific suggestions available. Try asking general questions![/yellow]")
        return
    
    # Display suggestions
    suggestions_table = Table(title="💡 Suggested Questions", show_header=True, header_style="bold green")
    suggestions_table.add_column("#", style="cyan", width=3)
    suggestions_table.add_column("Question", style="white", width=60)
    suggestions_table.add_column("Type", style="yellow", width=15)
    
    for i, suggestion in enumerate(suggestions, 1):
        # Determine question type based on keywords
        if "what data" in suggestion.lower() or "available" in suggestion.lower():
            q_type = "Exploration"
        elif "summary" in suggestion.lower() or "how many" in suggestion.lower():
            q_type = "Summary"
        elif "trend" in suggestion.lower() or "over time" in suggestion.lower():
            q_type = "Trend"
        elif "compare" in suggestion.lower():
            q_type = "Comparison"
        else:
            q_type = "General"
        
        suggestions_table.add_row(str(i), suggestion, q_type)
    
    console.print(suggestions_table)
    
    # Show usage example
    usage_panel = Panel(
        f"""[green]To ask a question, use:[/green]
[cyan]python main.py ask "Your question here"[/cyan]

[green]For example:[/green]
[cyan]python main.py ask "{suggestions[0] if suggestions else 'What data do we have?'}"[/cyan]
        """,
        title="Usage",
        border_style="green"
    )
    console.print(usage_panel)

@app.command()
def version():
    """📋 Show version information"""
    version_info = Panel.fit(
        """[bold blue]🤖 Dynamic Multi-Agent Question-Answering System[/bold blue]

[green]Version:[/green] 2.0.0 (Dynamic Q&A)
[green]Framework:[/green] CrewAI 0.70.1
[green]AI Model:[/green] OpenAI GPT-4o
[green]Python:[/green] 3.11+

[yellow]🎯 Ask questions about your data in natural language[/yellow]
[white]💬 Supports various question types: exploration, summary, trends, comparison, correlation, prediction[/white]
[white]🔧 Dynamic workflow planning based on your specific questions[/white]
        """,
        title="Version Information",
        border_style="blue"
    )
    console.print(version_info)

@app.command()
def check_system():
    """🔍 Check system status and configuration"""
    
    status_panel = Panel.fit(
        f"""[bold green]✅ System Status Check[/bold green]

[cyan]OpenAI API:[/cyan] {'✅ Configured' if settings.openai.api_key else '❌ Missing API Key'}
[cyan]Database:[/cyan] {'✅ Configured' if settings.database.url else '❌ Not Configured'}
[cyan]Question Understanding:[/cyan] ✅ Ready
[cyan]Schema Inspection:[/cyan] ✅ Ready
[cyan]Dynamic Planning:[/cyan] ✅ Ready

[yellow]Configuration:[/yellow]
• Model: {settings.openai.model}
• Temperature: {settings.openai.temperature}
• Max Tokens: {settings.openai.max_tokens}
• Database: {settings.database.url.split('/')[-1] if settings.database.url else 'Not configured'}
        """,
        title="System Status",
        border_style="green"
    )
    console.print(status_panel)

@app.command()
def list_agents():
    """🤖 List all available agents in the system"""
    table = Table(title="🤖 Dynamic Multi-Agent System", show_header=True, header_style="bold magenta")
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
    
    # Show workflow info
    workflow_panel = Panel(
        """[yellow]🔄 Dynamic Workflow:[/yellow]
The system creates custom workflows based on your questions:

[cyan]1. Manager Agent[/cyan] - Understands questions and plans workflows
[cyan]2. Data Pulling Agent[/cyan] - Extracts relevant data
[cyan]3. Preprocessing Agent[/cyan] - Cleans and prepares data
[cyan]4. Analysis Agent[/cyan] - Performs analysis based on question type
[cyan]5. QA Agent[/cyan] - Validates results
[cyan]6. Insight Agent[/cyan] - Generates answers and insights

[green]Each question gets a custom workflow tailored to its requirements![/green]
        """,
        title="Workflow Information",
        border_style="yellow"
    )
    console.print(workflow_panel)

@app.command()
def interactive():
    """🔄 Start interactive question-answering session"""
    
    qa_interface.display_welcome()
    
    console.print("\n[bold green]🔄 Interactive Mode Started![/bold green]")
    console.print("[white]Type your questions or 'help' for commands, 'quit' to exit[/white]\n")
    
    while True:
        try:
            # Get user input
            question = typer.prompt("🤖 Ask a question")
            
            if question.lower() in ['quit', 'exit', 'q']:
                console.print("[green]👋 Goodbye![/green]")
                break
            elif question.lower() in ['help', 'h']:
                help_panel = Panel(
                    """[yellow]Available Commands:[/yellow]
• Just type your question naturally
• 'suggestions' - Get question suggestions
• 'history' - Show session history
• 'help' - Show this help
• 'quit' - Exit interactive mode

[cyan]Example Questions:[/cyan]
• What data do we have?
• Show me user trends over time
• Compare new vs returning users
                    """,
                    title="Help",
                    border_style="yellow"
                )
                console.print(help_panel)
                continue
            elif question.lower() == 'suggestions':
                suggestions = qa_interface.suggest_questions()
                for i, suggestion in enumerate(suggestions[:5], 1):
                    console.print(f"[cyan]{i}.[/cyan] {suggestion}")
                continue
            elif question.lower() == 'history':
                if qa_interface.session_history:
                    console.print("[yellow]Session History:[/yellow]")
                    for i, item in enumerate(qa_interface.session_history, 1):
                        console.print(f"[cyan]{i}.[/cyan] {item['question']}")
                else:
                    console.print("[yellow]No questions asked yet![/yellow]")
                continue
            
            # Process the question
            result = qa_interface.process_question(question)
            
            if result["status"] == "failed":
                console.print(f"[red]❌ Error: {result['error']}[/red]\n")
                continue
            
            # Show brief summary
            question_info = result["question"]
            console.print(f"[green]✅ Understood as:[/green] {question_info['question_type'].replace('_', ' ').title()}")
            console.print(f"[cyan]📊 Found {len(result['database_context']['schemas'])} relevant tables[/cyan]")
            console.print(f"[yellow]⏱️ Estimated time: {result['estimated_time']} seconds[/yellow]\n")
            
        except KeyboardInterrupt:
            console.print("\n[green]👋 Goodbye![/green]")
            break
        except Exception as e:
            console.print(f"[red]❌ Error: {str(e)}[/red]\n")

if __name__ == "__main__":
    app() 