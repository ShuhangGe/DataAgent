# 🤖 Dynamic Multi-Agent Question-Answering System

**Ask questions about your data in natural language!**

A powerful multi-agent system built with CrewAI that understands natural language questions and coordinates specialized agents to provide accurate, insightful answers about your data.

## 🎯 What Makes This Special?

**Dynamic Question Understanding**: Instead of fixed analysis workflows, the system understands what you're asking and creates custom workflows tailored to your specific questions.

**Natural Language Interface**: Ask questions like a human:
- "What data do we have?"
- "Show me user activity trends over the last month"
- "Compare new vs returning users"
- "How many users are active daily?"
- "What factors correlate with user retention?"

## 🚀 Quick Start

### 1. Ask a Question
```bash
python main.py ask "What data do we have?"
```

### 2. Get Question Suggestions
```bash
python main.py suggest-questions
```

### 3. Start Interactive Mode
```bash
python main.py interactive
```

## 💬 Supported Question Types

| Question Type | Examples | What It Does |
|--------------|----------|--------------|
| **Data Exploration** | "What data do we have?", "Show me available tables" | Discovers available data sources and structure |
| **Statistical Summary** | "Show me user summary", "How many users are active?" | Generates descriptive statistics and summaries |
| **Trend Analysis** | "How has engagement changed over time?", "Show monthly trends" | Analyzes changes and patterns over time |
| **Comparison** | "Compare new vs returning users", "Mobile vs web usage" | Compares different segments or groups |
| **Correlation** | "What factors correlate with retention?", "Usage vs revenue relationship" | Finds relationships between variables |
| **Prediction** | "Predict user churn", "Forecast next month's usage" | Builds predictive models and forecasts |

## 🏗️ Architecture

### 6 Specialized Agents Working Together

```mermaid
graph TD
    A[🧠 Dynamic Manager] --> B[📥 Data Pulling Agent]
    A --> C[🔧 Preprocessing Agent] 
    A --> D[📊 Analysis Agent]
    A --> E[✅ QA Agent]
    A --> F[💡 Insight Agent]
    
    B --> C
    C --> D
    D --> E
    E --> F
```

| Agent | Role | Capabilities |
|-------|------|-------------|
| **🧠 Dynamic Manager** | Question-Answering Orchestrator | Natural language understanding, workflow planning, agent coordination |
| **📥 Data Pulling Agent** | Dynamic Data Retrieval | Database querying, multi-source extraction, schema-aware retrieval |
| **🔧 Preprocessing Agent** | Adaptive Data Processing | Data cleaning, feature engineering, type conversion |
| **📊 Analysis Agent** | Question-Driven Analytics | Statistical analysis, trends, comparisons, correlations, predictions |
| **✅ QA Agent** | Result Validation Expert | Quality assessment, result validation, significance testing |
| **💡 Insight Agent** | Answer Synthesis Expert | Natural language generation, insight synthesis, recommendations |

### Dynamic Workflow Creation

Each question gets a **custom workflow** based on its requirements:

- **"What data do we have?"** → Manager + Data Pulling + Insight
- **"Show me user trends"** → Manager + Data Pulling + Preprocessing + Analysis + Insight
- **"Predict user churn"** → All 6 agents with advanced modeling

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.11+
- OpenAI API key
- Database connection (PostgreSQL, MySQL, SQLite, etc.)

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Configure Environment
```bash
cp env.example .env
# Edit .env with your settings:
# OPENAI_API_KEY=your_key_here
# DATABASE_URL=your_database_url
```

### Verify Setup
```bash
python main.py check-system
```

## 📊 Example Usage

### Data Exploration
```bash
python main.py ask "What data do we have in the users table?"
```
**Response**: Shows table schema, sample data, available metrics

### Statistical Analysis  
```bash
python main.py ask "Show me a summary of daily active users"
```
**Response**: Count, averages, trends, key statistics

### Time-Based Analysis
```bash
python main.py ask "How has user engagement changed over the last 3 months?"
```
**Response**: Trend analysis, growth rates, seasonal patterns

### Comparative Analysis
```bash
python main.py ask "Compare user behavior between mobile and web platforms"
```
**Response**: Side-by-side comparison, key differences, insights

### Advanced Analytics
```bash
python main.py ask "What factors are most predictive of user churn?"
```
**Response**: Correlation analysis, feature importance, recommendations

## 🔧 Configuration

### Database Setup
The system supports multiple database types:

```python
# PostgreSQL
DATABASE_URL=postgresql://user:pass@localhost:5432/mydb

# MySQL  
DATABASE_URL=mysql://user:pass@localhost:3306/mydb

# SQLite
DATABASE_URL=sqlite:///data/mydb.db
```

### OpenAI Configuration
```python
OPENAI_API_KEY=your_api_key
OPENAI_MODEL=gpt-4o  # Latest model
OPENAI_TEMPERATURE=0.1
```

## 🎮 Interactive Mode

Start an interactive session for multiple questions:

```bash
python main.py interactive
```

Available commands in interactive mode:
- Type your question naturally
- `suggestions` - Get question ideas
- `history` - Show your session history  
- `help` - Show available commands
- `quit` - Exit

## 📋 CLI Commands

| Command | Description |
|---------|-------------|
| `ask "question"` | Ask a question about your data |
| `suggest-questions` | Get question suggestions based on your data |
| `interactive` | Start interactive Q&A session |
| `list-agents` | Show all available agents |
| `check-system` | Verify system configuration |
| `version` | Show version information |

## 🔍 How It Works

### 1. Question Understanding
The system analyzes your natural language question to:
- Detect question type (exploration, summary, trend, etc.)
- Extract entities (tables, columns, metrics)
- Identify time filters and conditions
- Determine required analysis methods

### 2. Dynamic Planning
Based on your question, it:
- Inspects your database schema
- Identifies relevant data sources
- Creates a custom workflow
- Estimates execution time

### 3. Agent Coordination
The workflow coordinates multiple agents:
- **Data extraction** from relevant sources
- **Data preparation** based on analysis needs
- **Analysis execution** using appropriate methods
- **Quality validation** of results
- **Insight generation** in natural language

### 4. Answer Delivery
You receive:
- Direct answers to your questions
- Supporting data and statistics
- Insights and recommendations
- Visualizations (when applicable)

## 🚀 Advanced Features

### Schema-Aware Querying
The system automatically:
- Discovers available tables and columns
- Suggests relevant questions based on your data
- Adapts queries to your specific schema

### Quality Assurance
Every analysis includes:
- Data quality validation
- Statistical significance testing
- Business rule validation
- Confidence scoring

### Extensible Architecture
Easy to extend with:
- New question types
- Custom analysis methods
- Additional data sources
- Domain-specific insights

## 🛡️ Error Handling

The system gracefully handles:
- Invalid questions or unclear requests
- Missing data or connection issues
- Analysis failures with automatic recovery
- Quality issues with validation warnings

## 📈 Performance

- **Question Understanding**: ~2-5 seconds
- **Simple Queries**: ~30-60 seconds  
- **Complex Analysis**: ~2-5 minutes
- **Predictive Modeling**: ~5-10 minutes

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add your enhancements
4. Write tests for new functionality
5. Submit a pull request

## 📜 License

MIT License - see LICENSE file for details

## 🆘 Support

- **Documentation**: Check this README and inline comments
- **Issues**: Open a GitHub issue
- **Questions**: Use the interactive mode to ask about the system itself!

---

**Transform your data into insights with natural language questions!** 🚀 