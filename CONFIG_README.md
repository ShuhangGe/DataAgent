# Configuration System Guide

The Event-Based User Analysis System uses a YAML-based configuration system that allows you to customize all aspects of the analysis without modifying code.

## Quick Start

1. **Copy the example configuration:**
   ```bash
   cp config_example.yaml config.yaml
   ```

2. **Edit your configuration:**
   ```bash
   # Edit with your preferred editor
   nano config.yaml
   # or
   code config.yaml
   ```

3. **Run with configuration:**
   ```bash
   python main.py config.yaml
   # or use default config.yaml
   python main.py
   ```

## Configuration File Structure

### Database Configuration
```yaml
database:
  path: "DataProcess/event_analysis.db"  # Path to SQLite database
  table_name: "device_event_dictionaries"  # Table containing processed data
```

### LLM Configuration
```yaml
llm:
  model: "gpt-4o-mini"  # OpenAI model to use
  temperature: 0.3      # Response creativity (0.0-2.0)
  api_key: ""          # API key (or use environment variable)
  max_tokens: null     # Max response length (null = model default)
  timeout: 60          # Request timeout in seconds
```

**Available Models:**
- `gpt-4` - Highest quality, most expensive
- `gpt-4o` - Optimized GPT-4 variant
- `gpt-4o-mini` - Fast and cost-effective (default)
- `gpt-3.5-turbo` - Good balance of speed and quality
- `gpt-3.5-turbo-16k` - Larger context window

**Temperature Guide:**
- `0.0-0.3` - Very focused, deterministic (good for analysis)
- `0.3-0.7` - Balanced creativity and consistency
- `0.7-1.0` - More creative responses
- `1.0-2.0` - Maximum creativity (may be less accurate)

### Analysis Configuration
```yaml
analysis:
  print_details: false        # Show detailed pattern insights
  enable_ai_insights: true    # Generate AI analysis
  enable_recommendations: true # Create actionable recommendations
```

### Output Configuration
```yaml
output:
  default_file: "analysis_report.md"  # Output file path
  format: "report"                    # Output format: "report" or "json"
  auto_save: true                     # Automatically save results
  include_raw_data: false             # Include raw data (JSON format only)
  
  # Report configuration (when format is "report")
  report:
    include_summary: true             # Executive summary section
    include_statistics: true          # Key statistics and metrics
    include_visualizations: true      # Event pattern tables/charts
    include_ai_insights: true         # AI-generated insights
    include_recommendations: true     # Actionable recommendations
    include_technical_details: false  # Technical details and logs
```

**Output Formats:**
- `report` - Comprehensive markdown report (recommended)
- `json` - Raw JSON data for programmatic use

**Report Sections:**
- **Executive Summary** - High-level overview and key findings
- **Statistics** - Session counts, event frequencies, conversion rates
- **Visualizations** - Event transitions, temporal patterns, sequences
- **AI Insights** - LLM-generated behavioral analysis
- **Recommendations** - Actionable suggestions for improvement
- **Technical Details** - Workflow logs and debugging information

### Logging Configuration
```yaml
logging:
  level: "INFO"              # Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
  log_to_file: false         # Save logs to file
  log_file: "analysis.log"   # Log file path
```

### Performance Configuration
```yaml
performance:
  max_sessions: 0      # Limit sessions processed (0 = no limit)
  batch_size: 1000     # Processing batch size
  enable_parallel: false # Enable parallel processing
```

## Configuration Priority

The system uses the following priority order (highest to lowest):

1. **Environment variables** (highest priority)
2. **Configuration file values**
3. **Default values** (lowest priority)

### Environment Variable Overrides

Set these environment variables to override configuration:

```bash
# OpenAI API Key (most common)
export OPENAI_API_KEY="sk-your-api-key-here"

# Database path
export DB_PATH="/path/to/database.db"

# Output file
export OUTPUT_FILE="results.json"

# Print details
export PRINT_DETAILS="true"
```

## Usage Examples

```bash
# Use default config.yaml
python main.py

# Use specific config file
python main.py production_config.yaml

# Use config in different directory
python main.py configs/research_config.yaml
```

## Example Configurations

### Development Configuration
```yaml
# config_dev.yaml
llm:
  model: "gpt-4o-mini"  # Fast and cheap for testing
  temperature: 0.3
analysis:
  print_details: true   # Verbose output for debugging
output:
  include_raw_data: true  # Full data for inspection
logging:
  level: "DEBUG"        # Detailed logs
performance:
  max_sessions: 100     # Limit for fast testing
```

### Production Configuration
```yaml
# config_prod.yaml
llm:
  model: "gpt-4"        # Highest quality
  temperature: 0.1      # Very focused
analysis:
  print_details: false  # Clean output
output:
  include_raw_data: false  # Smaller files
  pretty_print: false      # Compact JSON
logging:
  level: "INFO"         # Essential logs only
  log_to_file: true     # Save logs
performance:
  max_sessions: 0       # Process all data
  enable_parallel: true # Maximum performance
```

### Research Configuration
```yaml
# config_research.yaml
llm:
  model: "gpt-4"
  temperature: 0.5      # More creative insights
  max_tokens: 4000      # Longer responses
analysis:
  print_details: true
  enable_ai_insights: true
  enable_recommendations: true
output:
  default_file: "research_analysis.json"
  include_raw_data: true
  pretty_print: true
```

## Configuration Validation

The system automatically validates your configuration and will:

- ✅ Use default values for missing settings
- ⚠️  Warn about invalid values and use defaults
- ❌ Stop execution for critical errors (missing API key, invalid database)

## Testing Configuration

Test your configuration without running full analysis:

```bash
# Test configuration loading
python config_loader.py

# Test with specific config file
python -c "from config_loader import load_config; config = load_config('your_config.yaml'); config.print_config_summary()"
```

## Troubleshooting

### Common Issues

1. **"Configuration file not found"**
   - Create `config.yaml` from `config_example.yaml`
   - Check file path and permissions

2. **"Invalid LLM temperature"**
   - Ensure temperature is between 0.0 and 2.0
   - Use decimal format (0.3, not .3)

3. **"OpenAI API key not provided"**
   - Set `OPENAI_API_KEY` environment variable
   - Or add `api_key` to config file

4. **"Database not found"**
   - Run data processing first: `python run_data_processing.py`
   - Check database path in configuration
   - Ensure file permissions allow reading

5. **"YAML parsing error"**
   - Check YAML syntax (indentation, colons, quotes)
   - Use a YAML validator online
   - Ensure no tabs (use spaces only)

6. **"Report generation failed"**
   - Check output directory permissions
   - Ensure sufficient disk space
   - Verify report configuration settings

### Getting Help

1. **View current configuration:**
   ```bash
   python main.py config.yaml
   # Configuration summary is printed at startup
   ```

2. **Test configuration loading:**
   ```bash
   python config_loader.py
   ```

3. **Test report generation:**
   ```bash
   python report_generator.py
   ```

4. **Validate YAML syntax:**
   ```bash
   python -c "import yaml; yaml.safe_load(open('config.yaml'))"
   ```

## Best Practices

1. **Version Control:**
   - ✅ `config.yaml` is already in `.gitignore` - your personal config won't be uploaded
   - ✅ `config_example.yaml` is tracked in git for others to use as a template
   - ✅ Copy `config_example.yaml` to `config.yaml` and customize for your needs
   - ✅ Use environment-specific config files (`config_dev.yaml`, `config_prod.yaml`)
   - ❌ Never commit files with API keys or sensitive paths
   - ❌ Analysis reports are also ignored by git (contain sensitive insights)

2. **Security:**
   - Never commit API keys to version control
   - Use environment variables for sensitive data
   - Set appropriate file permissions on config files
   - Be cautious with report sharing (may contain business insights)

3. **Organization:**
   - Use descriptive config file names (`config_prod.yaml`, `config_dev.yaml`)
   - Use descriptive report file names (`weekly_analysis_report.md`)
   - Document custom configurations
   - Keep backups of working configurations

4. **Performance:**
   - Use `gpt-4o-mini` for development and testing
   - Use `gpt-4` for production analysis
   - Adjust `max_sessions` for testing with large datasets
   - Enable parallel processing for production workloads
   - Use `format: "json"` for automated processing pipelines