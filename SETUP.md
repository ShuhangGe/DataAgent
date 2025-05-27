# Quick Setup Guide

## Getting Started

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd DataAgent
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Create your configuration:**
   ```bash
   # Copy the example configuration
   cp config_example.yaml config.yaml
   
   # Edit with your settings
   nano config.yaml  # or use your preferred editor
   ```

4. **Set your OpenAI API key:**
   ```bash
   # Option 1: Environment variable (recommended)
   export OPENAI_API_KEY="sk-your-api-key-here"
   
   # Option 2: Add to config.yaml (less secure)
   # Edit config.yaml and set llm.api_key: "sk-your-api-key-here"
   ```

5. **Process your data (if needed):**
   ```bash
   python run_data_processing.py
   ```

6. **Run the analysis:**
   ```bash
   python main.py
   ```

## Configuration Files

- `config_example.yaml` - Template configuration (tracked in git)
- `config.yaml` - Your personal configuration (ignored by git)
- `CONFIG_README.md` - Detailed configuration guide

## Important Notes

- ✅ Your `config.yaml` file is automatically ignored by git
- ✅ Never commit API keys or sensitive information
- ✅ Use environment variables for sensitive data
- ✅ The example config shows all available options

## Need Help?

- Read `CONFIG_README.md` for detailed configuration options
- Check `README.md` for project overview
- Test your config: `python config_loader.py` 