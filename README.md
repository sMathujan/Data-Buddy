# CrewAI Tabular Data Agent

## 🚀 Advanced Multi-Agent SQL Assistant with Human-in-the-Loop & Cost Control

A production-ready AI system that enables natural language interaction with tabular datasets using CrewAI's multi-agent framework. This system provides transparent cost tracking, human oversight, and robust security features for enterprise use.

## 🌟 Key Features

### Multi-Agent Architecture
- **Specialized AI Agents**: Five dedicated agents for different aspects of data analysis
- **Sequential Processing**: Structured workflow from schema analysis to insight generation
- **Agent Coordination**: Seamless handoffs between agents with context preservation

### Human-in-the-Loop
- **Query Review**: Human approval required before SQL execution
- **Interactive Interface**: Streamlit-based web application for easy interaction
- **Transparency**: Full visibility into agent reasoning and decision-making

### Cost Control & Monitoring
- **Real-time Tracking**: Monitor LLM usage and costs per session
- **Token Counting**: Accurate token usage calculation with tiktoken
- **Agent Breakdown**: Cost attribution by individual agent
- **Budget Awareness**: Prevent unexpected API costs

### Security & Safety
- **SQL Injection Prevention**: Only safe SELECT statements allowed
- **Query Validation**: Multi-layer security checks before execution
- **Sandboxed Execution**: Isolated database environment
- **Audit Trail**: Complete logging of all operations

## 🏗️ System Architecture

### Agent Roles

1. **Database Schema Analyst**
   - Extracts and analyzes database structure
   - Provides comprehensive schema information
   - Identifies data types, relationships, and constraints

2. **Senior SQL Developer**
   - Converts natural language to SQL queries
   - Optimizes query performance
   - Ensures query accuracy and efficiency

3. **SQL Code Reviewer & Security Auditor**
   - Validates SQL queries for safety
   - Checks for security vulnerabilities
   - Ensures best practices compliance

4. **Database Query Executor**
   - Safely executes approved SQL queries
   - Handles runtime errors gracefully
   - Provides execution statistics

5. **Data Analyst & Insight Generator**
   - Analyzes query results
   - Generates statistical insights
   - Recommends visualizations

### Technology Stack

- **CrewAI**: Multi-agent orchestration framework
- **Streamlit**: Web interface and user interaction
- **SQLite**: In-memory database for data processing
- **Pandas**: Data manipulation and analysis
- **Plotly**: Interactive data visualizations
- **OpenAI GPT**: Language model for AI agents
- **tiktoken**: Token counting for cost tracking

## 📦 Installation

### Prerequisites

- Python 3.11+
- OpenAI API key
- 4GB+ RAM recommended

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd crewai_tabular_agent
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env with your OpenAI API key
   ```

4. **Run the application**
   ```bash
   streamlit run streamlit_app.py
   ```

## 🚀 Quick Start

### 1. Launch the Application
```bash
streamlit run streamlit_app.py
```

### 2. Load Your Dataset
- Upload CSV, Excel, or JSON files via the web interface
- Or use the "Load Sample Dataset" button for testing

### 3. Ask Questions
- Enter natural language queries like:
  - "What is the average sales by region?"
  - "Show me the top 10 customers by revenue"
  - "Which products have the highest profit margins?"

### 4. Review and Approve
- Review the generated SQL query
- Approve or request modifications
- View execution results and insights

## 💡 Usage Examples

### Basic Queries
```
"How many records are in the dataset?"
"What are the column names and data types?"
"Show me the first 10 rows"
```

### Analytical Queries
```
"What is the correlation between price and quantity sold?"
"Which region has the highest average order value?"
"Show me monthly sales trends over time"
```

### Complex Aggregations
```
"Calculate the total revenue by product category and region"
"Find customers who haven't made a purchase in the last 90 days"
"What is the customer lifetime value distribution?"
```

## 🔧 Configuration

### Agent Configuration (`config/agents.yaml`)
```yaml
sql_generator_agent:
  role: Senior SQL Developer
  goal: Convert natural language queries into accurate SQL statements
  model: gpt-4.1-mini
  temperature: 0.2
```

### Task Configuration (`config/tasks.yaml`)
```yaml
sql_generation_task:
  description: Generate SQL query for: {user_query}
  agent: sql_generator_agent
  human_input: true
```

### Environment Variables (`.env`)
```
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_API_BASE=https://api.openai.com/v1
```

## 📊 Cost Management

### Cost Tracking Features
- **Session-based tracking**: Costs isolated per query session
- **Token-level accuracy**: Precise token counting with tiktoken
- **Agent attribution**: Cost breakdown by individual agent
- **Real-time monitoring**: Live cost updates during processing

### Cost Optimization Tips
- Use smaller models (gpt-4.1-mini) for routine tasks
- Implement query caching for repeated questions
- Set token limits to prevent runaway costs
- Monitor usage patterns and optimize prompts

## 🛡️ Security Features

### SQL Injection Prevention
- Whitelist approach: Only SELECT statements allowed
- Keyword filtering: Dangerous operations blocked
- Pattern detection: Suspicious query patterns identified
- Parameterized queries: Safe value substitution

### Data Privacy
- Local processing: Data never leaves your environment
- Temporary storage: Automatic cleanup of temporary files
- Session isolation: No data sharing between sessions
- Audit logging: Complete operation history

## 🧪 Testing

### Unit Tests
```bash
python -m pytest tests/
```

### Integration Tests
```bash
python test_integration.py
```

### Load Testing
```bash
python test_performance.py
```

## 📈 Performance Optimization

### Database Optimization
- Automatic LIMIT clauses to prevent large result sets
- Query timeout protection
- Memory-efficient data processing
- Indexed temporary tables

### Agent Optimization
- Temperature tuning for different agent roles
- Prompt optimization for accuracy and efficiency
- Context management for large datasets
- Parallel processing where possible

## 🔍 Troubleshooting

### Common Issues

**"Model not supported" error**
- Update model names in `config/agents.yaml`
- Ensure API key has access to specified models

**High API costs**
- Check token usage in cost tracking
- Optimize prompts for efficiency
- Use smaller models for simple tasks

**Query timeout errors**
- Increase timeout in database tools
- Optimize query complexity
- Add appropriate LIMIT clauses

**Memory issues with large datasets**
- Process data in chunks
- Use streaming for large files
- Increase system memory allocation

## 🤝 Contributing

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Install development dependencies
4. Run tests before submitting
5. Follow code style guidelines

### Code Style
- Use Black for code formatting
- Follow PEP 8 guidelines
- Add type hints where appropriate
- Include docstrings for all functions

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **CrewAI Team**: For the excellent multi-agent framework
- **Streamlit**: For the intuitive web application framework
- **OpenAI**: For providing powerful language models
- **Community Contributors**: For feedback and improvements

## 📞 Support

For questions, issues, or feature requests:
- Create an issue on GitHub
- Check the documentation
- Review existing discussions
- Contact the development team

---

**Built with ❤️ using CrewAI and modern AI technologies**

