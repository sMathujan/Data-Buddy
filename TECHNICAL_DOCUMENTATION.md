# CrewAI Tabular Data Agent - Technical Documentation

## Table of Contents
1. [System Architecture](#system-architecture)
2. [Agent Design and Implementation](#agent-design-and-implementation)
3. [Multi-Agent Workflow](#multi-agent-workflow)
4. [Human-in-the-Loop Implementation](#human-in-the-loop-implementation)
5. [Cost Tracking and Control](#cost-tracking-and-control)
6. [Security and Safety Measures](#security-and-safety-measures)
7. [Database Tools and Integration](#database-tools-and-integration)
8. [Performance Optimization](#performance-optimization)
9. [Deployment and Scaling](#deployment-and-scaling)
10. [API Reference](#api-reference)

## System Architecture

### Overview

The CrewAI Tabular Data Agent represents a sophisticated multi-agent system designed to bridge the gap between natural language queries and structured data analysis. The architecture follows a modular, microservices-inspired design where each agent specializes in a specific aspect of the data analysis pipeline. This approach ensures maintainability, scalability, and clear separation of concerns.

### Core Components

#### 1. CrewAI Orchestrator (`crew_orchestrator.py`)

The central orchestration engine manages the entire multi-agent workflow. It serves as the primary interface between the user interface and the agent ecosystem, handling session management, agent coordination, and result aggregation.

**Key Responsibilities:**
- Agent lifecycle management and initialization
- Task creation and context propagation
- Session-based cost tracking integration
- Error handling and recovery mechanisms
- Resource cleanup and memory management

**Implementation Details:**
```python
class CrewAITabularAgent:
    def __init__(self, config_dir: str = "config"):
        self.data_loader = DataLoader()
        self.cost_tracker = CostTracker()
        self.agents_config = self._load_config("agents.yaml")
        self.tasks_config = self._load_config("tasks.yaml")
```

The orchestrator implements a factory pattern for agent creation, allowing for dynamic configuration and easy extensibility. Each agent is instantiated with specific tools and configurations based on their role in the pipeline.

#### 2. Data Loading and Schema Extraction (`utils/data_loader.py`)

The data loading subsystem handles the ingestion and preprocessing of various file formats, converting them into a standardized SQLite database format for consistent querying across the system.

**Supported Formats:**
- CSV (Comma-Separated Values)
- Excel (XLSX, XLS)
- JSON (JavaScript Object Notation)

**Schema Extraction Process:**
1. **File Validation**: Ensures file format compatibility and size constraints
2. **Data Type Inference**: Automatically detects column data types and constraints
3. **Null Value Analysis**: Identifies missing data patterns and percentages
4. **Statistical Profiling**: Generates basic statistics for numeric columns
5. **Relationship Detection**: Identifies potential foreign key relationships

The schema extraction generates comprehensive metadata that informs subsequent agents about the data structure, enabling more accurate SQL generation and analysis.

#### 3. Custom Database Tools (`utils/database_tools.py`)

A suite of specialized tools designed specifically for CrewAI integration, providing safe and controlled database interactions.

**Tool Categories:**

**SQLiteQueryTool**: Executes SQL queries with built-in safety mechanisms
- Query validation and sanitization
- Automatic LIMIT clause injection
- Timeout protection for long-running queries
- Result formatting and error handling

**SchemaInspectorTool**: Provides detailed database schema information
- Table structure analysis
- Column metadata extraction
- Sample data generation
- Relationship mapping

**DataAnalysisTool**: Performs statistical analysis on query results
- Descriptive statistics calculation
- Data quality assessment
- Visualization recommendations
- Insight generation

**QueryValidatorTool**: Validates SQL queries before execution
- Syntax checking using EXPLAIN QUERY PLAN
- Security vulnerability detection
- Performance impact assessment
- Best practice compliance verification

#### 4. Cost Tracking System (`utils/cost_tracker.py`)

A comprehensive cost monitoring system that provides real-time tracking of LLM usage and associated costs, essential for production deployments where cost control is critical.

**Features:**
- **Token-level Accuracy**: Uses tiktoken for precise token counting
- **Model-specific Pricing**: Maintains up-to-date pricing for different OpenAI models
- **Session Isolation**: Tracks costs per user session for accountability
- **Agent Attribution**: Breaks down costs by individual agent for optimization
- **Real-time Monitoring**: Provides live cost updates during processing

**Cost Calculation Algorithm:**
```python
def calculate_cost(self, prompt_tokens: int, completion_tokens: int, model: str) -> float:
    pricing = self.PRICING[model]
    prompt_cost = (prompt_tokens / 1000) * pricing['prompt']
    completion_cost = (completion_tokens / 1000) * pricing['completion']
    return prompt_cost + completion_cost
```

### Data Flow Architecture

The system follows a sequential data flow pattern where each agent builds upon the work of its predecessors:

1. **Input Processing**: User query and dataset ingestion
2. **Schema Analysis**: Database structure extraction and documentation
3. **Query Generation**: Natural language to SQL conversion
4. **Security Review**: Query validation and safety verification
5. **Execution**: Safe query execution with monitoring
6. **Analysis**: Result interpretation and insight generation
7. **Presentation**: User-friendly result formatting and visualization

This linear flow ensures data integrity and provides multiple checkpoints for quality assurance and human intervention.

## Agent Design and Implementation

### Agent Architecture Principles

Each agent in the system is designed following the Single Responsibility Principle, where each agent has a clearly defined role and expertise area. This design promotes maintainability, testability, and allows for independent optimization of each component.

### Individual Agent Specifications

#### 1. Database Schema Analyst

**Role**: Extract and provide comprehensive database schema information for SQL query generation

**Core Capabilities:**
- Database structure analysis and documentation
- Data type identification and validation
- Relationship mapping between tables
- Statistical profiling of data distributions
- Sample data extraction for context

**Implementation Strategy:**
The Schema Analyst uses a combination of SQL introspection queries and pandas-based analysis to build a comprehensive understanding of the dataset structure. It generates human-readable schema descriptions that serve as context for subsequent agents.

**Key Outputs:**
- Detailed schema documentation
- Column metadata with statistics
- Sample data for reference
- Data quality assessment
- Relationship recommendations

#### 2. Senior SQL Developer

**Role**: Convert natural language queries into accurate and efficient SQL statements

**Core Capabilities:**
- Natural language understanding and interpretation
- SQL query construction and optimization
- Business logic translation to database operations
- Query performance consideration
- Error handling and edge case management

**Prompt Engineering Strategy:**
The SQL Developer agent uses a sophisticated prompt structure that includes:
- Complete schema context from the Schema Analyst
- Few-shot examples of natural language to SQL conversions
- Best practices for query optimization
- Safety guidelines for query construction

**Query Generation Process:**
1. **Intent Recognition**: Identifies the type of analysis requested
2. **Entity Extraction**: Maps natural language entities to database columns
3. **Logic Construction**: Builds the appropriate SQL logic structure
4. **Optimization**: Applies performance optimization techniques
5. **Validation**: Performs initial syntax and logic validation

#### 3. SQL Code Reviewer and Security Auditor

**Role**: Review, validate, and optimize SQL queries for correctness, performance, and security

**Security Validation Framework:**
- **Whitelist Approach**: Only SELECT statements are permitted
- **Keyword Filtering**: Dangerous operations (DROP, DELETE, UPDATE) are blocked
- **Pattern Detection**: Identifies potential SQL injection attempts
- **Syntax Validation**: Uses EXPLAIN QUERY PLAN for syntax verification
- **Performance Assessment**: Evaluates query complexity and resource usage

**Review Criteria:**
1. **Security Compliance**: Ensures no dangerous operations
2. **Syntax Correctness**: Validates SQL syntax and structure
3. **Performance Impact**: Assesses query efficiency
4. **Best Practices**: Checks adherence to SQL coding standards
5. **Result Accuracy**: Verifies query logic matches intent

#### 4. Database Query Executor

**Role**: Execute validated SQL queries against the database and return results safely

**Execution Framework:**
- **Timeout Protection**: Prevents long-running queries from blocking the system
- **Resource Monitoring**: Tracks memory and CPU usage during execution
- **Error Handling**: Graceful handling of runtime errors
- **Result Formatting**: Standardizes output format for consistency
- **Audit Logging**: Records all executed queries for compliance

**Safety Mechanisms:**
- Automatic LIMIT clause injection for large result sets
- Query timeout enforcement
- Memory usage monitoring
- Connection pooling and management
- Transaction isolation

#### 5. Data Analyst and Insight Generator

**Role**: Analyze query results and generate meaningful insights and visualizations

**Analysis Capabilities:**
- **Descriptive Statistics**: Mean, median, mode, standard deviation
- **Distribution Analysis**: Histograms, box plots, density curves
- **Correlation Analysis**: Pearson and Spearman correlation coefficients
- **Trend Detection**: Time series analysis and pattern recognition
- **Anomaly Detection**: Outlier identification and analysis

**Insight Generation Process:**
1. **Data Profiling**: Analyzes result set characteristics
2. **Statistical Analysis**: Computes relevant statistical measures
3. **Pattern Recognition**: Identifies trends and relationships
4. **Visualization Recommendations**: Suggests appropriate chart types
5. **Natural Language Summary**: Generates human-readable insights

## Multi-Agent Workflow

### Sequential Processing Model

The multi-agent workflow follows a carefully orchestrated sequence designed to maximize accuracy while maintaining efficiency. Each stage builds upon the previous one, creating a robust pipeline for natural language data analysis.

### Workflow Stages

#### Stage 1: Schema Extraction and Analysis

**Objective**: Establish a comprehensive understanding of the dataset structure

**Process Flow:**
1. **File Ingestion**: The uploaded dataset is processed and converted to SQLite format
2. **Schema Introspection**: Database metadata is extracted using SQL system tables
3. **Statistical Profiling**: Basic statistics are computed for all columns
4. **Data Quality Assessment**: Null values, duplicates, and anomalies are identified
5. **Documentation Generation**: Human-readable schema description is created

**Output Artifacts:**
- Structured schema metadata
- Statistical summaries
- Data quality report
- Sample data extracts

#### Stage 2: Natural Language to SQL Translation

**Objective**: Convert user queries into executable SQL statements

**Translation Process:**
1. **Query Parsing**: Natural language query is analyzed for intent and entities
2. **Context Integration**: Schema information is incorporated into the translation context
3. **SQL Generation**: Appropriate SQL query is constructed using LLM capabilities
4. **Initial Validation**: Basic syntax and logic checks are performed
5. **Optimization**: Query is optimized for performance and clarity

**Quality Assurance:**
- Intent verification against user query
- Schema compliance checking
- Syntax validation
- Performance consideration

#### Stage 3: Security Review and Validation

**Objective**: Ensure query safety and compliance with security policies

**Review Process:**
1. **Security Scanning**: Query is scanned for dangerous operations and patterns
2. **Syntax Verification**: EXPLAIN QUERY PLAN is used to validate syntax
3. **Performance Assessment**: Query complexity and resource requirements are evaluated
4. **Compliance Checking**: Adherence to organizational policies is verified
5. **Approval Recommendation**: Final recommendation for query execution is made

**Security Checkpoints:**
- Operation type validation (SELECT only)
- Injection pattern detection
- Resource usage estimation
- Compliance verification

#### Stage 4: Query Execution and Monitoring

**Objective**: Safely execute approved queries with comprehensive monitoring

**Execution Framework:**
1. **Pre-execution Setup**: Connection establishment and resource allocation
2. **Monitored Execution**: Query execution with real-time monitoring
3. **Result Collection**: Systematic collection and formatting of results
4. **Performance Logging**: Execution metrics and statistics recording
5. **Resource Cleanup**: Proper cleanup of connections and temporary resources

**Monitoring Metrics:**
- Execution time
- Memory usage
- Row count returned
- Error conditions
- Resource utilization

#### Stage 5: Analysis and Insight Generation

**Objective**: Transform raw query results into actionable insights

**Analysis Pipeline:**
1. **Data Profiling**: Comprehensive analysis of result set characteristics
2. **Statistical Computation**: Calculation of relevant statistical measures
3. **Pattern Detection**: Identification of trends, correlations, and anomalies
4. **Visualization Planning**: Recommendation of appropriate visualization types
5. **Insight Synthesis**: Generation of natural language insights and recommendations

**Insight Categories:**
- Descriptive insights (what happened)
- Diagnostic insights (why it happened)
- Predictive insights (what might happen)
- Prescriptive insights (what should be done)

### Context Propagation

A critical aspect of the multi-agent workflow is the seamless propagation of context between agents. Each agent receives not only the direct output of its predecessor but also relevant context from earlier stages.

**Context Elements:**
- Original user query and intent
- Complete schema information
- Generated SQL query and rationale
- Security review results and recommendations
- Execution results and metadata

This rich context enables each agent to make informed decisions and provide more accurate and relevant outputs.

### Error Handling and Recovery

The workflow includes comprehensive error handling mechanisms at each stage:

**Error Categories:**
1. **Input Validation Errors**: Invalid file formats or corrupted data
2. **Schema Extraction Errors**: Database connection or structure issues
3. **Translation Errors**: Ambiguous queries or unsupported operations
4. **Security Violations**: Dangerous queries or policy violations
5. **Execution Errors**: Runtime failures or resource constraints
6. **Analysis Errors**: Insufficient data or computational failures

**Recovery Strategies:**
- Graceful degradation with partial results
- Alternative approach suggestions
- Human intervention requests
- Automatic retry with modified parameters
- Fallback to simpler analysis methods

## Human-in-the-Loop Implementation

### Philosophy and Design Principles

The human-in-the-loop (HITL) implementation is based on the principle that AI systems should augment human decision-making rather than replace it entirely. This approach is particularly critical in data analysis scenarios where business context and domain expertise are essential for accurate interpretation.

### HITL Integration Points

#### 1. Query Review and Approval

**Implementation**: Before any SQL query is executed, it is presented to the user for review and approval through the Streamlit interface.

**Review Interface Features:**
- **Syntax Highlighting**: SQL queries are displayed with proper syntax highlighting for readability
- **Explanation Generation**: Natural language explanation of what the query will accomplish
- **Security Assessment**: Clear indication of security review results
- **Modification Options**: Ability to edit queries before execution
- **Approval Workflow**: Explicit approval required before execution

**User Decision Points:**
- Approve query as-is
- Request modifications with specific feedback
- Reject query and provide alternative requirements
- Ask for clarification on query logic

#### 2. Result Interpretation and Validation

**Implementation**: Query results are presented with AI-generated insights, but users can provide feedback and alternative interpretations.

**Validation Features:**
- **Result Preview**: Clear presentation of query results with formatting
- **Insight Review**: AI-generated insights presented for user validation
- **Business Context Integration**: Ability to add business context to interpretations
- **Alternative Analysis**: Options to request different analytical approaches

#### 3. Cost Approval and Monitoring

**Implementation**: Real-time cost tracking with user-defined thresholds and approval gates.

**Cost Control Features:**
- **Pre-execution Cost Estimates**: Estimated costs before query processing
- **Real-time Monitoring**: Live cost updates during processing
- **Threshold Alerts**: Warnings when approaching cost limits
- **Budget Controls**: Ability to set session or daily spending limits

### User Experience Design

The HITL interface is designed to be intuitive and non-intrusive, providing necessary oversight without impeding workflow efficiency.

**Design Principles:**
1. **Transparency**: All AI decisions and reasoning are clearly explained
2. **Control**: Users maintain ultimate control over all operations
3. **Efficiency**: Approval processes are streamlined for common scenarios
4. **Education**: Interface helps users understand AI capabilities and limitations
5. **Trust**: Consistent and reliable behavior builds user confidence

### Feedback Integration

User feedback is systematically collected and used to improve system performance:

**Feedback Mechanisms:**
- **Query Quality Ratings**: Users rate the accuracy of generated SQL queries
- **Insight Validation**: Users confirm or correct AI-generated insights
- **Performance Feedback**: Users provide feedback on system speed and reliability
- **Feature Requests**: Structured collection of enhancement requests

**Feedback Utilization:**
- **Prompt Optimization**: User feedback informs prompt engineering improvements
- **Model Fine-tuning**: Systematic feedback collection for model improvement
- **Feature Development**: User requests drive new feature development
- **Quality Assurance**: Feedback helps identify and resolve system issues

## Cost Tracking and Control

### Economic Model and Pricing Structure

Understanding and controlling LLM usage costs is crucial for production deployments. The cost tracking system provides comprehensive monitoring and control mechanisms to ensure predictable and manageable expenses.

### Token-Level Cost Calculation

**Pricing Model**: The system uses OpenAI's token-based pricing model with different rates for prompt and completion tokens.

**Current Pricing Structure** (as of 2024):
- **GPT-4.1-Mini**: $0.15/$0.60 per 1M prompt/completion tokens
- **GPT-4.1**: $5.00/$15.00 per 1M prompt/completion tokens
- **GPT-3.5-Turbo**: $0.50/$1.50 per 1M prompt/completion tokens

**Token Counting Methodology**:
The system uses the `tiktoken` library for accurate token counting, which matches OpenAI's internal counting mechanisms:

```python
def count_tokens(self, text: str, model: str = "gpt-4.1-mini") -> int:
    encoding_map = {
        'gpt-4.1-mini': 'o200k_base',
        'gpt-4.1': 'o200k_base',
        'gpt-3.5-turbo': 'cl100k_base'
    }
    encoding = tiktoken.get_encoding(encoding_map.get(model, 'cl100k_base'))
    return len(encoding.encode(text))
```

### Session-Based Cost Tracking

**Session Management**: Each user interaction creates a unique session with isolated cost tracking.

**Session Lifecycle**:
1. **Session Initialization**: Unique session ID generation and cost tracker setup
2. **Usage Recording**: Real-time tracking of all LLM API calls
3. **Cost Calculation**: Immediate cost calculation for each API call
4. **Session Summary**: Comprehensive cost breakdown upon completion
5. **Session Archival**: Historical cost data storage for analysis

**Cost Attribution**: Costs are attributed to specific agents and tasks, enabling detailed analysis of system efficiency:

```python
class LLMUsage:
    timestamp: datetime
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost: float
    agent_name: str
    task_name: str
    duration: float
```

### Cost Optimization Strategies

#### 1. Model Selection Optimization

**Strategy**: Use the most cost-effective model for each task type.

**Implementation**:
- **Routine Tasks**: Use GPT-4.1-Mini for schema analysis and basic operations
- **Complex Reasoning**: Use GPT-4.1 for complex SQL generation and analysis
- **Simple Operations**: Use GPT-3.5-Turbo for straightforward tasks

#### 2. Prompt Engineering for Efficiency

**Strategy**: Optimize prompts to minimize token usage while maintaining accuracy.

**Techniques**:
- **Concise Instructions**: Clear, brief instructions that minimize prompt length
- **Context Optimization**: Include only necessary context information
- **Template Reuse**: Standardized prompt templates to reduce redundancy
- **Response Formatting**: Structured response formats to minimize completion tokens

#### 3. Caching and Memoization

**Strategy**: Cache results for repeated queries to avoid redundant API calls.

**Implementation Areas**:
- **Schema Caching**: Cache database schema information for reuse
- **Query Pattern Caching**: Store results for common query patterns
- **Analysis Caching**: Reuse statistical analysis for similar datasets
- **Insight Caching**: Cache generated insights for similar data patterns

#### 4. Intelligent Batching

**Strategy**: Batch multiple operations into single API calls where possible.

**Batching Opportunities**:
- **Multi-query Analysis**: Analyze multiple queries in a single call
- **Batch Validation**: Validate multiple queries simultaneously
- **Aggregate Insights**: Generate insights for multiple results together

### Cost Monitoring and Alerting

**Real-time Monitoring**: The system provides live cost updates during processing, allowing users to monitor expenses as they occur.

**Alert Mechanisms**:
- **Threshold Warnings**: Alerts when session costs approach predefined limits
- **Budget Notifications**: Daily or monthly budget consumption notifications
- **Anomaly Detection**: Alerts for unusual cost patterns or spikes
- **Cost Projection**: Estimates of total session costs based on current usage

**Reporting and Analytics**:
- **Usage Trends**: Historical analysis of cost patterns and trends
- **Agent Efficiency**: Comparative analysis of cost per agent
- **ROI Analysis**: Cost-benefit analysis of system usage
- **Optimization Recommendations**: Suggestions for cost reduction

## Security and Safety Measures

### Multi-Layer Security Architecture

The security framework implements defense-in-depth principles with multiple layers of protection to ensure safe operation in production environments.

### SQL Injection Prevention

#### 1. Whitelist Approach

**Implementation**: Only SELECT statements are permitted, with all other SQL operations explicitly blocked.

**Validation Process**:
```python
def _is_safe_query(self, query: str) -> bool:
    query_upper = query.upper().strip()
    if not query_upper.startswith('SELECT'):
        return False
    
    dangerous_keywords = [
        'DROP', 'DELETE', 'UPDATE', 'INSERT', 'ALTER', 'CREATE',
        'TRUNCATE', 'REPLACE', 'PRAGMA', 'ATTACH', 'DETACH'
    ]
    
    for keyword in dangerous_keywords:
        if keyword in query_upper:
            return False
    return True
```

#### 2. Pattern Detection

**Advanced Pattern Recognition**: The system uses regular expressions and heuristic analysis to detect potential injection attempts:

**Detected Patterns**:
- Comment injection (`'; --`)
- Union-based injection (`UNION SELECT`)
- Boolean-based injection (`OR 1=1`)
- Time-based injection (`WAITFOR DELAY`)
- Stacked queries (`; DROP TABLE`)

#### 3. Parameterized Query Support

**Implementation**: While the current system focuses on SELECT operations, parameterized query support is implemented for future extensibility:

```python
def execute_parameterized_query(self, query: str, parameters: Dict[str, Any]) -> str:
    # Parameterized execution for enhanced security
    cursor.execute(query, parameters)
```

### Data Privacy and Isolation

#### 1. Session Isolation

**Implementation**: Each user session operates in complete isolation with no data sharing between sessions.

**Isolation Mechanisms**:
- **Temporary Databases**: Each session uses a unique temporary SQLite database
- **Memory Isolation**: No shared memory structures between sessions
- **Process Isolation**: Separate process contexts for concurrent sessions
- **Cleanup Procedures**: Automatic cleanup of all session data upon completion

#### 2. Data Minimization

**Principle**: Only necessary data is processed and retained, with automatic cleanup of temporary artifacts.

**Data Handling Policies**:
- **Temporary Storage**: All uploaded data is stored in temporary files with automatic cleanup
- **Memory Management**: Efficient memory usage with immediate cleanup of large objects
- **Log Sanitization**: Sensitive data is excluded from system logs
- **Retention Policies**: No long-term storage of user data without explicit consent

#### 3. Encryption and Secure Communication

**Implementation**: All data transmission and storage uses appropriate encryption mechanisms.

**Security Measures**:
- **HTTPS Enforcement**: All web communication uses HTTPS encryption
- **API Security**: Secure API key management and transmission
- **Local Encryption**: Temporary files are encrypted when stored locally
- **Memory Protection**: Sensitive data in memory is properly protected

### Access Control and Authentication

#### 1. API Key Management

**Secure Storage**: API keys are stored securely using environment variables and secure configuration management.

**Best Practices**:
- **Environment Variables**: API keys stored in environment variables, not code
- **Key Rotation**: Support for regular API key rotation
- **Access Logging**: Comprehensive logging of API key usage
- **Scope Limitation**: API keys with minimal necessary permissions

#### 2. User Session Management

**Session Security**: Robust session management with appropriate security controls.

**Security Features**:
- **Session Tokens**: Secure session token generation and management
- **Timeout Controls**: Automatic session timeout for inactive sessions
- **Concurrent Session Limits**: Controls on number of concurrent sessions per user
- **Session Invalidation**: Secure session cleanup and invalidation

### Audit Logging and Compliance

#### 1. Comprehensive Audit Trail

**Implementation**: All system operations are logged for security monitoring and compliance purposes.

**Logged Events**:
- **User Actions**: All user interactions and decisions
- **System Operations**: Database queries, API calls, and system events
- **Security Events**: Authentication attempts, access violations, and security alerts
- **Performance Metrics**: System performance and resource usage data

#### 2. Compliance Framework

**Standards Adherence**: The system is designed to support various compliance requirements.

**Compliance Features**:
- **Data Governance**: Clear data handling and retention policies
- **Access Controls**: Role-based access control implementation
- **Audit Capabilities**: Comprehensive audit trail and reporting
- **Privacy Controls**: Data privacy and protection mechanisms

### Incident Response and Recovery

#### 1. Threat Detection

**Monitoring Systems**: Continuous monitoring for security threats and anomalies.

**Detection Capabilities**:
- **Anomaly Detection**: Unusual usage patterns and behaviors
- **Threat Intelligence**: Integration with security threat feeds
- **Real-time Monitoring**: Live monitoring of system security status
- **Automated Alerts**: Immediate notification of security events

#### 2. Response Procedures

**Incident Response**: Structured procedures for handling security incidents.

**Response Framework**:
- **Immediate Containment**: Rapid isolation of affected systems
- **Investigation Procedures**: Systematic investigation of security incidents
- **Recovery Planning**: Structured recovery and restoration procedures
- **Post-Incident Analysis**: Comprehensive analysis and improvement planning

## Database Tools and Integration

### Custom Tool Architecture

The database tools represent a sophisticated abstraction layer that provides CrewAI agents with safe, controlled, and efficient access to database operations. Each tool is designed as a specialized component that encapsulates specific database functionality while maintaining security and performance standards.

### Tool Implementation Framework

#### 1. Base Tool Structure

All database tools inherit from CrewAI's `BaseTool` class and implement a consistent interface:

```python
class SQLiteQueryTool(BaseTool):
    name: str = "SQLite Query Executor"
    description: str = "Execute SQL queries against a SQLite database..."
    
    def __init__(self, database_path: str, **kwargs):
        super().__init__(**kwargs)
        self._database_path = database_path
        self._max_rows = 1000
        self._timeout = 30
    
    def _run(self, sql_query: str) -> str:
        # Implementation details
```

#### 2. Error Handling and Resilience

**Comprehensive Error Management**: Each tool implements robust error handling to ensure system stability:

**Error Categories**:
- **Connection Errors**: Database connection failures and timeouts
- **Syntax Errors**: SQL syntax validation and error reporting
- **Runtime Errors**: Query execution failures and resource constraints
- **Security Violations**: Unsafe query attempts and policy violations
- **Resource Exhaustion**: Memory and timeout limit violations

**Recovery Strategies**:
- **Graceful Degradation**: Partial results when complete execution fails
- **Automatic Retry**: Intelligent retry mechanisms for transient failures
- **Alternative Approaches**: Fallback methods for failed operations
- **User Notification**: Clear error messages and resolution suggestions

### Individual Tool Specifications

#### 1. SQLiteQueryTool

**Purpose**: Safe execution of SQL queries with comprehensive monitoring and protection.

**Key Features**:
- **Query Validation**: Multi-layer validation before execution
- **Timeout Protection**: Configurable timeout limits to prevent hanging queries
- **Result Limiting**: Automatic LIMIT clause injection for large result sets
- **Performance Monitoring**: Execution time and resource usage tracking
- **Error Recovery**: Graceful handling of execution errors

**Implementation Details**:
```python
def _run(self, sql_query: str) -> str:
    if not self._is_safe_query(sql_query):
        return json.dumps({"error": "Unsafe query detected"})
    
    start_time = time.time()
    conn = sqlite3.connect(self._database_path, timeout=self._timeout)
    
    try:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        limited_query = self._add_limit_if_needed(sql_query)
        cursor.execute(limited_query)
        rows = cursor.fetchall()
        
        execution_time = time.time() - start_time
        results = [dict(zip([desc[0] for desc in cursor.description], row)) 
                  for row in rows]
        
        return json.dumps({
            "success": True,
            "results": results,
            "execution_time": execution_time,
            "row_count": len(results)
        })
    finally:
        conn.close()
```

#### 2. SchemaInspectorTool

**Purpose**: Comprehensive database schema analysis and documentation.

**Analysis Capabilities**:
- **Table Structure**: Complete table and column information
- **Data Types**: Detailed data type analysis and validation
- **Constraints**: Primary keys, foreign keys, and other constraints
- **Statistics**: Row counts, null percentages, and unique value counts
- **Relationships**: Potential relationship identification between tables

**Schema Documentation Format**:
```python
schema_info = {
    'table_name': 'employees',
    'row_count': 1000,
    'columns': [
        {
            'name': 'employee_id',
            'type': 'INTEGER',
            'unique_values': 1000,
            'null_count': 0,
            'null_percentage': 0.0
        }
    ],
    'sample_data': [...]
}
```

#### 3. DataAnalysisTool

**Purpose**: Statistical analysis and insight generation from query results.

**Analysis Categories**:

**Descriptive Statistics**:
- Central tendency measures (mean, median, mode)
- Variability measures (standard deviation, variance, range)
- Distribution characteristics (skewness, kurtosis)
- Percentile analysis and quartile calculations

**Data Quality Assessment**:
- Missing value analysis and patterns
- Duplicate record identification
- Outlier detection using statistical methods
- Data consistency validation

**Relationship Analysis**:
- Correlation analysis between numeric variables
- Association analysis for categorical variables
- Trend detection in time series data
- Pattern recognition in data distributions

**Visualization Recommendations**:
```python
viz_recommendations = [
    "Scatter plot for correlation analysis between numeric variables",
    "Histogram for distribution analysis of continuous variables",
    "Bar chart for categorical variable frequency analysis",
    "Box plot for outlier detection and distribution comparison"
]
```

#### 4. QueryValidatorTool

**Purpose**: Comprehensive validation of SQL queries before execution.

**Validation Framework**:

**Security Validation**:
- Operation type verification (SELECT only)
- Dangerous keyword detection and blocking
- SQL injection pattern recognition
- Parameter validation and sanitization

**Syntax Validation**:
- SQL syntax verification using EXPLAIN QUERY PLAN
- Column and table name validation against schema
- Function and operator compatibility checking
- Query structure and logic validation

**Performance Assessment**:
- Query complexity analysis and scoring
- Resource usage estimation
- Execution time prediction
- Optimization recommendation generation

**Validation Result Format**:
```python
validation_result = {
    "query": "SELECT * FROM employees WHERE age > 30",
    "is_safe": True,
    "is_valid_syntax": True,
    "estimated_complexity": "moderate",
    "issues": [],
    "recommendations": ["Consider adding LIMIT clause"]
}
```

### Tool Integration and Orchestration

#### 1. Agent-Tool Binding

**Dynamic Tool Assignment**: Tools are dynamically assigned to agents based on their roles and responsibilities:

```python
def _create_agents(self) -> Dict[str, Agent]:
    agents = {}
    for agent_name, config in self.agents_config.items():
        tools = []
        if agent_name in ['data_execution_agent', 'sql_generator_agent']:
            tools = self.database_tools
        
        agent = Agent(
            role=config['role'],
            goal=config['goal'],
            tools=tools,
            # ... other configuration
        )
        agents[agent_name] = agent
    return agents
```

#### 2. Context Sharing

**Tool Context Management**: Tools share context and state information to enable coordinated operations:

**Shared Context Elements**:
- Database schema information
- Previous query results
- Performance metrics
- Security validation results
- User preferences and settings

#### 3. Performance Optimization

**Tool Performance Strategies**:

**Connection Pooling**: Efficient database connection management to reduce overhead:
```python
class ConnectionPool:
    def __init__(self, database_path: str, max_connections: int = 10):
        self.database_path = database_path
        self.max_connections = max_connections
        self.connections = queue.Queue(maxsize=max_connections)
```

**Query Caching**: Intelligent caching of query results and metadata:
- **Result Caching**: Cache query results for identical queries
- **Schema Caching**: Cache database schema information
- **Validation Caching**: Cache validation results for similar queries

**Resource Management**: Efficient resource allocation and cleanup:
- **Memory Management**: Proper cleanup of large result sets
- **Connection Management**: Automatic connection cleanup and pooling
- **Temporary File Management**: Systematic cleanup of temporary files

## Performance Optimization

### System Performance Architecture

Performance optimization in the CrewAI Tabular Data Agent focuses on multiple dimensions: query execution speed, memory efficiency, cost optimization, and user experience responsiveness. The system implements various optimization strategies at different architectural layers.

### Database Performance Optimization

#### 1. Query Optimization Strategies

**Automatic Query Enhancement**: The system automatically optimizes user queries for better performance:

**LIMIT Clause Injection**:
```python
def _add_limit_if_needed(self, query: str) -> str:
    query_upper = query.upper()
    if 'LIMIT' not in query_upper:
        return f"{query.rstrip(';')} LIMIT {self._max_rows}"
    return query
```

**Index Recommendations**: The system analyzes query patterns and suggests appropriate indexes:
- **Frequent Column Analysis**: Identifies commonly queried columns
- **Join Optimization**: Recommends indexes for frequently joined tables
- **Filter Optimization**: Suggests indexes for commonly filtered columns

**Query Rewriting**: Automatic rewriting of inefficient query patterns:
- **Subquery Optimization**: Converting correlated subqueries to joins
- **Aggregation Optimization**: Optimizing GROUP BY and aggregate functions
- **Predicate Pushdown**: Moving filter conditions closer to data sources

#### 2. Memory Management

**Efficient Data Handling**: Strategies to minimize memory usage while processing large datasets:

**Streaming Processing**: For large result sets, implement streaming to avoid memory exhaustion:
```python
def stream_query_results(self, query: str, chunk_size: int = 1000):
    conn = sqlite3.connect(self.database_path)
    cursor = conn.cursor()
    cursor.execute(query)
    
    while True:
        rows = cursor.fetchmany(chunk_size)
        if not rows:
            break
        yield rows
```

**Memory Pool Management**: Efficient allocation and deallocation of memory resources:
- **Object Pooling**: Reuse of expensive objects like database connections
- **Garbage Collection Optimization**: Strategic garbage collection timing
- **Memory Monitoring**: Real-time monitoring of memory usage patterns

#### 3. Connection Optimization

**Connection Pooling**: Efficient management of database connections to reduce overhead:

```python
class DatabaseConnectionPool:
    def __init__(self, database_path: str, pool_size: int = 5):
        self.database_path = database_path
        self.pool_size = pool_size
        self.connections = queue.Queue(maxsize=pool_size)
        self._initialize_pool()
    
    def get_connection(self):
        try:
            return self.connections.get_nowait()
        except queue.Empty:
            return sqlite3.connect(self.database_path)
    
    def return_connection(self, conn):
        try:
            self.connections.put_nowait(conn)
        except queue.Full:
            conn.close()
```

### Agent Performance Optimization

#### 1. Prompt Engineering for Efficiency

**Token Optimization**: Strategies to minimize token usage while maintaining accuracy:

**Context Compression**: Intelligent compression of context information:
```python
def compress_schema_context(self, schema_info: Dict) -> str:
    # Extract only essential schema information
    essential_info = {
        'tables': [table['name'] for table in schema_info['tables']],
        'key_columns': self._identify_key_columns(schema_info),
        'data_types': self._summarize_data_types(schema_info)
    }
    return json.dumps(essential_info, separators=(',', ':'))
```

**Template Optimization**: Standardized prompt templates to reduce redundancy:
- **Modular Prompts**: Reusable prompt components for common operations
- **Dynamic Context**: Context inclusion based on query complexity
- **Response Formatting**: Structured response formats to minimize completion tokens

#### 2. Caching Strategies

**Multi-Level Caching**: Comprehensive caching strategy across different system components:

**Schema Caching**: Cache database schema information to avoid repeated analysis:
```python
class SchemaCache:
    def __init__(self, ttl: int = 3600):  # 1 hour TTL
        self.cache = {}
        self.ttl = ttl
    
    def get_schema(self, database_path: str) -> Optional[Dict]:
        cache_key = f"schema_{hash(database_path)}"
        if cache_key in self.cache:
            entry = self.cache[cache_key]
            if time.time() - entry['timestamp'] < self.ttl:
                return entry['data']
        return None
```

**Query Result Caching**: Cache results for identical or similar queries:
- **Query Fingerprinting**: Generate unique fingerprints for queries
- **Result Similarity**: Identify similar queries that can share cached results
- **Cache Invalidation**: Intelligent cache invalidation strategies

**Analysis Caching**: Cache statistical analysis results for reuse:
- **Statistical Summaries**: Cache descriptive statistics for datasets
- **Visualization Metadata**: Cache chart configuration and data
- **Insight Templates**: Cache generated insights for similar data patterns

#### 3. Parallel Processing

**Concurrent Agent Execution**: Where possible, execute independent agent tasks in parallel:

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

async def parallel_agent_execution(self, tasks: List[Task]):
    with ThreadPoolExecutor(max_workers=3) as executor:
        loop = asyncio.get_event_loop()
        futures = [
            loop.run_in_executor(executor, self.execute_task, task)
            for task in tasks if task.can_run_parallel
        ]
        results = await asyncio.gather(*futures)
    return results
```

**Asynchronous Operations**: Non-blocking operations for improved responsiveness:
- **Background Processing**: Long-running tasks executed in background
- **Progressive Results**: Streaming results as they become available
- **User Interface Responsiveness**: Non-blocking UI updates during processing

### Cost Performance Optimization

#### 1. Model Selection Optimization

**Dynamic Model Selection**: Choose the most cost-effective model for each task:

```python
def select_optimal_model(self, task_complexity: str, accuracy_requirement: str) -> str:
    model_matrix = {
        ('simple', 'standard'): 'gpt-3.5-turbo',
        ('simple', 'high'): 'gpt-4.1-mini',
        ('moderate', 'standard'): 'gpt-4.1-mini',
        ('moderate', 'high'): 'gpt-4.1',
        ('complex', 'standard'): 'gpt-4.1-mini',
        ('complex', 'high'): 'gpt-4.1'
    }
    return model_matrix.get((task_complexity, accuracy_requirement), 'gpt-4.1-mini')
```

#### 2. Intelligent Batching

**Request Batching**: Combine multiple operations into single API calls:

**Multi-Query Analysis**: Analyze multiple queries in a single request:
```python
def batch_query_analysis(self, queries: List[str]) -> Dict[str, Any]:
    batch_prompt = f"""
    Analyze the following SQL queries for safety and efficiency:
    
    {chr(10).join(f"{i+1}. {query}" for i, query in enumerate(queries))}
    
    Provide analysis for each query in the format:
    Query 1: [analysis]
    Query 2: [analysis]
    ...
    """
    return self.llm_client.complete(batch_prompt)
```

#### 3. Response Optimization

**Structured Responses**: Use structured response formats to minimize completion tokens:

```python
response_template = {
    "sql_query": "SELECT ...",
    "explanation": "Brief explanation",
    "confidence": 0.95,
    "warnings": []
}
```

### User Experience Performance

#### 1. Progressive Loading

**Incremental Results**: Display results progressively as they become available:

```python
def progressive_query_processing(self, user_query: str):
    # Stage 1: Schema analysis
    yield {"stage": "schema", "status": "processing", "progress": 20}
    schema_result = self.analyze_schema()
    yield {"stage": "schema", "status": "complete", "progress": 20, "result": schema_result}
    
    # Stage 2: SQL generation
    yield {"stage": "sql_generation", "status": "processing", "progress": 40}
    sql_result = self.generate_sql(user_query, schema_result)
    yield {"stage": "sql_generation", "status": "complete", "progress": 40, "result": sql_result}
    
    # Continue for remaining stages...
```

#### 2. Predictive Preloading

**Anticipatory Loading**: Preload likely next steps based on user behavior:

**Schema Preloading**: Load schema information immediately after dataset upload:
- **Background Analysis**: Perform schema analysis while user reviews upload
- **Predictive Caching**: Cache likely query patterns based on schema
- **Resource Preallocation**: Prepare resources for expected operations

#### 3. Response Time Optimization

**Target Response Times**:
- **Schema Analysis**: < 5 seconds
- **SQL Generation**: < 10 seconds
- **Query Execution**: < 15 seconds
- **Insight Generation**: < 20 seconds
- **Total Pipeline**: < 60 seconds

**Performance Monitoring**: Continuous monitoring of response times with alerting:

```python
class PerformanceMonitor:
    def __init__(self):
        self.metrics = defaultdict(list)
    
    def record_timing(self, operation: str, duration: float):
        self.metrics[operation].append(duration)
        if duration > self.get_threshold(operation):
            self.alert_slow_operation(operation, duration)
    
    def get_average_timing(self, operation: str) -> float:
        return sum(self.metrics[operation]) / len(self.metrics[operation])
```

## Deployment and Scaling

### Production Deployment Architecture

The CrewAI Tabular Data Agent is designed for production deployment with considerations for scalability, reliability, and maintainability. The deployment architecture supports both single-instance and distributed deployments depending on usage requirements.

### Deployment Options

#### 1. Single-Instance Deployment

**Use Case**: Small to medium organizations with moderate usage patterns.

**Architecture Components**:
- **Application Server**: Single Streamlit application instance
- **Database**: Local SQLite for temporary data processing
- **File Storage**: Local file system for temporary file handling
- **Monitoring**: Basic logging and health checks

**Deployment Configuration**:
```yaml
# docker-compose.yml
version: '3.8'
services:
  crewai-agent:
    build: .
    ports:
      - "8501:8501"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - ENVIRONMENT=production
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
    restart: unless-stopped
```

**Resource Requirements**:
- **CPU**: 2-4 cores
- **Memory**: 4-8 GB RAM
- **Storage**: 20-50 GB SSD
- **Network**: 100 Mbps bandwidth

#### 2. Distributed Deployment

**Use Case**: Large organizations with high concurrency and availability requirements.

**Architecture Components**:
- **Load Balancer**: Nginx or HAProxy for request distribution
- **Application Cluster**: Multiple Streamlit instances behind load balancer
- **Shared Storage**: Network-attached storage for temporary files
- **Database Cluster**: Distributed database for session management
- **Monitoring Stack**: Comprehensive monitoring and alerting

**Kubernetes Deployment**:
```yaml
# kubernetes-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: crewai-agent
spec:
  replicas: 3
  selector:
    matchLabels:
      app: crewai-agent
  template:
    metadata:
      labels:
        app: crewai-agent
    spec:
      containers:
      - name: crewai-agent
        image: crewai-agent:latest
        ports:
        - containerPort: 8501
        env:
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: openai-secret
              key: api-key
        resources:
          requests:
            memory: "2Gi"
            cpu: "1"
          limits:
            memory: "4Gi"
            cpu: "2"
```

### Scalability Considerations

#### 1. Horizontal Scaling

**Stateless Design**: The application is designed to be stateless, enabling easy horizontal scaling:

**Session Management**: External session storage for multi-instance deployments:
```python
class ExternalSessionStore:
    def __init__(self, redis_url: str):
        self.redis_client = redis.from_url(redis_url)
    
    def store_session(self, session_id: str, session_data: Dict):
        self.redis_client.setex(
            f"session:{session_id}",
            3600,  # 1 hour TTL
            json.dumps(session_data)
        )
    
    def get_session(self, session_id: str) -> Optional[Dict]:
        data = self.redis_client.get(f"session:{session_id}")
        return json.loads(data) if data else None
```

**Load Balancing Strategies**:
- **Round Robin**: Simple distribution across instances
- **Least Connections**: Route to instance with fewest active connections
- **Resource-Based**: Route based on instance resource utilization
- **Session Affinity**: Maintain session stickiness when required

#### 2. Vertical Scaling

**Resource Optimization**: Efficient use of available resources on single instances:

**Memory Scaling**: Dynamic memory allocation based on workload:
```python
class DynamicResourceManager:
    def __init__(self):
        self.memory_threshold = 0.8  # 80% memory usage threshold
        self.cpu_threshold = 0.7     # 70% CPU usage threshold
    
    def check_resource_usage(self):
        memory_usage = psutil.virtual_memory().percent / 100
        cpu_usage = psutil.cpu_percent() / 100
        
        if memory_usage > self.memory_threshold:
            self.trigger_memory_cleanup()
        
        if cpu_usage > self.cpu_threshold:
            self.throttle_requests()
```

#### 3. Auto-Scaling

**Dynamic Scaling**: Automatic scaling based on demand metrics:

**Metrics-Based Scaling**:
- **Request Rate**: Scale based on incoming request volume
- **Response Time**: Scale when response times exceed thresholds
- **Resource Utilization**: Scale based on CPU and memory usage
- **Queue Length**: Scale based on request queue depth

**Auto-Scaling Configuration**:
```yaml
# kubernetes-hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: crewai-agent-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: crewai-agent
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

### High Availability and Reliability

#### 1. Fault Tolerance

**Redundancy**: Multiple layers of redundancy to ensure system availability:

**Application Redundancy**:
- **Multiple Instances**: Run multiple application instances
- **Health Checks**: Continuous health monitoring and automatic failover
- **Circuit Breakers**: Prevent cascade failures with circuit breaker patterns
- **Graceful Degradation**: Maintain partial functionality during failures

**Data Redundancy**:
- **Backup Strategies**: Regular backups of configuration and session data
- **Replication**: Data replication across multiple storage systems
- **Recovery Procedures**: Automated recovery from data corruption or loss

#### 2. Monitoring and Alerting

**Comprehensive Monitoring**: Multi-dimensional monitoring for proactive issue detection:

**Application Metrics**:
```python
class ApplicationMetrics:
    def __init__(self):
        self.request_count = Counter('requests_total', 'Total requests')
        self.request_duration = Histogram('request_duration_seconds', 'Request duration')
        self.error_count = Counter('errors_total', 'Total errors')
        self.active_sessions = Gauge('active_sessions', 'Active sessions')
    
    def record_request(self, duration: float, status: str):
        self.request_count.inc()
        self.request_duration.observe(duration)
        if status == 'error':
            self.error_count.inc()
```

**Infrastructure Metrics**:
- **System Resources**: CPU, memory, disk, and network utilization
- **Application Performance**: Response times, throughput, and error rates
- **External Dependencies**: API response times and availability
- **Business Metrics**: User engagement and system usage patterns

**Alerting Rules**:
```yaml
# prometheus-alerts.yaml
groups:
- name: crewai-agent
  rules:
  - alert: HighErrorRate
    expr: rate(errors_total[5m]) > 0.1
    for: 2m
    annotations:
      summary: "High error rate detected"
  
  - alert: HighResponseTime
    expr: histogram_quantile(0.95, request_duration_seconds) > 30
    for: 5m
    annotations:
      summary: "High response time detected"
```

#### 3. Disaster Recovery

**Recovery Planning**: Comprehensive disaster recovery procedures:

**Backup Strategies**:
- **Configuration Backups**: Regular backups of system configuration
- **Data Backups**: Automated backups of session and user data
- **Code Backups**: Version control and deployment artifact storage
- **Infrastructure Backups**: Infrastructure as code and configuration management

**Recovery Procedures**:
- **Automated Recovery**: Automated recovery procedures for common failures
- **Manual Recovery**: Documented procedures for complex recovery scenarios
- **Testing**: Regular disaster recovery testing and validation
- **Documentation**: Comprehensive recovery documentation and runbooks

### Security in Production

#### 1. Network Security

**Network Isolation**: Proper network segmentation and access controls:

**Firewall Configuration**:
```bash
# iptables rules for production deployment
iptables -A INPUT -p tcp --dport 8501 -s 10.0.0.0/8 -j ACCEPT
iptables -A INPUT -p tcp --dport 8501 -j DROP
iptables -A OUTPUT -p tcp --dport 443 -d api.openai.com -j ACCEPT
```

**TLS/SSL Configuration**:
- **Certificate Management**: Automated certificate provisioning and renewal
- **Strong Ciphers**: Use of strong encryption ciphers and protocols
- **HSTS**: HTTP Strict Transport Security implementation
- **Certificate Pinning**: Certificate pinning for critical connections

#### 2. Application Security

**Security Hardening**: Production security hardening measures:

**Environment Isolation**:
```python
class ProductionSecurityConfig:
    def __init__(self):
        self.debug_mode = False
        self.log_level = 'WARNING'
        self.api_rate_limits = {
            'requests_per_minute': 60,
            'requests_per_hour': 1000
        }
        self.session_timeout = 3600  # 1 hour
        self.max_file_size = 100 * 1024 * 1024  # 100MB
```

**Access Controls**:
- **Authentication**: Strong authentication mechanisms
- **Authorization**: Role-based access control implementation
- **API Security**: API key management and rate limiting
- **Audit Logging**: Comprehensive security audit logging

#### 3. Compliance and Governance

**Regulatory Compliance**: Support for various compliance requirements:

**Data Protection**:
- **GDPR Compliance**: Data protection and privacy controls
- **HIPAA Compliance**: Healthcare data protection measures
- **SOC 2**: Security and availability controls
- **ISO 27001**: Information security management

**Governance Framework**:
- **Policy Management**: Security policy implementation and enforcement
- **Risk Assessment**: Regular security risk assessments
- **Incident Response**: Security incident response procedures
- **Compliance Monitoring**: Continuous compliance monitoring and reporting

## API Reference

### Core Classes and Methods

#### CrewAITabularAgent

**Primary orchestration class for the multi-agent system.**

```python
class CrewAITabularAgent:
    def __init__(self, config_dir: str = "config")
    def load_dataset(self, file_path: str) -> Dict[str, Any]
    def process_query(self, user_query: str, session_id: Optional[str] = None) -> Dict[str, Any]
    def get_cost_summary(self, session_id: Optional[str] = None) -> str
    def cleanup(self)
```

**Methods:**

**`__init__(config_dir: str = "config")`**
- **Purpose**: Initialize the CrewAI orchestrator with configuration
- **Parameters**: 
  - `config_dir`: Directory containing agent and task configurations
- **Returns**: None
- **Raises**: `FileNotFoundError` if configuration files are missing

**`load_dataset(file_path: str) -> Dict[str, Any]`**
- **Purpose**: Load and process a dataset for analysis
- **Parameters**:
  - `file_path`: Path to the dataset file (CSV, Excel, JSON)
- **Returns**: Dictionary containing load status and metadata
- **Example**:
```python
result = orchestrator.load_dataset("data/sales.csv")
if result['success']:
    print(f"Loaded {result['metadata']['rows']} rows")
```

**`process_query(user_query: str, session_id: Optional[str] = None) -> Dict[str, Any]`**
- **Purpose**: Process a natural language query through the agent pipeline
- **Parameters**:
  - `user_query`: Natural language question about the data
  - `session_id`: Optional session identifier for cost tracking
- **Returns**: Dictionary containing query results and execution metadata
- **Example**:
```python
result = orchestrator.process_query("What is the average sales by region?")
if result['success']:
    print(result['result'])
```

#### DataLoader

**Handles dataset loading and SQLite database creation.**

```python
class DataLoader:
    def __init__(self)
    def load_file(self, file_path: str) -> Tuple[pd.DataFrame, Dict[str, Any]]
    def create_sqlite_database(self, df: pd.DataFrame, table_name: str = 'data_table') -> str
    def get_schema_description(self) -> str
    def cleanup(self)
```

**Methods:**

**`load_file(file_path: str) -> Tuple[pd.DataFrame, Dict[str, Any]]`**
- **Purpose**: Load data from various file formats
- **Parameters**:
  - `file_path`: Path to the data file
- **Returns**: Tuple of (DataFrame, metadata dictionary)
- **Supported Formats**: CSV, Excel (XLSX/XLS), JSON
- **Example**:
```python
loader = DataLoader()
df, metadata = loader.load_file("data.csv")
print(f"Loaded {metadata['rows']} rows, {metadata['columns']} columns")
```

#### CostTracker

**Monitors and tracks LLM usage costs.**

```python
class CostTracker:
    def __init__(self)
    def start_session(self, session_id: str) -> str
    def end_session(self, session_id: Optional[str] = None) -> SessionCosts
    def track_usage(self, prompt: str, completion: str, model: str, agent_name: str, task_name: str) -> LLMUsage
    def get_session_summary(self, session_id: Optional[str] = None) -> Dict[str, Any]
    def format_cost_summary(self, session_id: Optional[str] = None) -> str
```

**Methods:**

**`start_session(session_id: str) -> str`**
- **Purpose**: Start a new cost tracking session
- **Parameters**:
  - `session_id`: Unique identifier for the session
- **Returns**: Session ID
- **Example**:
```python
tracker = CostTracker()
session_id = tracker.start_session("user_session_123")
```

**`track_usage(...) -> LLMUsage`**
- **Purpose**: Record LLM usage for cost calculation
- **Parameters**:
  - `prompt`: Input prompt text
  - `completion`: Generated completion text
  - `model`: Model name used
  - `agent_name`: Name of the agent making the call
  - `task_name`: Name of the task being performed
- **Returns**: LLMUsage object with cost details

#### Database Tools

**Custom tools for database interaction within CrewAI.**

**SQLiteQueryTool**
```python
class SQLiteQueryTool(BaseTool):
    name: str = "SQLite Query Executor"
    description: str = "Execute SQL queries against a SQLite database..."
    
    def __init__(self, database_path: str, **kwargs)
    def _run(self, sql_query: str) -> str
```

**SchemaInspectorTool**
```python
class SchemaInspectorTool(BaseTool):
    name: str = "Database Schema Inspector"
    description: str = "Inspect the structure of a SQLite database..."
    
    def __init__(self, database_path: str, **kwargs)
    def _run(self, table_name: str = "") -> str
```

### Configuration Reference

#### Agent Configuration (agents.yaml)

```yaml
agent_name:
  role: "Agent Role Description"
  goal: "Agent Goal Statement"
  backstory: "Agent Background and Expertise"
  allow_delegation: false
  verbose: true
  model: "gpt-4.1-mini"
  temperature: 0.1
```

**Configuration Parameters:**
- **role**: Brief description of the agent's role
- **goal**: Specific goal the agent should achieve
- **backstory**: Detailed background providing context
- **allow_delegation**: Whether agent can delegate tasks
- **verbose**: Enable detailed logging
- **model**: LLM model to use
- **temperature**: Creativity/randomness parameter (0.0-1.0)

#### Task Configuration (tasks.yaml)

```yaml
task_name:
  description: "Task description with {placeholders}"
  expected_output: "Expected output format"
  agent: "agent_name"
  human_input: false
```

**Configuration Parameters:**
- **description**: Detailed task description with context placeholders
- **expected_output**: Format specification for task output
- **agent**: Name of the agent responsible for the task
- **human_input**: Whether human approval is required

### Error Handling

#### Exception Types

**DataLoadingError**
```python
class DataLoadingError(Exception):
    """Raised when dataset loading fails"""
    pass
```

**QueryExecutionError**
```python
class QueryExecutionError(Exception):
    """Raised when SQL query execution fails"""
    pass
```

**SecurityViolationError**
```python
class SecurityViolationError(Exception):
    """Raised when security policies are violated"""
    pass
```

#### Error Response Format

```python
error_response = {
    "success": False,
    "error": "Error message",
    "error_type": "ErrorType",
    "details": {
        "context": "Additional context",
        "suggestions": ["Suggestion 1", "Suggestion 2"]
    }
}
```

### Usage Examples

#### Basic Usage

```python
from crew_orchestrator import CrewAITabularAgent

# Initialize the orchestrator
orchestrator = CrewAITabularAgent()

# Load a dataset
load_result = orchestrator.load_dataset("sales_data.csv")
if not load_result['success']:
    print(f"Failed to load dataset: {load_result['error']}")
    exit(1)

# Process a query
query_result = orchestrator.process_query(
    "What is the total revenue by product category?"
)

if query_result['success']:
    print("Query Result:", query_result['result'])
    
    # Get cost summary
    cost_summary = orchestrator.get_cost_summary(query_result['session_id'])
    print("Cost Summary:", cost_summary)
else:
    print(f"Query failed: {query_result['error']}")

# Cleanup
orchestrator.cleanup()
```

#### Advanced Usage with Custom Configuration

```python
import yaml
from crew_orchestrator import CrewAITabularAgent

# Custom agent configuration
custom_config = {
    'sql_generator_agent': {
        'role': 'Custom SQL Developer',
        'goal': 'Generate optimized SQL queries',
        'model': 'gpt-4.1',
        'temperature': 0.1
    }
}

# Save custom configuration
with open('config/custom_agents.yaml', 'w') as f:
    yaml.dump(custom_config, f)

# Initialize with custom configuration
orchestrator = CrewAITabularAgent(config_dir='config')

# Process multiple queries in a session
session_id = "analysis_session_001"

queries = [
    "Show me the top 10 customers by revenue",
    "What is the monthly sales trend?",
    "Which products have the highest profit margin?"
]

for query in queries:
    result = orchestrator.process_query(query, session_id)
    if result['success']:
        print(f"Query: {query}")
        print(f"Result: {result['result']}")
        print("---")

# Get comprehensive cost analysis
final_cost_summary = orchestrator.get_cost_summary(session_id)
print("Final Cost Summary:", final_cost_summary)
```

#### Integration with Streamlit

```python
import streamlit as st
from crew_orchestrator import CrewAITabularAgent

# Initialize in session state
if 'orchestrator' not in st.session_state:
    st.session_state.orchestrator = CrewAITabularAgent()

# File upload
uploaded_file = st.file_uploader("Upload dataset", type=['csv', 'xlsx', 'json'])

if uploaded_file:
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name
    
    # Load dataset
    with st.spinner("Loading dataset..."):
        load_result = st.session_state.orchestrator.load_dataset(tmp_file_path)
    
    if load_result['success']:
        st.success("Dataset loaded successfully!")
        
        # Query interface
        user_query = st.text_input("Enter your question:")
        
        if st.button("Process Query") and user_query:
            with st.spinner("Processing query..."):
                result = st.session_state.orchestrator.process_query(user_query)
            
            if result['success']:
                st.write("**Result:**")
                st.write(result['result'])
                
                # Display cost information
                cost_summary = st.session_state.orchestrator.get_cost_summary(
                    result['session_id']
                )
                with st.expander("Cost Summary"):
                    st.text(cost_summary)
            else:
                st.error(f"Query failed: {result['error']}")
    
    # Cleanup temporary file
    os.unlink(tmp_file_path)
```

This comprehensive technical documentation provides detailed information about the CrewAI Tabular Data Agent system architecture, implementation details, and usage patterns. The system represents a sophisticated approach to natural language data analysis with strong emphasis on security, cost control, and human oversight.

