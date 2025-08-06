"""
CrewAI Orchestrator for Tabular Data Agent
Main orchestration logic for the multi-agent system
"""

import os
import yaml
import json
import uuid
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path

from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool
from dotenv import load_dotenv

from utils.data_loader import DataLoader
from utils.cost_tracker import CostTracker
from utils.database_tools import create_database_tools

import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CrewAITabularAgent:
    """Main orchestrator for the CrewAI-based tabular data agent system"""
    
    def __init__(self, config_dir: str = "config"):
        """Initialize the CrewAI orchestrator"""
        # Load environment variables
        load_dotenv()
        
        self.config_dir = Path(config_dir)
        self.data_loader = DataLoader()
        self.cost_tracker = CostTracker()
        
        # Session management
        self.current_session_id = None
        self.database_path = None
        self.database_tools = []
        
        # Load configurations
        self.agents_config = self._load_config("agents.yaml")
        self.tasks_config = self._load_config("tasks.yaml")
        
        # Initialize agents and tasks
        self.agents = {}
        self.tasks = {}
        
        logger.info("CrewAI Tabular Agent initialized")
        
    def _load_config(self, filename: str) -> Dict[str, Any]:
        """Load YAML configuration file"""
        config_path = self.config_dir / filename
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except Exception as e:
            logger.error(f"Failed to load config {filename}: {str(e)}")
            return {}
            
    def _create_agents(self) -> Dict[str, Agent]:
        """Create CrewAI agents from configuration"""
        agents = {}
        
        for agent_name, config in self.agents_config.items():
            try:
                # Add database tools to specific agents
                tools = []
                if agent_name in ['data_execution_agent', 'sql_generator_agent', 'sql_reviewer_agent']:
                    tools = self.database_tools
                    
                agent = Agent(
                    role=config['role'],
                    goal=config['goal'],
                    backstory=config['backstory'],
                    verbose=config.get('verbose', True),
                    allow_delegation=config.get('allow_delegation', False),
                    tools=tools,
                    llm=self._get_llm_config(config)
                )
                
                agents[agent_name] = agent
                logger.info(f"Created agent: {agent_name}")
                
            except Exception as e:
                logger.error(f"Failed to create agent {agent_name}: {str(e)}")
                
        return agents
        
    def _get_llm_config(self, agent_config: Dict[str, Any]) -> Optional[Any]:
        """Get LLM configuration for agent"""
        # For now, we'll use the default OpenAI configuration
        # This can be extended to support different LLM providers
        return None  # CrewAI will use default OpenAI configuration
        
    def _create_tasks(self, context: Dict[str, Any]) -> List[Task]:
        """Create CrewAI tasks from configuration with context"""
        tasks = []
        
        # Define task execution order
        task_order = [
            'schema_extraction_task',
            'sql_generation_task', 
            'sql_review_task',
            'data_execution_task',
            'data_analysis_task'
        ]
        
        previous_task = None
        
        for task_name in task_order:
            if task_name not in self.tasks_config:
                logger.warning(f"Task {task_name} not found in configuration")
                continue
                
            config = self.tasks_config[task_name]
            
            try:
                # Format description with context
                description = config['description'].format(**context)
                expected_output = config['expected_output'].format(**context) if 'expected_output' in config else None
                
                # Get the agent for this task
                agent_name = config['agent']
                if agent_name not in self.agents:
                    logger.error(f"Agent {agent_name} not found for task {task_name}")
                    continue
                    
                task = Task(
                    description=description,
                    expected_output=expected_output,
                    agent=self.agents[agent_name],
                    context=[previous_task] if previous_task else None,
                    human_input=config.get('human_input', False)
                )
                
                tasks.append(task)
                previous_task = task
                self.tasks[task_name] = task
                
                logger.info(f"Created task: {task_name}")
                
            except Exception as e:
                logger.error(f"Failed to create task {task_name}: {str(e)}")
                
        return tasks
        
    def load_dataset(self, file_path: str) -> Dict[str, Any]:
        """Load dataset and prepare for analysis"""
        try:
            logger.info(f"Loading dataset: {file_path}")
            
            # Load the file
            df, metadata = self.data_loader.load_file(file_path)
            
            # Create SQLite database
            self.database_path = self.data_loader.create_sqlite_database(df)
            
            # Create database tools
            self.database_tools = create_database_tools(self.database_path)
            
            # Get schema description
            schema_description = self.data_loader.get_schema_description()
            
            result = {
                'success': True,
                'file_path': file_path,
                'database_path': self.database_path,
                'metadata': metadata,
                'schema_description': schema_description,
                'message': f"Successfully loaded {metadata['rows']} rows and {metadata['columns']} columns"
            }
            
            logger.info(f"Dataset loaded successfully: {result['message']}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to load dataset: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'message': f"Failed to load dataset: {str(e)}"
            }
            
    def process_query(self, user_query: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Process a natural language query using the CrewAI system"""
        try:
            # Start cost tracking session
            if not session_id:
                session_id = str(uuid.uuid4())
                
            self.current_session_id = session_id
            self.cost_tracker.start_session(session_id)
            
            logger.info(f"Processing query in session {session_id}: {user_query}")
            
            # Check if dataset is loaded
            if not self.database_path:
                return {
                    'success': False,
                    'error': 'No dataset loaded',
                    'message': 'Please load a dataset first'
                }
                
            # Create agents
            self.agents = self._create_agents()
            
            if not self.agents:
                return {
                    'success': False,
                    'error': 'Failed to create agents',
                    'message': 'Agent creation failed'
                }
                
            # Prepare context for tasks
            context = {
                'user_query': user_query,
                'dataset_path': self.database_path,
                'file_type': 'SQLite Database',
                'database_schema': self.data_loader.get_schema_description(),
                'database_path': self.database_path,
                'generated_sql': '{generated_sql}',  # Will be filled by previous tasks
                'approved_sql': '{approved_sql}',    # Will be filled by previous tasks
                'query_results': '{query_results}'   # Will be filled by previous tasks
            }
            
            # Create tasks
            tasks = self._create_tasks(context)
            
            if not tasks:
                return {
                    'success': False,
                    'error': 'Failed to create tasks',
                    'message': 'Task creation failed'
                }
                
            # Create and execute crew
            crew = Crew(
                agents=list(self.agents.values()),
                tasks=tasks,
                process=Process.sequential,
                verbose=True
            )
            
            logger.info("Starting crew execution...")
            start_time = datetime.now()
            
            # Execute the crew
            result = crew.kickoff()
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # End cost tracking
            cost_summary = self.cost_tracker.end_session(session_id)
            
            logger.info(f"Crew execution completed in {execution_time:.2f} seconds")
            
            return {
                'success': True,
                'session_id': session_id,
                'user_query': user_query,
                'result': str(result),
                'execution_time': execution_time,
                'cost_summary': self.cost_tracker.get_session_summary(session_id),
                'message': 'Query processed successfully'
            }
            
        except Exception as e:
            logger.error(f"Failed to process query: {str(e)}")
            
            # End cost tracking even on error
            if session_id:
                self.cost_tracker.end_session(session_id)
                
            return {
                'success': False,
                'error': str(e),
                'message': f'Query processing failed: {str(e)}',
                'session_id': session_id
            }
            
    def get_cost_summary(self, session_id: Optional[str] = None) -> str:
        """Get formatted cost summary for a session"""
        return self.cost_tracker.format_cost_summary(session_id)
        
    def cleanup(self):
        """Clean up resources"""
        if self.data_loader:
            self.data_loader.cleanup()
        logger.info("Resources cleaned up")
        
    def __del__(self):
        """Ensure cleanup on destruction"""
        self.cleanup()


def create_sample_data():
    """Create sample data for testing"""
    import pandas as pd
    import tempfile
    
    # Create sample sales data
    sample_data = {
        'order_id': range(1, 101),
        'customer_name': [f'Customer_{i}' for i in range(1, 101)],
        'product': ['Product_A', 'Product_B', 'Product_C'] * 33 + ['Product_A'],
        'quantity': [1, 2, 3, 4, 5] * 20,
        'price': [10.99, 15.99, 25.99, 35.99, 45.99] * 20,
        'order_date': pd.date_range('2024-01-01', periods=100, freq='D'),
        'region': ['North', 'South', 'East', 'West'] * 25
    }
    
    df = pd.DataFrame(sample_data)
    df['total_amount'] = df['quantity'] * df['price']
    
    # Save to temporary file
    temp_file = tempfile.NamedTemporaryFile(suffix='.csv', delete=False)
    temp_file.close()
    
    df.to_csv(temp_file.name, index=False)
    return temp_file.name


def test_crew_orchestrator():
    """Test the CrewAI orchestrator"""
    # Create sample data
    sample_file = create_sample_data()
    
    try:
        # Initialize orchestrator
        orchestrator = CrewAITabularAgent()
        
        # Load dataset
        load_result = orchestrator.load_dataset(sample_file)
        print("Load Result:")
        print(json.dumps(load_result, indent=2, default=str))
        
        if load_result['success']:
            # Process a query
            query_result = orchestrator.process_query(
                "What is the total sales amount by region?"
            )
            
            print("\nQuery Result:")
            print(json.dumps(query_result, indent=2, default=str))
            
            # Get cost summary
            if query_result.get('session_id'):
                cost_summary = orchestrator.get_cost_summary(query_result['session_id'])
                print("\nCost Summary:")
                print(cost_summary)
                
    finally:
        # Cleanup
        orchestrator.cleanup()
        os.unlink(sample_file)


if __name__ == "__main__":
    test_crew_orchestrator()

