"""
Streamlit Web Interface for CrewAI Tabular Agent
Provides user-friendly interface with human-in-the-loop capabilities
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import time
import uuid
from datetime import datetime
from pathlib import Path
import tempfile
import os

from crew_orchestrator import CrewAITabularAgent
from utils.cost_tracker import CostTracker

# Configure Streamlit page
st.set_page_config(
    page_title="CrewAI Tabular Data Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .agent-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #1f77b4;
    }
    .cost-summary {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #1f77b4;
    }
    .human-input-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #ffc107;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #28a745;
    }
    .error-box {
        background-color: #f8d7da;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #dc3545;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'orchestrator' not in st.session_state:
    st.session_state.orchestrator = CrewAITabularAgent()
    
if 'dataset_loaded' not in st.session_state:
    st.session_state.dataset_loaded = False
    
if 'query_history' not in st.session_state:
    st.session_state.query_history = []
    
if 'current_session_id' not in st.session_state:
    st.session_state.current_session_id = None

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<h1 class="main-header">🤖 CrewAI Tabular Data Agent</h1>', unsafe_allow_html=True)
    st.markdown("**Multi-Agent SQL Assistant with Human-in-the-Loop & Cost Control**")
    
    # Sidebar
    with st.sidebar:
        st.header("📊 Dataset Management")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload your dataset",
            type=['csv', 'xlsx', 'xls', 'json'],
            help="Supported formats: CSV, Excel, JSON"
        )
        
        if uploaded_file is not None:
            if st.button("Load Dataset", type="primary"):
                load_dataset(uploaded_file)
                
        # Dataset info
        if st.session_state.dataset_loaded:
            st.success("✅ Dataset loaded successfully!")
            if hasattr(st.session_state, 'dataset_info'):
                info = st.session_state.dataset_info
                st.write(f"**Rows:** {info.get('rows', 'N/A'):,}")
                st.write(f"**Columns:** {info.get('columns', 'N/A')}")
                
        st.divider()
        
        # Agent Status
        st.header("🤖 Agent Status")
        display_agent_status()
        
        st.divider()
        
        # Cost Tracking
        st.header("💰 Cost Tracking")
        display_cost_summary()
    
    # Main content area
    if not st.session_state.dataset_loaded:
        display_welcome_screen()
    else:
        display_query_interface()

def load_dataset(uploaded_file):
    """Load and process uploaded dataset"""
    try:
        with st.spinner("Loading dataset..."):
            # Save uploaded file to temporary location
            temp_file = tempfile.NamedTemporaryFile(
                delete=False, 
                suffix=Path(uploaded_file.name).suffix
            )
            temp_file.write(uploaded_file.getvalue())
            temp_file.close()
            
            # Load dataset using orchestrator
            result = st.session_state.orchestrator.load_dataset(temp_file.name)
            
            if result['success']:
                st.session_state.dataset_loaded = True
                st.session_state.dataset_info = result['metadata']
                st.session_state.schema_description = result['schema_description']
                
                st.success(result['message'])
                
                # Display dataset preview
                display_dataset_preview(temp_file.name)
                
            else:
                st.error(f"Failed to load dataset: {result.get('error', 'Unknown error')}")
                
            # Cleanup temp file
            os.unlink(temp_file.name)
            
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")

def display_dataset_preview(file_path):
    """Display preview of loaded dataset"""
    try:
        # Read file for preview
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file_path)
        elif file_path.endswith('.json'):
            df = pd.read_json(file_path)
        else:
            return
            
        st.subheader("📋 Dataset Preview")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**First 5 rows:**")
            st.dataframe(df.head(), use_container_width=True)
            
        with col2:
            st.write("**Data Types:**")
            dtype_df = pd.DataFrame({
                'Column': df.columns,
                'Type': df.dtypes.astype(str),
                'Non-Null': df.count(),
                'Null %': ((df.isnull().sum() / len(df)) * 100).round(1)
            })
            st.dataframe(dtype_df, use_container_width=True)
            
    except Exception as e:
        st.warning(f"Could not display preview: {str(e)}")

def display_welcome_screen():
    """Display welcome screen when no dataset is loaded"""
    st.markdown("""
    ## Welcome to CrewAI Tabular Data Agent! 🚀
    
    This advanced multi-agent system helps you analyze tabular data using natural language queries.
    
    ### 🌟 Key Features:
    - **Multi-Agent Architecture**: Specialized AI agents for different tasks
    - **Human-in-the-Loop**: Review and approve SQL queries before execution
    - **Cost Transparency**: Track LLM usage and costs in real-time
    - **Security First**: Only safe SELECT queries are executed
    - **Interactive Visualizations**: Automatic chart generation
    
    ### 🚀 How It Works:
    1. **Upload** your CSV, Excel, or JSON file
    2. **Ask** questions in natural language
    3. **Review** the generated SQL query
    4. **Approve** or modify the query
    5. **Explore** results with interactive visualizations
    
    ### 🤖 Meet Your AI Team:
    """)
    
    agents_info = [
        ("🔍 Schema Analyst", "Analyzes your data structure and extracts schema information"),
        ("💻 SQL Generator", "Converts your questions into precise SQL queries"),
        ("🛡️ Security Reviewer", "Validates queries for safety and correctness"),
        ("⚡ Query Executor", "Safely executes approved queries"),
        ("📊 Data Analyst", "Generates insights and visualization recommendations")
    ]
    
    for agent_name, description in agents_info:
        st.markdown(f"""
        <div class="agent-card">
            <strong>{agent_name}</strong><br>
            {description}
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("### 📁 Get Started")
    st.info("👈 Upload your dataset using the sidebar to begin!")

def display_query_interface():
    """Display the main query interface"""
    st.subheader("💬 Ask Questions About Your Data")
    
    # Query input
    user_query = st.text_area(
        "Enter your question:",
        placeholder="e.g., What is the average sales by region?",
        height=100
    )
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        if st.button("🚀 Process Query", type="primary", disabled=not user_query.strip()):
            process_user_query(user_query)
            
    with col2:
        if st.button("📋 Show Schema"):
            display_schema_info()
            
    with col3:
        if st.button("🔄 Clear History"):
            st.session_state.query_history = []
            st.rerun()
    
    # Display query history
    if st.session_state.query_history:
        st.subheader("📚 Query History")
        for i, query_item in enumerate(reversed(st.session_state.query_history)):
            with st.expander(f"Query {len(st.session_state.query_history) - i}: {query_item['query'][:50]}..."):
                display_query_result(query_item)

def process_user_query(user_query):
    """Process user query through CrewAI system"""
    try:
        # Start new session
        session_id = str(uuid.uuid4())
        st.session_state.current_session_id = session_id
        
        with st.spinner("🤖 AI agents are working on your query..."):
            # Process query
            result = st.session_state.orchestrator.process_query(user_query, session_id)
            
            # Add to history
            query_item = {
                'timestamp': datetime.now(),
                'query': user_query,
                'session_id': session_id,
                'result': result
            }
            st.session_state.query_history.append(query_item)
            
            # Display result
            display_query_result(query_item)
            
    except Exception as e:
        st.error(f"Error processing query: {str(e)}")

def display_query_result(query_item):
    """Display the result of a query"""
    result = query_item['result']
    
    st.write(f"**Query:** {query_item['query']}")
    st.write(f"**Time:** {query_item['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
    
    if result['success']:
        st.markdown('<div class="success-box">✅ Query processed successfully!</div>', unsafe_allow_html=True)
        
        # Display execution time
        st.write(f"**Execution Time:** {result.get('execution_time', 0):.2f} seconds")
        
        # Display result
        st.write("**Result:**")
        st.text(result.get('result', 'No result available'))
        
        # Display cost summary
        if 'cost_summary' in result:
            display_detailed_cost_summary(result['cost_summary'])
            
    else:
        st.markdown(f'<div class="error-box">❌ Query failed: {result.get("error", "Unknown error")}</div>', unsafe_allow_html=True)

def display_schema_info():
    """Display database schema information"""
    if hasattr(st.session_state, 'schema_description'):
        st.subheader("🗄️ Database Schema")
        st.code(st.session_state.schema_description, language='sql')
    else:
        st.warning("No schema information available")

def display_agent_status():
    """Display current agent status"""
    agents = [
        "Schema Analyst",
        "SQL Generator", 
        "Security Reviewer",
        "Query Executor",
        "Data Analyst"
    ]
    
    for agent in agents:
        if st.session_state.dataset_loaded:
            st.success(f"✅ {agent}")
        else:
            st.info(f"⏳ {agent}")

def display_cost_summary():
    """Display cost tracking summary"""
    if st.session_state.current_session_id:
        try:
            cost_summary = st.session_state.orchestrator.get_cost_summary(
                st.session_state.current_session_id
            )
            st.text(cost_summary)
        except:
            st.info("No cost data available")
    else:
        st.info("No active session")

def display_detailed_cost_summary(cost_data):
    """Display detailed cost breakdown"""
    st.subheader("💰 Cost Breakdown")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Cost", f"${cost_data.get('total_cost', 0):.4f}")
        
    with col2:
        st.metric("Total Tokens", f"{cost_data.get('total_tokens', 0):,}")
        
    with col3:
        st.metric("API Calls", cost_data.get('total_api_calls', 0))
        
    with col4:
        st.metric("Duration", f"{cost_data.get('duration_seconds', 0):.1f}s")
    
    # Agent breakdown
    if 'agent_breakdown' in cost_data:
        st.write("**Cost by Agent:**")
        agent_df = pd.DataFrame.from_dict(cost_data['agent_breakdown'], orient='index')
        if not agent_df.empty:
            st.dataframe(agent_df, use_container_width=True)

def create_sample_data():
    """Create sample data for demonstration"""
    sample_data = {
        'product': ['Laptop', 'Mouse', 'Keyboard', 'Monitor', 'Headphones'] * 20,
        'category': ['Electronics', 'Accessories', 'Accessories', 'Electronics', 'Accessories'] * 20,
        'price': [999.99, 29.99, 79.99, 299.99, 149.99] * 20,
        'quantity_sold': [1, 5, 3, 2, 4] * 20,
        'region': ['North', 'South', 'East', 'West', 'Central'] * 20,
        'sales_date': pd.date_range('2024-01-01', periods=100, freq='D'),
        'customer_rating': [4.5, 4.0, 4.2, 4.8, 4.1] * 20
    }
    
    df = pd.DataFrame(sample_data)
    df['total_revenue'] = df['price'] * df['quantity_sold']
    
    return df

# Add sample data option
with st.sidebar:
    st.divider()
    st.header("🎯 Try Sample Data")
    if st.button("Load Sample Dataset"):
        try:
            sample_df = create_sample_data()
            
            # Save to temporary file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
            sample_df.to_csv(temp_file.name, index=False)
            temp_file.close()
            
            # Load using orchestrator
            result = st.session_state.orchestrator.load_dataset(temp_file.name)
            
            if result['success']:
                st.session_state.dataset_loaded = True
                st.session_state.dataset_info = result['metadata']
                st.session_state.schema_description = result['schema_description']
                st.success("Sample dataset loaded!")
                st.rerun()
            else:
                st.error("Failed to load sample dataset")
                
            # Cleanup
            os.unlink(temp_file.name)
            
        except Exception as e:
            st.error(f"Error loading sample data: {str(e)}")

if __name__ == "__main__":
    main()

