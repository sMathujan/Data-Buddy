"""
Custom Database Tools for CrewAI Tabular Agent
Provides specialized tools for database interaction within the CrewAI framework
"""

import sqlite3
import pandas as pd
import time
from typing import Dict, Any, List, Optional, Tuple
from crewai.tools import BaseTool
import logging
import json
import re

logger = logging.getLogger(__name__)

class SQLiteQueryTool(BaseTool):
    """Custom tool for executing SQL queries against SQLite databases"""
    
    name: str = "SQLite Query Executor"
    description: str = """
    Execute SQL queries against a SQLite database and return results.
    Input should be a valid SQL SELECT statement.
    The tool will execute the query and return results in a structured format.
    """
    
    def __init__(self, database_path: str, **kwargs):
        super().__init__(**kwargs)
        self._database_path = database_path
        self._max_rows = 1000  # Limit result size
        self._timeout = 30  # Query timeout in seconds
        
    def _run(self, sql_query: str) -> str:
        """Execute SQL query and return results"""
        try:
            # Validate query safety
            if not self._is_safe_query(sql_query):
                return json.dumps({
                    "error": "Unsafe query detected. Only SELECT statements are allowed.",
                    "query": sql_query
                })
                
            # Execute query
            start_time = time.time()
            conn = sqlite3.connect(self._database_path, timeout=self._timeout)
            
            try:
                # Set row factory for better column access
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                # Add LIMIT if not present to prevent large result sets
                limited_query = self._add_limit_if_needed(sql_query)
                
                cursor.execute(limited_query)
                rows = cursor.fetchall()
                
                execution_time = time.time() - start_time
                
                # Convert to list of dictionaries
                results = []
                if rows:
                    column_names = [description[0] for description in cursor.description]
                    results = [dict(zip(column_names, row)) for row in rows]
                
                return json.dumps({
                    "success": True,
                    "query": limited_query,
                    "results": results,
                    "row_count": len(results),
                    "execution_time": round(execution_time, 3),
                    "columns": column_names if rows else []
                }, default=str)
                
            finally:
                conn.close()
                
        except sqlite3.Error as e:
            return json.dumps({
                "error": f"Database error: {str(e)}",
                "query": sql_query
            })
        except Exception as e:
            return json.dumps({
                "error": f"Execution error: {str(e)}",
                "query": sql_query
            })
            
    def _is_safe_query(self, query: str) -> bool:
        """Check if the query is safe to execute"""
        query_upper = query.upper().strip()
        
        # Only allow SELECT statements
        if not query_upper.startswith('SELECT'):
            return False
            
        # Block dangerous keywords
        dangerous_keywords = [
            'DROP', 'DELETE', 'UPDATE', 'INSERT', 'ALTER', 'CREATE',
            'TRUNCATE', 'REPLACE', 'PRAGMA', 'ATTACH', 'DETACH'
        ]
        
        for keyword in dangerous_keywords:
            if keyword in query_upper:
                return False
                
        return True
        
    def _add_limit_if_needed(self, query: str) -> str:
        """Add LIMIT clause if not present"""
        query_upper = query.upper()
        if 'LIMIT' not in query_upper:
            return f"{query.rstrip(';')} LIMIT {self._max_rows}"
        return query


class SchemaInspectorTool(BaseTool):
    """Tool for inspecting database schema and structure"""
    
    name: str = "Database Schema Inspector"
    description: str = """
    Inspect the structure of a SQLite database.
    Returns detailed information about tables, columns, data types, and sample data.
    Input should be empty or a specific table name to inspect.
    """
    
    def __init__(self, database_path: str, **kwargs):
        super().__init__(**kwargs)
        self._database_path = database_path
        
    def _run(self, table_name: str = "") -> str:
        """Inspect database schema"""
        try:
            conn = sqlite3.connect(self._database_path)
            cursor = conn.cursor()
            
            # Get all tables if no specific table requested
            if not table_name:
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = [row[0] for row in cursor.fetchall()]
            else:
                tables = [table_name]
                
            schema_info = {}
            
            for table in tables:
                # Get table info
                cursor.execute(f"PRAGMA table_info({table})")
                columns_info = cursor.fetchall()
                
                # Get row count
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                row_count = cursor.fetchone()[0]
                
                # Get sample data
                cursor.execute(f"SELECT * FROM {table} LIMIT 3")
                sample_rows = cursor.fetchall()
                
                table_info = {
                    'row_count': row_count,
                    'columns': [],
                    'sample_data': sample_rows
                }
                
                for col_info in columns_info:
                    col_name = col_info[1]
                    col_type = col_info[2]
                    
                    # Get column statistics
                    cursor.execute(f"SELECT COUNT(DISTINCT {col_name}) FROM {table}")
                    unique_count = cursor.fetchone()[0]
                    
                    cursor.execute(f"SELECT COUNT(*) FROM {table} WHERE {col_name} IS NULL")
                    null_count = cursor.fetchone()[0]
                    
                    table_info['columns'].append({
                        'name': col_name,
                        'type': col_type,
                        'unique_values': unique_count,
                        'null_count': null_count,
                        'null_percentage': (null_count / row_count) * 100 if row_count > 0 else 0
                    })
                    
                schema_info[table] = table_info
                
            conn.close()
            
            return json.dumps({
                "success": True,
                "database_path": self._database_path,
                "tables": schema_info
            }, default=str)
            
        except Exception as e:
            return json.dumps({
                "error": f"Schema inspection failed: {str(e)}"
            })


class DataAnalysisTool(BaseTool):
    """Tool for performing data analysis on query results"""
    
    name: str = "Data Analysis Tool"
    description: str = """
    Perform statistical analysis on query results.
    Input should be JSON data from a query result.
    Returns statistical summaries, insights, and visualization recommendations.
    """
    
    def _run(self, query_results: str) -> str:
        """Analyze query results and provide insights"""
        try:
            # Parse query results
            data = json.loads(query_results)
            
            if not data.get('success') or not data.get('results'):
                return json.dumps({
                    "error": "No valid data to analyze",
                    "input": query_results
                })
                
            # Convert to DataFrame for analysis
            df = pd.DataFrame(data['results'])
            
            if df.empty:
                return json.dumps({
                    "message": "No data returned from query",
                    "row_count": 0
                })
                
            analysis = {
                "success": True,
                "row_count": len(df),
                "column_count": len(df.columns),
                "columns": list(df.columns),
                "data_types": df.dtypes.astype(str).to_dict(),
                "summary_statistics": {},
                "insights": [],
                "visualization_recommendations": []
            }
            
            # Generate summary statistics for numeric columns
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                analysis["summary_statistics"] = df[numeric_cols].describe().to_dict()
                
            # Generate insights
            insights = []
            
            # Check for null values
            null_counts = df.isnull().sum()
            if null_counts.sum() > 0:
                insights.append(f"Data contains {null_counts.sum()} null values across {(null_counts > 0).sum()} columns")
                
            # Check for duplicates
            duplicate_count = df.duplicated().sum()
            if duplicate_count > 0:
                insights.append(f"Found {duplicate_count} duplicate rows ({duplicate_count/len(df)*100:.1f}%)")
                
            # Analyze numeric columns
            for col in numeric_cols:
                col_data = df[col].dropna()
                if len(col_data) > 0:
                    insights.append(f"{col}: mean={col_data.mean():.2f}, std={col_data.std():.2f}, range=[{col_data.min():.2f}, {col_data.max():.2f}]")
                    
            # Analyze categorical columns
            categorical_cols = df.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                unique_count = df[col].nunique()
                total_count = len(df[col].dropna())
                if unique_count > 0:
                    insights.append(f"{col}: {unique_count} unique values out of {total_count} non-null entries")
                    
            analysis["insights"] = insights
            
            # Visualization recommendations
            viz_recommendations = []
            
            if len(numeric_cols) >= 2:
                viz_recommendations.append("Scatter plot for correlation analysis between numeric variables")
                viz_recommendations.append("Correlation heatmap for numeric variables")
                
            if len(numeric_cols) >= 1:
                viz_recommendations.append("Histogram or box plot for distribution analysis")
                
            if len(categorical_cols) >= 1:
                viz_recommendations.append("Bar chart for categorical variable frequency")
                
            if len(categorical_cols) >= 1 and len(numeric_cols) >= 1:
                viz_recommendations.append("Box plot or violin plot for numeric vs categorical analysis")
                
            if len(df) > 10:
                viz_recommendations.append("Time series plot if temporal data is present")
                
            analysis["visualization_recommendations"] = viz_recommendations
            
            return json.dumps(analysis, default=str)
            
        except Exception as e:
            return json.dumps({
                "error": f"Analysis failed: {str(e)}",
                "input": query_results
            })


class QueryValidatorTool(BaseTool):
    """Tool for validating SQL queries before execution"""
    
    name: str = "SQL Query Validator"
    description: str = """
    Validate SQL queries for syntax, safety, and correctness.
    Input should be a SQL query string.
    Returns validation results with safety assessment and recommendations.
    """
    
    def __init__(self, database_path: str, **kwargs):
        super().__init__(**kwargs)
        self._database_path = database_path
        
    def _run(self, sql_query: str) -> str:
        """Validate SQL query"""
        try:
            validation_result = {
                "query": sql_query,
                "is_safe": False,
                "is_valid_syntax": False,
                "issues": [],
                "recommendations": [],
                "estimated_complexity": "unknown"
            }
            
            # Check safety
            safety_check = self._check_query_safety(sql_query)
            validation_result.update(safety_check)
            
            # Check syntax by attempting to explain the query
            if validation_result["is_safe"]:
                syntax_check = self._check_syntax(sql_query)
                validation_result.update(syntax_check)
                
            # Analyze query complexity
            complexity = self._analyze_complexity(sql_query)
            validation_result["estimated_complexity"] = complexity
            
            return json.dumps(validation_result, default=str)
            
        except Exception as e:
            return json.dumps({
                "error": f"Validation failed: {str(e)}",
                "query": sql_query
            })
            
    def _check_query_safety(self, query: str) -> Dict[str, Any]:
        """Check if query is safe to execute"""
        issues = []
        recommendations = []
        
        query_upper = query.upper().strip()
        
        # Check if it's a SELECT statement
        if not query_upper.startswith('SELECT'):
            issues.append("Only SELECT statements are allowed")
            return {"is_safe": False, "issues": issues}
            
        # Check for dangerous keywords
        dangerous_keywords = [
            'DROP', 'DELETE', 'UPDATE', 'INSERT', 'ALTER', 'CREATE',
            'TRUNCATE', 'REPLACE', 'PRAGMA', 'ATTACH', 'DETACH'
        ]
        
        for keyword in dangerous_keywords:
            if keyword in query_upper:
                issues.append(f"Dangerous keyword detected: {keyword}")
                
        # Check for LIMIT clause
        if 'LIMIT' not in query_upper:
            recommendations.append("Consider adding a LIMIT clause to prevent large result sets")
            
        # Check for potential SQL injection patterns
        suspicious_patterns = [
            r"';.*--",  # Comment injection
            r"UNION.*SELECT",  # Union injection
            r"OR.*1=1",  # Always true condition
        ]
        
        for pattern in suspicious_patterns:
            if re.search(pattern, query_upper):
                issues.append(f"Potential SQL injection pattern detected")
                
        is_safe = len([issue for issue in issues if "keyword" in issue or "injection" in issue]) == 0
        
        return {
            "is_safe": is_safe,
            "issues": issues,
            "recommendations": recommendations
        }
        
    def _check_syntax(self, query: str) -> Dict[str, Any]:
        """Check query syntax by attempting to explain it"""
        try:
            conn = sqlite3.connect(self._database_path)
            cursor = conn.cursor()
            
            # Use EXPLAIN to check syntax without executing
            cursor.execute(f"EXPLAIN QUERY PLAN {query}")
            cursor.fetchall()  # Consume results
            
            conn.close()
            
            return {
                "is_valid_syntax": True,
                "syntax_message": "Query syntax is valid"
            }
            
        except sqlite3.Error as e:
            return {
                "is_valid_syntax": False,
                "syntax_error": str(e)
            }
            
    def _analyze_complexity(self, query: str) -> str:
        """Analyze query complexity"""
        query_upper = query.upper()
        
        complexity_indicators = {
            'JOIN': 1,
            'SUBQUERY': 2,
            'GROUP BY': 1,
            'ORDER BY': 1,
            'HAVING': 2,
            'WINDOW': 3,
            'WITH': 2,
            'CASE': 1
        }
        
        score = 0
        
        # Count JOINs
        score += query_upper.count('JOIN') * complexity_indicators['JOIN']
        
        # Count subqueries (rough estimate)
        score += query_upper.count('(SELECT') * complexity_indicators['SUBQUERY']
        
        # Count other indicators
        for indicator, weight in complexity_indicators.items():
            if indicator in ['JOIN', 'SUBQUERY']:
                continue
            if indicator in query_upper:
                score += weight
                
        if score == 0:
            return "simple"
        elif score <= 3:
            return "moderate"
        elif score <= 6:
            return "complex"
        else:
            return "very_complex"


def create_database_tools(database_path: str) -> List[BaseTool]:
    """Create a set of database tools for CrewAI agents"""
    return [
        SQLiteQueryTool(database_path),
        SchemaInspectorTool(database_path),
        DataAnalysisTool(),
        QueryValidatorTool(database_path)
    ]


def test_database_tools():
    """Test the database tools"""
    # Create a sample database for testing
    import tempfile
    import os
    
    # Create sample data
    sample_data = pd.DataFrame({
        'id': [1, 2, 3, 4, 5],
        'name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
        'age': [25, 30, 35, 28, 32],
        'salary': [50000, 60000, 70000, 55000, 65000],
        'department': ['IT', 'HR', 'IT', 'Finance', 'HR']
    })
    
    # Create temporary database
    temp_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
    temp_db.close()
    
    try:
        # Write to database
        conn = sqlite3.connect(temp_db.name)
        sample_data.to_sql('employees', conn, index=False)
        conn.close()
        
        # Test tools
        tools = create_database_tools(temp_db.name)
        
        print("Testing Schema Inspector:")
        schema_tool = tools[1]
        result = schema_tool._run("")
        print(result)
        
        print("\nTesting Query Validator:")
        validator_tool = tools[3]
        result = validator_tool._run("SELECT * FROM employees WHERE age > 30")
        print(result)
        
        print("\nTesting Query Executor:")
        query_tool = tools[0]
        result = query_tool._run("SELECT department, AVG(salary) as avg_salary FROM employees GROUP BY department")
        print(result)
        
        print("\nTesting Data Analysis:")
        analysis_tool = tools[2]
        result = analysis_tool._run(result)
        print(result)
        
    finally:
        # Cleanup
        os.unlink(temp_db.name)


if __name__ == "__main__":
    test_database_tools()

