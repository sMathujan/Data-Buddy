"""
Data Loading Utility for CrewAI Tabular Agent
Handles file upload, validation, and SQLite database creation
"""

import pandas as pd
import sqlite3
import os
import tempfile
from typing import Dict, Any, Tuple, Optional
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataLoader:
    """Handles data loading and SQLite database creation for the CrewAI system"""
    
    def __init__(self):
        self.supported_formats = ['.csv', '.xlsx', '.xls', '.json']
        self.temp_db_path = None
        self.schema_info = None
        
    def load_file(self, file_path: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Load data from various file formats
        
        Args:
            file_path: Path to the data file
            
        Returns:
            Tuple of (DataFrame, metadata dict)
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
            
        file_ext = file_path.suffix.lower()
        
        if file_ext not in self.supported_formats:
            raise ValueError(f"Unsupported file format: {file_ext}")
            
        logger.info(f"Loading file: {file_path}")
        
        try:
            if file_ext == '.csv':
                df = pd.read_csv(file_path)
            elif file_ext in ['.xlsx', '.xls']:
                df = pd.read_excel(file_path)
            elif file_ext == '.json':
                df = pd.read_json(file_path)
            else:
                raise ValueError(f"Unsupported format: {file_ext}")
                
            # Generate metadata
            metadata = self._generate_metadata(df, file_path)
            
            logger.info(f"Successfully loaded {len(df)} rows and {len(df.columns)} columns")
            return df, metadata
            
        except Exception as e:
            logger.error(f"Error loading file: {str(e)}")
            raise
            
    def _generate_metadata(self, df: pd.DataFrame, file_path: Path) -> Dict[str, Any]:
        """Generate metadata about the loaded dataset"""
        return {
            'file_name': file_path.name,
            'file_path': str(file_path),
            'file_size': file_path.stat().st_size,
            'rows': len(df),
            'columns': len(df.columns),
            'column_names': list(df.columns),
            'data_types': df.dtypes.to_dict(),
            'memory_usage': df.memory_usage(deep=True).sum(),
            'has_null_values': df.isnull().any().any(),
            'null_counts': df.isnull().sum().to_dict()
        }
        
    def create_sqlite_database(self, df: pd.DataFrame, table_name: str = 'data_table') -> str:
        """
        Create a temporary SQLite database from the DataFrame
        
        Args:
            df: DataFrame to convert
            table_name: Name for the table in SQLite
            
        Returns:
            Path to the temporary SQLite database
        """
        # Create temporary database file
        temp_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        temp_db.close()
        self.temp_db_path = temp_db.name
        
        try:
            # Connect to SQLite and create table
            conn = sqlite3.connect(self.temp_db_path)
            
            # Clean column names for SQLite compatibility
            df_clean = df.copy()
            df_clean.columns = [self._clean_column_name(col) for col in df.columns]
            
            # Write DataFrame to SQLite
            df_clean.to_sql(table_name, conn, index=False, if_exists='replace')
            
            # Generate schema information
            self.schema_info = self._extract_schema_info(conn, table_name)
            
            conn.close()
            logger.info(f"Created SQLite database: {self.temp_db_path}")
            
            return self.temp_db_path
            
        except Exception as e:
            logger.error(f"Error creating SQLite database: {str(e)}")
            if os.path.exists(self.temp_db_path):
                os.unlink(self.temp_db_path)
            raise
            
    def _clean_column_name(self, column_name: str) -> str:
        """Clean column names for SQLite compatibility"""
        # Replace spaces and special characters with underscores
        import re
        cleaned = re.sub(r'[^\w]', '_', str(column_name))
        # Remove multiple consecutive underscores
        cleaned = re.sub(r'_+', '_', cleaned)
        # Remove leading/trailing underscores
        cleaned = cleaned.strip('_')
        # Ensure it doesn't start with a number
        if cleaned and cleaned[0].isdigit():
            cleaned = f'col_{cleaned}'
        return cleaned or 'unnamed_column'
        
    def _extract_schema_info(self, conn: sqlite3.Connection, table_name: str) -> Dict[str, Any]:
        """Extract comprehensive schema information from the SQLite database"""
        cursor = conn.cursor()
        
        # Get table info
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns_info = cursor.fetchall()
        
        # Get sample data
        cursor.execute(f"SELECT * FROM {table_name} LIMIT 5")
        sample_data = cursor.fetchall()
        
        # Get row count
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        row_count = cursor.fetchone()[0]
        
        # Build schema information
        schema_info = {
            'table_name': table_name,
            'row_count': row_count,
            'columns': [],
            'sample_data': sample_data
        }
        
        for col_info in columns_info:
            col_name = col_info[1]
            col_type = col_info[2]
            
            # Get column statistics
            cursor.execute(f"SELECT COUNT(DISTINCT {col_name}) FROM {table_name}")
            unique_count = cursor.fetchone()[0]
            
            cursor.execute(f"SELECT COUNT(*) FROM {table_name} WHERE {col_name} IS NULL")
            null_count = cursor.fetchone()[0]
            
            schema_info['columns'].append({
                'name': col_name,
                'type': col_type,
                'unique_values': unique_count,
                'null_count': null_count,
                'null_percentage': (null_count / row_count) * 100 if row_count > 0 else 0
            })
            
        return schema_info
        
    def get_schema_description(self) -> str:
        """Generate a human-readable schema description for agents"""
        if not self.schema_info:
            return "No schema information available"
            
        description = f"""
DATABASE SCHEMA INFORMATION:

Table: {self.schema_info['table_name']}
Total Rows: {self.schema_info['row_count']:,}

COLUMNS:
"""
        
        for col in self.schema_info['columns']:
            description += f"""
- {col['name']} ({col['type']})
  * Unique values: {col['unique_values']:,}
  * Null values: {col['null_count']:,} ({col['null_percentage']:.1f}%)
"""
        
        # Add sample data
        if self.schema_info['sample_data']:
            description += "\nSAMPLE DATA (first 5 rows):\n"
            col_names = [col['name'] for col in self.schema_info['columns']]
            description += " | ".join(col_names) + "\n"
            description += "-" * (len(" | ".join(col_names))) + "\n"
            
            for row in self.schema_info['sample_data']:
                description += " | ".join([str(val) if val is not None else 'NULL' for val in row]) + "\n"
                
        return description
        
    def cleanup(self):
        """Clean up temporary files"""
        if self.temp_db_path and os.path.exists(self.temp_db_path):
            try:
                os.unlink(self.temp_db_path)
                logger.info(f"Cleaned up temporary database: {self.temp_db_path}")
            except Exception as e:
                logger.warning(f"Failed to cleanup temporary database: {str(e)}")
                
    def __del__(self):
        """Ensure cleanup on object destruction"""
        self.cleanup()


def test_data_loader():
    """Test function for the DataLoader class"""
    # Create sample data for testing
    sample_data = {
        'name': ['Alice', 'Bob', 'Charlie', 'Diana'],
        'age': [25, 30, 35, 28],
        'city': ['New York', 'London', 'Paris', 'Tokyo'],
        'salary': [50000, 60000, 70000, 55000]
    }
    
    df = pd.DataFrame(sample_data)
    
    # Save as CSV for testing
    test_file = '/tmp/test_data.csv'
    df.to_csv(test_file, index=False)
    
    # Test the DataLoader
    loader = DataLoader()
    
    try:
        # Load file
        loaded_df, metadata = loader.load_file(test_file)
        print("Loaded DataFrame:")
        print(loaded_df)
        print("\nMetadata:")
        print(metadata)
        
        # Create SQLite database
        db_path = loader.create_sqlite_database(loaded_df)
        print(f"\nCreated database: {db_path}")
        
        # Get schema description
        schema_desc = loader.get_schema_description()
        print("\nSchema Description:")
        print(schema_desc)
        
    finally:
        # Cleanup
        loader.cleanup()
        if os.path.exists(test_file):
            os.unlink(test_file)


if __name__ == "__main__":
    test_data_loader()

