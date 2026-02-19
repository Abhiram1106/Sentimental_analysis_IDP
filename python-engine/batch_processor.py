"""
Batch File Processing for Sentiment Analysis
Supports CSV, TXT, XLSX files
"""

import os
import csv
import pandas as pd
from typing import List, Dict, Optional
from io import StringIO, BytesIO


class BatchProcessor:
    """Process sentiment analysis on uploaded files"""
    
    def __init__(self, analyzer):
        """
        Initialize batch processor
        
        Args:
            analyzer: SentimentAnalyzer or TransformerSentimentAnalyzer instance
        """
        self.analyzer = analyzer
    
    def process_txt_file(self, file_content: bytes, encoding: str = 'utf-8') -> Dict:
        """
        Process a TXT file (one text per line)
        
        Args:
            file_content: File content as bytes
            encoding: Text encoding
        
        Returns:
            Analysis results
        """
        try:
            # Decode content
            text_content = file_content.decode(encoding)
            
            # Split into lines and filter empty
            texts = [line.strip() for line in text_content.split('\n') if line.strip()]
            
            if not texts:
                return {'error': 'No valid texts found in file'}
            
            # Analyze
            result = self.analyzer.analyze_parallel(texts)
            result['file_type'] = 'txt'
            result['lines_processed'] = len(texts)
            
            return result
        
        except Exception as e:
            return {'error': f'Failed to process TXT file: {str(e)}'}
    
    def process_csv_file(
        self,
        file_content: bytes,
        text_column: str = 'text',
        encoding: str = 'utf-8'
    ) -> Dict:
        """
        Process a CSV file
        
        Args:
            file_content: File content as bytes
            text_column: Name of column containing text
            encoding: Text encoding
        
        Returns:
            Analysis results with additional CSV metadata
        """
        try:
            # Read CSV
            csv_content = StringIO(file_content.decode(encoding))
            df = pd.read_csv(csv_content)
            
            # Validate column exists
            if text_column not in df.columns:
                return {
                    'error': f'Column "{text_column}" not found. Available columns: {list(df.columns)}'
                }
            
            # Extract texts
            texts = df[text_column].dropna().astype(str).tolist()
            
            if not texts:
                return {'error': 'No valid texts found in specified column'}
            
            # Analyze
            result = self.analyzer.analyze_parallel(texts)
            result['file_type'] = 'csv'
            result['total_rows'] = len(df)
            result['columns'] = list(df.columns)
            result['text_column'] = text_column
            
            return result
        
        except Exception as e:
            return {'error': f'Failed to process CSV file: {str(e)}'}
    
    def process_xlsx_file(
        self,
        file_content: bytes,
        text_column: str = 'text',
        sheet_name: Optional[str] = None
    ) -> Dict:
        """
        Process an Excel (XLSX) file
        
        Args:
            file_content: File content as bytes
            text_column: Name of column containing text
            sheet_name: Sheet name (default: first sheet)
        
        Returns:
            Analysis results with Excel metadata
        """
        try:
            # Read Excel
            excel_content = BytesIO(file_content)
            
            if sheet_name:
                df = pd.read_excel(excel_content, sheet_name=sheet_name)
            else:
                df = pd.read_excel(excel_content)
            
            # Validate column exists
            if text_column not in df.columns:
                return {
                    'error': f'Column "{text_column}" not found. Available columns: {list(df.columns)}'
                }
            
            # Extract texts
            texts = df[text_column].dropna().astype(str).tolist()
            
            if not texts:
                return {'error': 'No valid texts found in specified column'}
            
            # Analyze
            result = self.analyzer.analyze_parallel(texts)
            result['file_type'] = 'xlsx'
            result['total_rows'] = len(df)
            result['columns'] = list(df.columns)
            result['text_column'] = text_column
            if sheet_name:
                result['sheet_name'] = sheet_name
            
            return result
        
        except Exception as e:
            return {'error': f'Failed to process Excel file: {str(e)}'}
    
    def process_file(
        self,
        file_content: bytes,
        filename: str,
        text_column: str = 'text',
        encoding: str = 'utf-8'
    ) -> Dict:
        """
        Auto-detect file type and process
        
        Args:
            file_content: File content as bytes
            filename: Original filename
            text_column: Column name for CSV/Excel
            encoding: Text encoding
        
        Returns:
            Analysis results
        """
        # Detect file type from extension
        ext = filename.lower().split('.')[-1]
        
        if ext == 'txt':
            return self.process_txt_file(file_content, encoding)
        elif ext == 'csv':
            return self.process_csv_file(file_content, text_column, encoding)
        elif ext in ['xlsx', 'xls']:
            return self.process_xlsx_file(file_content, text_column)
        else:
            return {'error': f'Unsupported file type: {ext}. Supported: txt, csv, xlsx'}
    
    def export_results_to_csv(self, results: Dict, output_path: str):
        """
        Export detailed results to CSV file
        
        Args:
            results: Analysis results dictionary
            output_path: Path to save CSV
        """
        if 'detailed_results' not in results:
            raise ValueError("Results don't contain detailed_results")
        
        df = pd.DataFrame(results['detailed_results'])
        df.to_csv(output_path, index=False)
        print(f"Results exported to {output_path}")


if __name__ == "__main__":
    # Example usage
    from sentiment_analyzer import SentimentAnalyzer
    
    analyzer = SentimentAnalyzer()
    processor = BatchProcessor(analyzer)
    
    # Create sample TXT file
    sample_txt = b"""I love this product!
This is terrible.
It's okay, nothing special.
Best purchase ever!
Very disappointed."""
    
    print("Processing sample TXT file...")
    result = processor.process_txt_file(sample_txt)
    print(f"Results: {result['summary']}")
    print(f"Processing time: {result['processing_time']:.3f}s")
