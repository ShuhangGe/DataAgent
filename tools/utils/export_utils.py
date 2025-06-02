#!/usr/bin/env python3
"""
Export Utilities for User Behavior Analysis Tools
Provides export functions and data extraction utilities
"""

import json
import pandas as pd
from datetime import datetime
from typing import Dict, List

def export_results(data, filename: str, format_type: str = 'json'):
    """
    Export analysis results or filtered data to file with custom formatting for event sequences
    
    Args:
        data: Either standardized analysis result or raw filtered data dictionary
        filename: Output filename (without extension)
        format_type: 'json' or 'csv'
    """
    if not data:
        print("⚠️  No data to export")
        return
    
    # Add file extension if not provided
    if not filename.endswith(('.json', '.csv')):
        filename = f"{filename}.{format_type}"
    
    try:
        if format_type.lower() == 'json':
            # Handle both standardized and raw data
            if isinstance(data, dict) and 'analysis_info' in data:
                # Standardized analysis result
                export_data = data
            else:
                # Raw filtered data - wrap in standardized format
                export_data = {
                    'analysis_info': {
                        'analysis_type': 'filtered_data_export',
                        'timestamp': datetime.now().isoformat(),
                        'total_records': len(data) if isinstance(data, dict) else 1,
                        'export_format': 'json'
                    },
                    'summary': {
                        'total_users': len(data) if isinstance(data, dict) else 1,
                        'data_type': 'filtered_user_data'
                    },
                    'results': data
                }
            
            # Custom JSON formatting to keep event sequence arrays on single lines
            def format_json_with_compact_arrays(obj, indent=0):
                """Custom JSON formatter that keeps event sequence arrays on single lines"""
                if isinstance(obj, dict):
                    items = []
                    for key, value in obj.items():
                        if key in ['event_sequence', 'day2_event_sequence', 'combined_event_sequence'] and isinstance(value, list):
                            # Format all event sequence arrays on a single line
                            formatted_value = json.dumps(value, ensure_ascii=False)
                        elif isinstance(value, (dict, list)) and key not in ['event_sequence', 'day2_event_sequence', 'combined_event_sequence']:
                            formatted_value = format_json_with_compact_arrays(value, indent + 2)
                        else:
                            formatted_value = json.dumps(value, ensure_ascii=False, default=str)
                        
                        items.append(f'{"  " * (indent + 1)}"{key}": {formatted_value}')
                    
                    if items:
                        return "{\n" + ",\n".join(items) + f"\n{'  ' * indent}}}"
                    else:
                        return "{}"
                        
                elif isinstance(obj, list):
                    if not obj:
                        return "[]"
                    # For non-event_sequence lists, format normally
                    items = [format_json_with_compact_arrays(item, indent + 2) for item in obj]
                    return "[\n" + ",\n".join(f"{'  ' * (indent + 1)}{item}" for item in items) + f"\n{'  ' * indent}]"
                else:
                    return json.dumps(obj, ensure_ascii=False, default=str)
            
            formatted_json = format_json_with_compact_arrays(export_data)
            
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(formatted_json)
            print(f"📁 Data exported to {filename}")
            
        elif format_type.lower() == 'csv':
            # Convert to DataFrame for CSV export
            if isinstance(data, dict) and 'results' in data:
                # Standardized format - extract results
                df_data = data['results']
            else:
                # Raw filtered data
                df_data = data
            
            if isinstance(df_data, dict):
                # Convert dict of user data to DataFrame
                df = pd.DataFrame.from_dict(df_data, orient='index')
            elif isinstance(df_data, list):
                # Convert list to DataFrame
                df = pd.DataFrame(df_data)
            else:
                print("⚠️  Cannot convert data to CSV format")
                return
            
            df.to_csv(filename, index=True, encoding='utf-8')
            print(f"📁 Data exported to {filename}")
            
        else:
            print(f"⚠️  Unsupported format: {format_type}. Use 'json' or 'csv'")
            
    except Exception as e:
        print(f"❌ Export failed: {e}")

def export_filtered_data(filtered_data: Dict, filename: str, format_type: str = 'json'):
    """
    Convenience function to export filtered data from Step 1 filter functions
    
    Args:
        filtered_data: Dictionary of filtered user data from filter functions
        filename: Output filename (without extension)
        format_type: 'json' or 'csv'
    """
    export_results(filtered_data, filename, format_type)

def export_analysis_results(analysis_result: Dict, filename: str, format_type: str = 'json'):
    """
    Convenience function to export standardized analysis results from Step 2 functions
    
    Args:
        analysis_result: Standardized analysis result from analysis functions
        filename: Output filename (without extension) 
        format_type: 'json' or 'csv'
    """
    export_results(analysis_result, filename, format_type)

def extract_device_ids(analysis_result: Dict) -> List[str]:
    """
    Extract device IDs from analysis results
    
    Args:
        analysis_result: Standardized analysis result from filter or analysis functions
        
    Returns:
        List of device IDs
    """
    device_ids = []
    
    # Handle standardized results
    if isinstance(analysis_result, dict) and 'results' in analysis_result:
        results = analysis_result['results']
    else:
        results = analysis_result
    
    if isinstance(results, dict):
        # Results is a dictionary - keys are device IDs
        device_ids = list(results.keys())
    elif isinstance(results, list):
        # Results is a list of records - extract device_id field
        for record in results:
            if isinstance(record, dict) and 'device_id' in record:
                device_ids.append(record['device_id'])
    
    return device_ids

def extract_user_data(analysis_result: Dict) -> Dict:
    """
    Extract user data dictionary from analysis results
    
    Args:
        analysis_result: Standardized analysis result from filter or analysis functions
        
    Returns:
        Dictionary of user data
    """
    # Handle standardized results
    if isinstance(analysis_result, dict) and 'results' in analysis_result:
        results = analysis_result['results']
    else:
        results = analysis_result
    
    if isinstance(results, dict):
        return results
    elif isinstance(results, list):
        # Convert list to dictionary using device_id as key
        user_data = {}
        for record in results:
            if isinstance(record, dict) and 'device_id' in record:
                user_data[record['device_id']] = record
        return user_data
    
    return {} 