import pandas as pd
import sqlite3
import json
from sqlalchemy import create_engine

def load_data():
    """
    Load CSV data from the specified file path.
    
    Returns:
        pd.DataFrame: Loaded event data
    """
    file_path = "/Users/shuhangge/Desktop/my_projects/Sekai/DataAgent/development_doc/mock_data.csv"
    print(f"Loading data from: {file_path}")
    
    try:
        df = pd.read_csv(file_path)
        print(f"Data loaded successfully. Shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        
        # Convert timestamp to datetime (UTC) - handle mixed formats
        print("Converting timestamps to datetime format...")
        df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed', utc=True)
        
        # Clean up country and timezone fields (remove quotes)
        df['country'] = df['country'].astype(str).str.strip().str.strip('"')
        df['timezone'] = df['timezone'].astype(str).str.strip().str.strip('"')
        
        print("Data preprocessing completed successfully!")
        return df
        
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        raise

class DataProcessor:
    """
    Data processor that saves event-time pairs as dictionary structures
    """
    
    def __init__(self, db_path="event_analysis.db"):
        self.db_path = db_path
        self.engine = create_engine(f'sqlite:///{db_path}')
        print(f"Database initialized: {db_path}")
    
    def process_device_events_as_dict(self):
        """
        Process events for each device_id as dictionary data structures
        Filters: US users only, remove clicked users
        Save event-time pairs as JSON dictionary for each device
        """
        print("=" * 60)
        print("   DEVICE EVENTS DICTIONARY PROCESSING")
        print("=" * 60)
        
        try:
            # Step 1: Load raw data
            print("\n🔄 Loading Raw Data")
            raw_data = load_data()
            print(f"✅ Raw data loaded: {len(raw_data)} rows")
            
            # Step 2: Apply filters
            print("\n🔄 Applying Filters")
            
            # Filter 1: Keep only US users
            us_data = raw_data[raw_data['country'] == 'United States'].copy()
            print(f"✅ US users filter: {len(us_data)} rows (removed {len(raw_data) - len(us_data)} non-US rows)")
            
            # Filter 2: Remove users who have any click events
            print("🔄 Identifying users with click events...")
            
            # Find device_ids that have click events (case-insensitive search for 'click')
            click_devices = us_data[us_data['event'].str.contains('click_foru_sekai_card', case=False, na=False)]['device_id'].unique()
            print(f"📊 Found {len(click_devices)} devices with click events")
            
            # Remove all rows for devices that have any click events
            filtered_data = us_data[~us_data['device_id'].isin(click_devices)].copy()
            print(f"✅ Click filter: {len(filtered_data)} rows (removed {len(us_data) - len(filtered_data)} rows from clicking devices)")
            
            if len(filtered_data) == 0:
                print("⚠️  No data remaining after filters!")
                return None
            
            # Step 3: Group by device_id and create dictionary structures
            print("\n🔄 Creating Event-Time Dictionary Structures by Device ID")
            
            # Sort by device_id and timestamp for proper ordering
            filtered_data_sorted = filtered_data.sort_values(['device_id', 'timestamp']).copy()
            
            device_data = []
            
            for device_id, group in filtered_data_sorted.groupby('device_id'):
                # Get device information (use first row for device attributes)
                device_info = group.iloc[0].to_dict()
                
                # Create event-time pairs dictionary
                event_time_pairs = []
                for _, row in group.iterrows():
                    event_time_pair = {
                        'event': row['event'],
                        'timestamp': row['timestamp'].isoformat(),
                        'sequence': len(event_time_pairs) + 1,
                        'hour': row['timestamp'].hour,
                        'day_of_week': row['timestamp'].day_name(),
                        'date': row['timestamp'].date().isoformat()
                    }
                    event_time_pairs.append(event_time_pair)
                
                # Create complete device record with all original columns
                device_record = {
                    'device_id': device_id,
                    'event': device_info['event'],  # Keep original column
                    'timestamp': device_info['timestamp'].isoformat(),  # Keep original column
                    'uuid': device_info['uuid'],
                    'distinct_id': device_info['distinct_id'],
                    'country': device_info['country'],
                    'timezone': device_info['timezone'],
                    'newDevice': device_info['newDevice'],
                    'event_time_pairs': json.dumps(event_time_pairs),  # Dictionary as JSON
                    'total_events': len(event_time_pairs),
                    'first_event_time': event_time_pairs[0]['timestamp'],
                    'last_event_time': event_time_pairs[-1]['timestamp'],
                    'event_types': json.dumps(list(group['event'].unique())),  # List of unique events
                    'time_span_hours': (group['timestamp'].max() - group['timestamp'].min()).total_seconds() / 3600
                }
                
                device_data.append(device_record)
            
            # Convert to DataFrame
            device_df = pd.DataFrame(device_data)
            
            print(f"✅ Dictionary structures created: {len(device_df)} devices with event-time pairs")
            print(f"📊 Events per device: min={device_df['total_events'].min()}, max={device_df['total_events'].max()}, avg={device_df['total_events'].mean():.1f}")
            
            # Step 4: Save to database
            print("\n🔄 Saving Device Event Dictionaries to Database")
            device_df.to_sql('device_event_dictionaries', self.engine, if_exists='replace', index=False)
            print(f"✅ Device dictionaries saved to database: {len(device_df)} devices")
            
            # Step 5: Create summary info
            print("\n🔄 Creating Summary Information")
            self._create_dict_summary(device_df, len(raw_data), len(us_data), len(click_devices))
            
            print("\n" + "=" * 60)
            print("✅ DEVICE EVENTS DICTIONARY PROCESSING COMPLETED")
            print("✅ EVENT-TIME PAIRS SAVED AS DICTIONARIES")
            print("   (US users only, no click events, all original columns preserved)")
            print("=" * 60)
            
            return device_df
            
        except Exception as e:
            print(f"\n❌ Error in processing: {str(e)}")
            raise
    
    def _create_dict_summary(self, device_df, total_raw_data, total_us_data, total_click_devices):
        """Create summary statistics for device event dictionaries"""
        
        # Calculate summary statistics
        total_devices = len(device_df)
        total_events = device_df['total_events'].sum()
        
        # Parse event types to get unique events across all devices
        all_event_types = set()
        for event_types_json in device_df['event_types']:
            event_types = json.loads(event_types_json)
            all_event_types.update(event_types)
        
        unique_events = len(all_event_types)
        avg_events_per_device = device_df['total_events'].mean()
        avg_time_span = device_df['time_span_hours'].mean()
        
        # Create summary dataframe
        summary_data = []
        
        # Data volume metrics
        summary_data.extend([
            {'metric': 'total_raw_rows', 'value': total_raw_data, 'category': 'data_volume'},
            {'metric': 'total_us_rows', 'value': total_us_data, 'category': 'data_volume'},
            {'metric': 'devices_with_clicks', 'value': total_click_devices, 'category': 'filtering'},
            {'metric': 'final_unique_devices', 'value': total_devices, 'category': 'device_stats'},
            {'metric': 'total_events_processed', 'value': total_events, 'category': 'device_stats'},
            {'metric': 'unique_event_types', 'value': unique_events, 'category': 'event_stats'},
            {'metric': 'avg_events_per_device', 'value': round(avg_events_per_device, 2), 'category': 'engagement'},
            {'metric': 'avg_time_span_hours', 'value': round(avg_time_span, 2), 'category': 'engagement'}
        ])
        
        # Device distribution by event count
        single_event_devices = (device_df['total_events'] == 1).sum()
        multi_event_devices = (device_df['total_events'] > 1).sum()
        highly_active_devices = (device_df['total_events'] >= 10).sum()
        
        summary_data.extend([
            {'metric': 'single_event_devices', 'value': single_event_devices, 'category': 'device_distribution'},
            {'metric': 'multi_event_devices', 'value': multi_event_devices, 'category': 'device_distribution'},
            {'metric': 'highly_active_devices', 'value': highly_active_devices, 'category': 'device_distribution'}
        ])
        
        summary_df = pd.DataFrame(summary_data)
        
        # Save summary stats to database
        summary_df.to_sql('device_dict_summary', self.engine, if_exists='replace', index=False)
        print(f"✅ Summary statistics saved: {len(summary_df)} metrics")
    
    def get_database_info(self):
        """Get information about tables in database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            
            print(f"\n📁 Database: {self.db_path}")
            print("Tables created:")
            
            for table in tables:
                table_name = table[0]
                cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                count = cursor.fetchone()[0]
                print(f"  • {table_name}: {count:,} rows")

def main():
    """Main function to run device events dictionary processing"""
    print("Starting Device Events Dictionary Processing...")
    print("This will create event-time pair dictionaries for each device_id")
    
    # Create processor and run pipeline
    processor = DataProcessor()
    result = processor.process_device_events_as_dict()
    
    if result is not None:
        processor.get_database_info()
        print("\n🎉 Device events dictionary processing completed!")
        print("📊 Data ready for further processing from database")
        
        # Show sample of processed data
        print(f"\nSample of processed data:")
        sample_cols = ['device_id', 'country', 'timezone', 'total_events', 'first_event_time', 'time_span_hours']
        print(result[sample_cols].head())
        
        # Show example of event-time pairs dictionary
        if len(result) > 0:
            print(f"\nExample event-time pairs dictionary for device {result.iloc[0]['device_id']}:")
            example_pairs = json.loads(result.iloc[0]['event_time_pairs'])
            for i, pair in enumerate(example_pairs[:3]):  # Show first 3 events
                print(f"  Event {i+1}: {pair}")
            if len(example_pairs) > 3:
                print(f"  ... and {len(example_pairs) - 3} more events")
        
        # Show unique device count
        unique_devices = len(result)
        total_events = result['total_events'].sum()
        print(f"\nProcessed: {total_events:,} events from {unique_devices:,} unique devices")
    else:
        print("\n❌ Processing failed!")

if __name__ == "__main__":
    main() 