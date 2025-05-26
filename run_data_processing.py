#!/usr/bin/env python3
"""
Data Processing Entry Point
Creates event-time pair dictionaries for each device_id
"""

from DataProcess.data_processor import DataProcessor
import json

def main():
    """Main function to run device events dictionary processing"""
    print("Starting Device Events Dictionary Processing...")
    print("This will create event-time pair dictionaries for each device_id")
    print("(US users only, no click events, all original columns preserved)")
    
    # Create processor and run pipeline
    processor = DataProcessor()
    result = processor.process_device_events_as_dict()
    
    if result is not None:
        processor.get_database_info()
        print("\n🎉 Device events dictionary processing completed!")
        print("📊 Data ready for further processing from database")
        print("\nDatabase table created:")
        print("  • device_event_dictionaries - Each row contains a device with all its event-time pairs as JSON")
        print("  • device_dict_summary - Summary statistics")
        
        # Show sample of processed data
        print(f"\nSample of processed data:")
        sample_cols = ['device_id', 'country', 'timezone', 'total_events', 'first_event_time', 'time_span_hours']
        print(result[sample_cols].head())
        
        # Show example of event-time pairs dictionary structure
        if len(result) > 0:
            print(f"\nExample event-time pairs dictionary for device {result.iloc[0]['device_id']}:")
            example_pairs = json.loads(result.iloc[0]['event_time_pairs'])
            for i, pair in enumerate(example_pairs[:3]):  # Show first 3 events
                print(f"  Event {i+1}: {pair}")
            if len(example_pairs) > 3:
                print(f"  ... and {len(example_pairs) - 3} more events")
        
        # Show processing summary
        unique_devices = len(result)
        total_events = result['total_events'].sum()
        print(f"\nProcessing Summary:")
        print(f"  📱 Unique devices: {unique_devices:,}")
        print(f"  📊 Total events: {total_events:,}")
        print(f"  📈 Avg events per device: {total_events/unique_devices:.1f}")
        
        print("\nData Structure:")
        print("  • Each device has ALL original columns preserved")
        print("  • Event-time pairs stored as JSON dictionary in 'event_time_pairs' column")
        print("  • Each pair includes: event, timestamp, sequence, hour, day_of_week, date")
    else:
        print("\n❌ Processing failed!")

if __name__ == "__main__":
    main() 