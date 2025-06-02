#!/usr/bin/env python3
"""
Main Entry Point for User Behavior Analysis Tools
Demonstrates the two-step function architecture
"""

import os
import sys

# Add the current directory to the path to import modules
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from filters.user_filters import analyze_day1_returning_users,analyze_users_without_day1_return
from core.analyzer import UserBehaviorAnalyzer
from analysis.event_analysis import get_users_with_search_terms_percentage
from utils.export_utils import export_results

def main(args=None):
    """
    Main function demonstrating the two-step function architecture
    """
    
    # Default database path
    db_path = "/Users/shuhangge/Desktop/my_projects/Sekai/DataAgent/DataProcess/event_analysis_fake.db"
    
    print("🚀 User Behavior Analysis Tools - Two-Step Architecture")
    print("=" * 60)
    
    print("\n📖 ARCHITECTURE:")
    print("Step 1: FILTER FUNCTIONS - Take database, return filtered data")
    print("Step 2: ANALYSIS FUNCTIONS - Take filtered data, perform analysis")
    
    # Check if database exists
    if not os.path.exists(db_path):
        print(f"\n❌ Database not found: {db_path}")
        print("Please run the data processing pipeline first.")
        return
    
    print(f"\n✅ Database found: {db_path}")
    print("\n📋 STEP 1 - FILTER FUNCTIONS:")
    print("• analyze_day1_returning_users(db_path)")
    print("• analyze_users_without_day1_return(db_path)")  
    print("• filter_by_location(db_path, country, timezone)")
    print("• get_users_by_device_ids(db_path, device_ids)")
    
    print("\n📊 STEP 2 - ANALYSIS FUNCTIONS:")
    print("• get_event_statistics(filtered_data)")
    print("• filter_devices_by_events(filtered_data, min_events, max_events)")
    print("• plot_event_per_user(filtered_data, ...)")
    print("• plot_event_counts(filtered_data, top_n, ...)")
    print("• find_frequent_events(filtered_data, ...)")
    print("• get_events_by_popularity(filtered_data, ascending)")
    print("• get_users_with_search_terms_percentage(filtered_data, search_terms, case_sensitive)")
    
    print("\n🔧 UTILITY FUNCTIONS:")
    print("• extract_device_ids(analysis_result)")
    print("• extract_user_data(analysis_result)") 
    print("• export_results(data, filename, format_type)")
    print("• export_filtered_data(filtered_data, filename, format_type)")
    print("• export_analysis_results(analysis_result, filename, format_type)")
    
    print("\n📚 For detailed documentation and examples:")
    print("See Tools/README_check_data.md")
    
    print("\n✅ Tools ready for use!")
    
    # Example usage
    print("\n🔄 Running example analysis...")
    try:
        print('start filter')
        # returning_users_day1 = analyze_day1_returning_users(db_path)
        returning_users_noday1= analyze_users_without_day1_return(db_path)
        print('got all users with day1 return')
        
        # # Export the filtered data
        analyzer = UserBehaviorAnalyzer(db_path)
        analyzer.export_analysis_results(returning_users_noday1, 'returning_users_data', 'csv')
        # # print(f"📊 Analyzed {len(returning_users)} returning users")
        # result1 = get_users_with_search_terms_percentage(returning_users_day1, search_terms = ['search'])
        # result2 = get_users_with_search_terms_percentage(returning_users_noday1, search_terms = ['search'])

        # export_results(result1, 'search_percentage_in_day1users_results.json')
        # export_results(result2, 'search_percentage_in_noday1users_results.json')

    except Exception as e:
        print(f"⚠️  Example analysis failed: {e}")

if __name__ == "__main__":
    main() 