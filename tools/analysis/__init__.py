"""Event analysis functions module"""
from .event_analysis import (
    get_event_statistics,
    get_events_by_popularity,
    plot_event_per_user,
    find_frequent_events,
    filter_devices_by_events,
    plot_event_counts,
    get_users_with_search_terms_percentage
)

__all__ = [
    'get_event_statistics',
    'get_events_by_popularity',
    'plot_event_per_user',
    'find_frequent_events',
    'filter_devices_by_events',
    'plot_event_counts',
    'get_users_with_search_terms_percentage'
] 