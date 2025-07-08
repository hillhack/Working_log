import csv
import time
from datetime import datetime

def setup_logger(log_file):
    with open(log_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Date', 'Start Time', 'End Time', 'Duration (seconds)', 'Status'])

def log_session(log_file, start_timestamp, end_timestamp, duration, status):
    with open(log_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            datetime.fromtimestamp(start_timestamp).date(), 
            datetime.fromtimestamp(start_timestamp).strftime('%H:%M:%S'),
            datetime.fromtimestamp(end_timestamp).strftime('%H:%M:%S'),
            round(duration), 
            status
        ])