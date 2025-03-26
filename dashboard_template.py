"""
This module contains HTML templates for the dashboard.
"""

def get_dashboard_css():
    """Return the CSS styles for the dashboard."""
    return """
        body { 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
            margin: 20px; 
            background-color: #f9f9f9;
            color: #333;
        }
        h1, h2 { 
            color: #2c3e50; 
            margin-bottom: 20px;
        }
        h1 {
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            background-color: white;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        table { 
            border-collapse: collapse; 
            width: 100%; 
            margin-top: 20px; 
            margin-bottom: 40px;
            font-size: 14px;
        }
        th, td { 
            padding: 12px 15px; 
            text-align: left; 
            border-bottom: 1px solid #e1e1e1; 
        }
        th { 
            background-color: #3498db; 
            color: white; 
            font-weight: bold;
            position: sticky;
            top: 0;
        }
        tr:nth-child(even) { background-color: #f7f7f7; }
        tr:hover { background-color: #f1f1f1; }
        .model-name {
            font-weight: bold;
            font-size: 15px;
        }
        .other-names { 
            color: #666; 
            font-size: 0.9em;
            font-style: italic;
            display: block;
            margin-top: 4px;
        }
        .running {
            background-color: #e8f5e9;
        }
        .status-true { 
            color: #2ecc71; 
            font-weight: bold; 
        }
        .status-false { 
            color: #e74c3c; 
        }
        .status-running {
            display: inline-block;
            padding: 4px 8px;
            border-radius: 4px;
            background-color: #2ecc71;
            color: white;
            font-weight: bold;
        }
        .status-stopped {
            display: inline-block;
            padding: 4px 8px;
            border-radius: 4px;
            background-color: #95a5a6;
            color: white;
        }
        .status-exited {
            display: inline-block;
            padding: 4px 8px;
            border-radius: 4px;
            background-color: #e67e22;
            color: white;
        }
        .status-created {
            display: inline-block;
            padding: 4px 8px;
            border-radius: 4px;
            background-color: #3498db;
            color: white;
        }
        .status-restarting {
            display: inline-block;
            padding: 4px 8px;
            border-radius: 4px;
            background-color: #f1c40f;
            color: white;
        }
        .status-paused {
            display: inline-block;
            padding: 4px 8px;
            border-radius: 4px;
            background-color: #9b59b6;
            color: white;
        }
        .success-rate { font-weight: bold; }
        .high-rate { color: #2ecc71; }
        .medium-rate { color: #f39c12; }
        .low-rate { color: #e74c3c; }
        .refresh-button { 
            background-color: #3498db; 
            color: white; 
            padding: 10px 20px; 
            border: none; 
            border-radius: 4px; 
            cursor: pointer; 
            margin-top: 20px;
            font-size: 16px;
            transition: background-color 0.3s;
        }
        .refresh-button:hover { 
            background-color: #2980b9; 
        }
        .section { 
            margin-bottom: 40px; 
        }
        .last-update { 
            color: #7f8c8d; 
            font-size: 0.9em; 
            margin: 10px 0 20px 0;
        }
        .error-message { 
            color: #e74c3c; 
            margin: 10px 0; 
            padding: 10px;
            background-color: #fadbd8;
            border-radius: 4px;
        }
        .time-period-selector {
            margin: 20px 0;
        }
        .time-period-selector label {
            margin-right: 15px;
            font-weight: bold;
        }
        .time-period-buttons {
            margin-top: 10px;
            display: flex;
            gap: 10px;
        }
        .period-button {
            padding: 8px 15px;
            border-radius: 4px;
            border: 1px solid #3498db;
            background-color: #fff;
            color: #3498db;
            cursor: pointer;
            transition: all 0.3s;
            font-weight: bold;
        }
        .period-button:hover {
            background-color: #eaf2f8;
        }
        .period-button.active {
            background-color: #3498db;
            color: white;
        }
        .model-row.hidden {
            display: none;
        }
        .port-gpu-info {
            white-space: nowrap;
        }
        .port-gpu-label {
            font-weight: bold;
            color: #7f8c8d;
            margin-right: 5px;
        }
    """

def get_dashboard_js():
    """Return the JavaScript for the dashboard."""
    return """
        function refreshPage() {
            location.reload();
        }
        
        // Auto refresh every 30 mins 
        setTimeout(function() {
            refreshPage();
        }, 1800000);
        
        // Change time period
        function changeTimePeriod(period) {
            // Update active button styling
            document.querySelectorAll('.period-button').forEach(btn => {
                btn.classList.remove('active');
            });
            document.getElementById(period + '-btn').classList.add('active');
            
            // Show all rows
            document.querySelectorAll('.all-time-row, .past-hour-row, .past-day-row').forEach(row => {
                row.classList.add('hidden');
            });
            
            // Show only selected period rows
            document.querySelectorAll('.' + period + '-row').forEach(row => {
                row.classList.remove('hidden');
            });
        }
        
        // Initialize on load
        window.onload = function() {
            // Default to all-time view
            changeTimePeriod('all-time');
        }
    """

def get_dashboard_header(last_update_time):
    """Return the dashboard header HTML."""
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Llama.cpp Scheduler Dashboard</title>
        <style>
        {get_dashboard_css()}
        </style>
        <script>
        {get_dashboard_js()}
        </script>
    </head>
    <body>
        <div class="container">
            <h1>Llama.cpp Scheduler Dashboard</h1>
            <button class="refresh-button" onclick="refreshPage()">Refresh Dashboard</button>
            <p><small>Page auto-refreshes every 30 minutes</small></p>
            <p class="last-update">Last updated: {last_update_time}</p>
            
            <div class="time-period-selector">
                <label>Select Time Period:</label>
                <div class="time-period-buttons">
                    <button id="all-time-btn" class="period-button active" onclick="changeTimePeriod('all-time')">All Time</button>
                    <button id="past-hour-btn" class="period-button" onclick="changeTimePeriod('past-hour')">Past Hour</button>
                    <button id="past-day-btn" class="period-button" onclick="changeTimePeriod('past-day')">Past Day</button>
                </div>
            </div>
            
            <div class="section">
                <h2>All Models</h2>
                <table>
                    <tr>
                        <th>Model</th>
                        <th>Status</th>
                        <th>Downloaded</th>
                        <th>Total Requests</th>
                        <th>Success Rate</th>
                        <th>Requests/Min</th>
                        <th>Avg Response Time</th>
                    </tr>
    """

def get_dashboard_footer():
    """Return the dashboard footer HTML."""
    return """
                </table>
            </div>
        </div>
    </body>
    </html>
    """

def get_error_page(error_message):
    """Return an error page HTML."""
    return f"""
    <html>
    <head>
        <title>Error - Llama.cpp Scheduler Dashboard</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .error-message {{ color: red; }}
        </style>
    </head>
    <body>
        <h1>Error Loading Dashboard</h1>
        <p class="error-message">An error occurred while generating the dashboard. Please try refreshing the page.</p>
        <p>Error details: {error_message}</p>
        <button onclick="location.reload()">Refresh Page</button>
    </body>
    </html>
    """

def update_last_request_time(model_name):
    """Update the last request time for a specific model."""
    with model_lock:
        model_last_request[model_name] = time.time()
        logger.debug(f"Updated last request time for {model_name}") 