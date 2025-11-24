"""
Simple web dashboard to monitor experiment progress remotely.

Run this on the server PC to serve a web interface that shows:
- Experiment status and progress
- Recent log messages
- Recent plots
- Error messages

Usage:
    python monitor_dashboard.py --status-dir /path/to/status --port 8888
    Then access http://server-ip:8888 in a browser
"""

import os
import json
import argparse
from pathlib import Path
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import threading
import time


class MonitorHandler(BaseHTTPRequestHandler):
    """HTTP handler for the monitoring dashboard."""
    
    def __init__(self, status_dir, log_dir, plots_dir, *args, **kwargs):
        self.status_dir = Path(status_dir)
        self.log_dir = Path(log_dir) if log_dir else None
        self.plots_dir = Path(plots_dir) if plots_dir else None
        super().__init__(*args, **kwargs)
    
    def do_GET(self):
        """Handle GET requests."""
        parsed_path = urlparse(self.path)
        path = parsed_path.path
        
        if path == '/' or path == '/index.html':
            self.send_html_dashboard()
        elif path == '/status':
            self.send_json_status()
        elif path == '/logs':
            self.send_logs()
        elif path.startswith('/plot/'):
            self.send_plot(path[6:])  # Remove '/plot/' prefix
        else:
            self.send_error(404, "Not Found")
    
    def send_html_dashboard(self):
        """Send the main HTML dashboard."""
        html = self.get_dashboard_html()
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(html.encode('utf-8'))
    
    def send_json_status(self):
        """Send experiment status as JSON."""
        status = self.get_latest_status()
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(status, indent=2).encode('utf-8'))
    
    def send_logs(self):
        """Send recent log entries."""
        logs = self.get_recent_logs()
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(logs, indent=2).encode('utf-8'))
    
    def send_plot(self, filename):
        """Send a plot image."""
        if not self.plots_dir:
            self.send_error(404, "Plots directory not configured")
            return
        
        plot_path = self.plots_dir / filename
        if not plot_path.exists():
            self.send_error(404, f"Plot not found: {filename}")
            return
        
        try:
            with open(plot_path, 'rb') as f:
                image_data = f.read()
            
            # Determine content type from extension
            ext = plot_path.suffix.lower()
            content_type = {
                '.png': 'image/png',
                '.jpg': 'image/jpeg',
                '.jpeg': 'image/jpeg',
                '.pdf': 'application/pdf',
                '.svg': 'image/svg+xml'
            }.get(ext, 'application/octet-stream')
            
            self.send_response(200)
            self.send_header('Content-type', content_type)
            self.end_headers()
            self.wfile.write(image_data)
        except Exception as e:
            self.send_error(500, f"Error reading plot: {str(e)}")
    
    def get_latest_status(self):
        """Get the latest status from status files."""
        status_files = list(self.status_dir.glob("*_status.json"))
        if not status_files:
            return {'error': 'No status files found'}
        
        # Get most recently modified
        latest_file = max(status_files, key=lambda p: p.stat().st_mtime)
        
        try:
            with open(latest_file, 'r') as f:
                status = json.load(f)
            status['status_file'] = str(latest_file)
            return status
        except Exception as e:
            return {'error': f'Error reading status: {str(e)}'}
    
    def get_recent_logs(self, n=100):
        """Get recent log entries."""
        if not self.log_dir:
            return {'logs': []}
        
        log_files = sorted(self.log_dir.glob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not log_files:
            return {'logs': []}
        
        # Read from most recent log file
        latest_log = log_files[0]
        try:
            with open(latest_log, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            # Return last n lines
            return {'logs': lines[-n:], 'log_file': str(latest_log)}
        except Exception as e:
            return {'error': f'Error reading logs: {str(e)}'}
    
    def get_dashboard_html(self):
        """Generate the HTML dashboard."""
        return f"""<!DOCTYPE html>
<html>
<head>
    <title>Experiment Monitor</title>
    <meta http-equiv="refresh" content="5">
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #333;
            border-bottom: 2px solid #4CAF50;
            padding-bottom: 10px;
        }}
        .status-box {{
            background: #f9f9f9;
            border: 1px solid #ddd;
            border-radius: 4px;
            padding: 15px;
            margin: 10px 0;
        }}
        .progress-bar {{
            width: 100%;
            height: 30px;
            background-color: #e0e0e0;
            border-radius: 15px;
            overflow: hidden;
            margin: 10px 0;
        }}
        .progress-fill {{
            height: 100%;
            background-color: #4CAF50;
            transition: width 0.3s ease;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: bold;
        }}
        .status-running {{ background-color: #4CAF50; }}
        .status-completed {{ background-color: #2196F3; }}
        .status-error {{ background-color: #f44336; }}
        .status-paused {{ background-color: #FF9800; }}
        .log-container {{
            max-height: 400px;
            overflow-y: auto;
            background: #1e1e1e;
            color: #d4d4d4;
            padding: 10px;
            border-radius: 4px;
            font-family: 'Courier New', monospace;
            font-size: 12px;
        }}
        .log-entry {{
            margin: 2px 0;
            padding: 2px 5px;
        }}
        .log-info {{ color: #d4d4d4; }}
        .log-warning {{ color: #ffa500; }}
        .log-error {{ color: #f44336; }}
        .error-box {{
            background: #ffebee;
            border-left: 4px solid #f44336;
            padding: 10px;
            margin: 10px 0;
        }}
        .timestamp {{
            color: #888;
            font-size: 0.9em;
        }}
        .plots-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}
        .plot-item {{
            border: 1px solid #ddd;
            border-radius: 4px;
            padding: 10px;
            text-align: center;
        }}
        .plot-item img {{
            max-width: 100%;
            height: auto;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🔬 Experiment Monitor</h1>
        
        <div id="status-container">
            <div class="status-box">
                <h2>Status</h2>
                <div id="status-info">Loading...</div>
                <div class="progress-bar">
                    <div id="progress-fill" class="progress-fill" style="width: 0%">0%</div>
                </div>
            </div>
        </div>
        
        <div class="status-box">
            <h2>Recent Messages</h2>
            <div id="messages-container" class="log-container">Loading...</div>
        </div>
        
        <div id="errors-container" style="display: none;">
            <div class="error-box">
                <h3>Errors</h3>
                <div id="errors-list"></div>
            </div>
        </div>
        
        <div class="status-box">
            <h2>Recent Logs</h2>
            <div id="logs-container" class="log-container">Loading...</div>
        </div>
    </div>
    
    <script>
        function updateStatus() {{
            fetch('/status')
                .then(response => response.json())
                .then(data => {{
                    if (data.error) {{
                        document.getElementById('status-info').innerHTML = 
                            '<p style="color: red;">' + data.error + '</p>';
                        return;
                    }}
                    
                    const statusDiv = document.getElementById('status-info');
                    const progressFill = document.getElementById('progress-fill');
                    const progressPercent = data.progress_percent || 0;
                    
                    statusDiv.innerHTML = `
                        <p><strong>Experiment:</strong> ${{data.experiment_name || 'Unknown'}}</p>
                        <p><strong>Status:</strong> <span class="status-${{data.status}}">${{data.status}}</span></p>
                        <p><strong>Current Step:</strong> ${{data.current_step || 'N/A'}} / ${{data.total_steps || 'N/A'}}</p>
                        <p class="timestamp"><strong>Last Update:</strong> ${{data.last_update || 'N/A'}}</p>
                    `;
                    
                    progressFill.style.width = progressPercent + '%';
                    progressFill.textContent = progressPercent.toFixed(1) + '%';
                    progressFill.className = 'progress-fill status-' + (data.status || 'running');
                    
                    // Update messages
                    const messagesDiv = document.getElementById('messages-container');
                    if (data.messages && data.messages.length > 0) {{
                        const recentMessages = data.messages.slice(-20).reverse();
                        messagesDiv.innerHTML = recentMessages.map(msg => `
                            <div class="log-entry log-${{msg.level || 'info'}}">
                                <span class="timestamp">[${{msg.timestamp}}]</span> ${{msg.message}}
                            </div>
                        `).join('');
                    }}
                    
                    // Update errors
                    if (data.errors && data.errors.length > 0) {{
                        document.getElementById('errors-container').style.display = 'block';
                        document.getElementById('errors-list').innerHTML = data.errors.map(err => `
                            <div class="log-entry log-error">
                                <span class="timestamp">[${{err.timestamp}}]</span> ${{err.message}}
                            </div>
                        `).join('');
                    }}
                }})
                .catch(error => {{
                    console.error('Error fetching status:', error);
                }});
        }}
        
        function updateLogs() {{
            fetch('/logs')
                .then(response => response.json())
                .then(data => {{
                    if (data.logs) {{
                        const logsDiv = document.getElementById('logs-container');
                        logsDiv.innerHTML = data.logs.slice(-50).map(line => `
                            <div class="log-entry log-info">${{line}}</div>
                        `).join('');
                        logsDiv.scrollTop = logsDiv.scrollHeight;
                    }}
                }})
                .catch(error => {{
                    console.error('Error fetching logs:', error);
                }});
        }}
        
        // Update every 2 seconds
        updateStatus();
        updateLogs();
        setInterval(updateStatus, 2000);
        setInterval(updateLogs, 5000);
    </script>
</body>
</html>"""
    
    def log_message(self, format, *args):
        """Override to suppress default logging."""
        pass


def create_handler_factory(status_dir, log_dir, plots_dir):
    """Factory function to create handler with custom directories."""
    def handler(*args, **kwargs):
        return MonitorHandler(status_dir, log_dir, plots_dir, *args, **kwargs)
    return handler


def run_dashboard(status_dir: str, log_dir: str = None, plots_dir: str = None, 
                  host: str = '0.0.0.0', port: int = 8888):
    """Run the monitoring dashboard server."""
    status_path = Path(status_dir)
    if not status_path.exists():
        print(f"Error: Status directory does not exist: {status_dir}")
        return
    
    handler = create_handler_factory(status_dir, log_dir, plots_dir)
    server = HTTPServer((host, port), handler)
    
    print(f"Experiment Monitor Dashboard")
    print(f"===========================")
    print(f"Status directory: {status_dir}")
    if log_dir:
        print(f"Log directory: {log_dir}")
    if plots_dir:
        print(f"Plots directory: {plots_dir}")
    print(f"Server running on http://{host}:{port}")
    print(f"Access the dashboard at: http://localhost:{port}")
    print(f"Press Ctrl+C to stop")
    print()
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        server.shutdown()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Experiment Monitoring Dashboard')
    parser.add_argument('--status-dir', type=str, required=True,
                       help='Directory containing status JSON files')
    parser.add_argument('--log-dir', type=str, default=None,
                       help='Directory containing log files')
    parser.add_argument('--plots-dir', type=str, default=None,
                       help='Directory containing plot images')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                       help='Host to bind to (default: 0.0.0.0 for all interfaces)')
    parser.add_argument('--port', type=int, default=8888,
                       help='Port to listen on (default: 8888)')
    
    args = parser.parse_args()
    run_dashboard(args.status_dir, args.log_dir, args.plots_dir, args.host, args.port)









