#!/usr/bin/env python3
"""
Simple HTTP server for the browser physics simulation
Run this script to serve the simulation locally
"""

import http.server
import socketserver
import webbrowser
import os
import sys

PORT = 8000

class MyHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        # Add CORS headers for local development
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()

def main():
    # Change to the directory containing this script
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    print(f"🚀 Starting Physics Simulation Server...")
    print(f"📁 Serving from: {os.getcwd()}")
    print(f"🌐 Server running at: http://localhost:{PORT}")
    print(f"🎯 Opening simulation in browser...")
    
    try:
        with socketserver.TCPServer(("", PORT), MyHTTPRequestHandler) as httpd:
            # Open browser automatically
            webbrowser.open(f'http://localhost:{PORT}')
            
            print(f"✅ Server started successfully!")
            print(f"🛑 Press Ctrl+C to stop the server")
            
            httpd.serve_forever()
    except KeyboardInterrupt:
        print(f"\n🛑 Server stopped by user")
    except OSError as e:
        if e.errno == 48:  # Address already in use
            print(f"❌ Port {PORT} is already in use. Try a different port:")
            print(f"   python3 serve.py --port 8001")
        else:
            print(f"❌ Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--port":
        try:
            PORT = int(sys.argv[2])
        except (IndexError, ValueError):
            print("❌ Invalid port number")
            sys.exit(1)
    
    main()
