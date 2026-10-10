"""An OpenAI-style chat endpoint that always answers FINISH([]): the do-nothing agent, to measure the benchmark shortcut."""
import json
from http.server import BaseHTTPRequestHandler, HTTPServer

class H(BaseHTTPRequestHandler):
    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        body = json.dumps({"choices": [{"message": {"role": "assistant", "content": "FINISH([])"}}]}).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers(); self.wfile.write(body)
    def log_message(self, *a): pass

HTTPServer(("127.0.0.1", 8012), H).serve_forever()
