"""
Streaming Server
================

FastAPI server that streams GPU-rendered frames to browser via WebSocket.
Also provides LLM chat API and simulation control.
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import asyncio
import json
from typing import Dict, List
import time

app = FastAPI(title="iVHL Hierarchical Dynamics Monitor")


# Global state
class SimulationState:
    def __init__(self):
        self.running = False
        self.current_step = 0
        self.total_steps = 0
        self.hierarchy = None
        self.renderer = None
        self.llm_notes = []
        self.metrics_history = []


state = SimulationState()


@app.get("/")
async def root():
    """Serve main monitoring page."""
    return HTMLResponse(content=get_monitor_html(), status_code=200)


@app.websocket("/ws/frames")
async def websocket_frames(websocket: WebSocket):
    """
    WebSocket endpoint for streaming GPU-rendered frames.

    Client receives JPEG frames at ~30 FPS.
    """
    await websocket.accept()

    try:
        while True:
            if state.running and state.renderer and state.hierarchy:
                # Render current frame on GPU
                jpeg_bytes = state.renderer.create_animation_frame(
                    state.hierarchy,
                    state.current_step,
                    state.total_steps
                )

                # Send to client
                await websocket.send_bytes(jpeg_bytes)

                # Target 30 FPS
                await asyncio.sleep(1/30)
            else:
                # Send idle message
                await websocket.send_text(json.dumps({
                    'type': 'status',
                    'message': 'Simulation not running'
                }))
                await asyncio.sleep(1)

    except WebSocketDisconnect:
        print("Client disconnected from frame stream")


@app.websocket("/ws/metrics")
async def websocket_metrics(websocket: WebSocket):
    """
    WebSocket endpoint for streaming metrics data.
    """
    await websocket.accept()

    try:
        while True:
            if state.running:
                metrics = {
                    'step': state.current_step,
                    'total_steps': state.total_steps,
                    'progress': state.current_step / max(state.total_steps, 1),
                    'running': state.running,
                }

                # Add hierarchy info if available
                if state.hierarchy:
                    metrics['layers'] = state.hierarchy.get_all_layer_info()

                await websocket.send_text(json.dumps(metrics))

            await asyncio.sleep(0.1)  # Update 10 times per second

    except WebSocketDisconnect:
        print("Client disconnected from metrics stream")


@app.websocket("/ws/llm-commentary")
async def websocket_llm_commentary(websocket: WebSocket):
    """
    WebSocket endpoint for streaming LLM notes and commentary.
    """
    await websocket.accept()

    try:
        last_note_count = 0

        while True:
            # Check for new notes
            if len(state.llm_notes) > last_note_count:
                new_notes = state.llm_notes[last_note_count:]
                for note in new_notes:
                    await websocket.send_text(json.dumps(note))
                last_note_count = len(state.llm_notes)

            await asyncio.sleep(0.5)

    except WebSocketDisconnect:
        print("Client disconnected from LLM commentary")


@app.post("/api/llm/chat")
async def chat_with_llm(message: Dict):
    """
    Chat with embedded LLM about simulation.

    Expects: {"message": "user question"}
    Returns: {"response": "LLM answer"}
    """
    user_message = message.get("message", "")

    # TODO: Integrate with vLLM server running Qwen2.5-2B
    # For now, return placeholder
    response = f"LLM would respond to: {user_message}"

    return {"response": response}


@app.post("/api/simulation/start")
async def start_simulation(config: Dict):
    """Start simulation with given configuration."""
    state.running = True
    state.current_step = 0
    state.total_steps = config.get("timesteps", 100)

    return {"status": "started", "config": config}


@app.post("/api/simulation/pause")
async def pause_simulation():
    """Pause running simulation."""
    state.running = False
    return {"status": "paused"}


@app.post("/api/simulation/resume")
async def resume_simulation():
    """Resume paused simulation."""
    state.running = True
    return {"status": "resumed"}


@app.get("/api/simulation/status")
async def get_simulation_status():
    """Get current simulation status."""
    return {
        "running": state.running,
        "step": state.current_step,
        "total_steps": state.total_steps,
        "notes_count": len(state.llm_notes)
    }


def get_monitor_html() -> str:
    """Generate monitoring page HTML."""
    return """
<!DOCTYPE html>
<html>
<head>
    <title>iVHL Hierarchical Dynamics Monitor</title>
    <style>
        body {
            margin: 0;
            padding: 0;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: #0a0a0a;
            color: #e0e0e0;
        }
        .container {
            display: grid;
            grid-template-columns: 2fr 1fr;
            grid-template-rows: auto 1fr;
            height: 100vh;
            gap: 10px;
            padding: 10px;
        }
        .header {
            grid-column: 1 / -1;
            background: #1a1a1a;
            padding: 15px;
            border-radius: 8px;
        }
        .viz-panel {
            background: #1a1a1a;
            border-radius: 8px;
            padding: 10px;
            display: flex;
            flex-direction: column;
        }
        .chat-panel {
            background: #1a1a1a;
            border-radius: 8px;
            padding: 15px;
            display: flex;
            flex-direction: column;
        }
        #frame-display {
            width: 100%;
            height: 100%;
            object-fit: contain;
            border-radius: 8px;
        }
        .controls {
            display: flex;
            gap: 10px;
            margin-top: 10px;
        }
        button {
            background: #2a7fff;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
            font-size: 14px;
        }
        button:hover {
            background: #1a5fdf;
        }
        .progress-bar {
            width: 100%;
            height: 8px;
            background: #333;
            border-radius: 4px;
            overflow: hidden;
            margin: 10px 0;
        }
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #2a7fff, #00d9ff);
            width: 0%;
            transition: width 0.3s;
        }
        .metrics {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 10px;
            margin-top: 10px;
        }
        .metric {
            background: #252525;
            padding: 10px;
            border-radius: 5px;
        }
        .metric-label {
            font-size: 12px;
            color: #888;
        }
        .metric-value {
            font-size: 20px;
            font-weight: bold;
            color: #00d9ff;
        }
        .llm-notes {
            flex: 1;
            overflow-y: auto;
            margin: 10px 0;
            padding: 10px;
            background: #252525;
            border-radius: 5px;
        }
        .note {
            margin: 10px 0;
            padding: 10px;
            background: #333;
            border-left: 3px solid #2a7fff;
            border-radius: 3px;
        }
        .chat-input {
            display: flex;
            gap: 10px;
        }
        input {
            flex: 1;
            padding: 10px;
            background: #252525;
            border: 1px solid #444;
            color: #e0e0e0;
            border-radius: 5px;
        }
        h1 { margin: 0; color: #00d9ff; }
        h2 { margin-top: 0; font-size: 18px; }
        .status-live { color: #00ff88; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔬 iVHL Hierarchical Information Dynamics</h1>
            <p style="margin: 5px 0; color: #888;">Real-time GPU-accelerated simulation monitoring</p>
        </div>

        <div class="viz-panel">
            <h2>3D Visualization <span class="status-live">● LIVE</span></h2>
            <img id="frame-display" src="" alt="Waiting for frames...">

            <div class="progress-bar">
                <div class="progress-fill" id="progress"></div>
            </div>

            <div class="metrics">
                <div class="metric">
                    <div class="metric-label">Step</div>
                    <div class="metric-value" id="step">0</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Progress</div>
                    <div class="metric-value" id="progress-pct">0%</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Notes</div>
                    <div class="metric-value" id="notes-count">0</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Status</div>
                    <div class="metric-value" id="status">Idle</div>
                </div>
            </div>

            <div class="controls">
                <button onclick="pauseSim()">⏸️ Pause</button>
                <button onclick="resumeSim()">▶️ Resume</button>
                <button onclick="generateWhitepaper()">📄 Generate Whitepaper</button>
            </div>
        </div>

        <div class="chat-panel">
            <h2>🤖 AI Assistant (Qwen 2.5B)</h2>

            <div class="llm-notes" id="notes">
                <div style="color: #888; text-align: center;">Waiting for simulation...</div>
            </div>

            <div class="chat-input">
                <input type="text" id="chat-input" placeholder="Ask the LLM about the simulation...">
                <button onclick="sendChat()">Send</button>
            </div>
        </div>
    </div>

    <script>
        // WebSocket for frames
        const wsFrames = new WebSocket(`ws://${window.location.host}/ws/frames`);
        wsFrames.binaryType = 'arraybuffer';

        wsFrames.onmessage = (event) => {
            if (event.data instanceof ArrayBuffer) {
                // Convert bytes to image
                const blob = new Blob([event.data], {type: 'image/jpeg'});
                const url = URL.createObjectURL(blob);
                document.getElementById('frame-display').src = url;
            }
        };

        // WebSocket for metrics
        const wsMetrics = new WebSocket(`ws://${window.location.host}/ws/metrics`);
        wsMetrics.onmessage = (event) => {
            const data = JSON.parse(event.data);
            document.getElementById('step').textContent = data.step || 0;
            document.getElementById('progress-pct').textContent =
                Math.round((data.progress || 0) * 100) + '%';
            document.getElementById('progress').style.width =
                ((data.progress || 0) * 100) + '%';
            document.getElementById('status').textContent =
                data.running ? 'Running' : 'Paused';
        };

        // WebSocket for LLM commentary
        const wsLLM = new WebSocket(`ws://${window.location.host}/ws/llm-commentary`);
        wsLLM.onmessage = (event) => {
            const note = JSON.parse(event.data);
            const notesDiv = document.getElementById('notes');
            const noteEl = document.createElement('div');
            noteEl.className = 'note';
            noteEl.innerHTML = `<strong>Step ${note.step}:</strong> ${note.content}`;
            notesDiv.appendChild(noteEl);
            notesDiv.scrollTop = notesDiv.scrollHeight;

            document.getElementById('notes-count').textContent =
                notesDiv.children.length;
        };

        async function pauseSim() {
            await fetch('/api/simulation/pause', {method: 'POST'});
        }

        async function resumeSim() {
            await fetch('/api/simulation/resume', {method: 'POST'});
        }

        async function sendChat() {
            const input = document.getElementById('chat-input');
            const message = input.value;
            if (!message) return;

            const response = await fetch('/api/llm/chat', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({message})
            });

            const data = await response.json();
            const notesDiv = document.getElementById('notes');
            const noteEl = document.createElement('div');
            noteEl.className = 'note';
            noteEl.style.borderLeftColor = '#00ff88';
            noteEl.innerHTML = `<strong>You:</strong> ${message}<br><strong>LLM:</strong> ${data.response}`;
            notesDiv.appendChild(noteEl);

            input.value = '';
        }

        async function generateWhitepaper() {
            alert('Whitepaper generation not yet implemented!');
        }
    </script>
</body>
</html>
    """


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
