import json
import asyncio
import traceback
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

# Import initialization functions from teleagent.main
from teleagent.main import initialize_reasoning_service, initialize_coordinator

# 1. Initialize backend services & coordinator once on startup
print("[TeleAgent Web]: Initializing backend model and coordinator...")
reasoning_service = initialize_reasoning_service()
coordinator = initialize_coordinator(reasoning_service)
print("[TeleAgent Web]: Coordinator ready.")

# 2. FastAPI App Instance
app = FastAPI(title="TeleAgent UI")


@app.get("/")
async def get_index():
    """Serves the single-file HTML frontend."""
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(
            content="<h3>index.html not found. Place index.html in the same directory as app.py</h3>",
            status_code=404
        )


@app.websocket("/ws/query")
async def websocket_query_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            raw_data = await websocket.receive_text()
            data = json.loads(raw_data)
            user_query = data.get("query", "").strip()

            if not user_query:
                continue

            await websocket.send_json({"type": "status", "message": "Executing Agent Pipeline..."})

            # Execute via the full Coordinator pipeline
            try:
                if asyncio.iscoroutinefunction(coordinator.execute):
                    response = await coordinator.execute(user_query)
                else:
                    response = await asyncio.to_thread(coordinator.execute, user_query)
            except Exception as e:
                print(f"[Error in Coordinator Execution]: {traceback.format_exc()}")
                await websocket.send_json({
                    "type": "result",
                    "intent": "error",
                    "content": f"**Execution Error**: `{str(e)}`\nCheck backend logs.",
                    "metadata": {"agents": [], "inference_time_ms": 0, "tokens": 0, "model": "N/A"}
                })
                continue

            # Extract metrics from coordinator results safely
            executed_agents = []
            inference_time = 0.0
            tokens = 0
            model_name = "Qwen/Qwen2.5-1.5B-Instruct"

            for result in getattr(response, "results", []):
                executed_agents.append(result.agent)
                resp = result.response
                metadata = resp.get("metrics", {}) if isinstance(resp, dict) else getattr(resp, "metadata", {})

                if isinstance(metadata, dict):
                    inference_time = max(inference_time, metadata.get("inference_time_ms", 0.0))
                    tokens += metadata.get("tokens_generated", 0)
                    model_name = metadata.get("model_name", model_name)
                else:
                    inference_time = max(inference_time, getattr(metadata, "inference_time_ms", 0.0))
                    tokens += getattr(metadata, "tokens_generated", 0)
                    model_name = getattr(metadata, "model_name", model_name)

            formatted_output = str(getattr(response, "response", ""))
            intent = str(getattr(response, "intent", "general"))

            # Send back response to UI
            await websocket.send_json({
                "type": "result",
                "intent": intent,
                "content": formatted_output,
                "metadata": {
                    "agents": executed_agents,
                    "inference_time_ms": inference_time,
                    "tokens": tokens,
                    "model": model_name,
                }
            })

    except WebSocketDisconnect:
        print("[WebSocket]: Client disconnected naturally.")
    except Exception as e:
        print(f"[WebSocket Error]: {e}")
