import asyncio, json, sys
from pathlib import Path
from typing import Any

sys.path.append(str(Path(__file__).resolve().parents[2]))
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import simulate
from .live_simulator import run_live


class EventBus:
    def __init__(self):
        self.subscribers = set[Any]()

    def subscribe(self):
        queue = asyncio.Queue()
        self.subscribers.add(queue)
        return queue

    def unsubscribe(self, queue):
        self.subscribers.discard(queue)

    async def publish(self, event):
        for queue in list(self.subscribers):
            await queue.put(event)


app = FastAPI()
bus = EventBus()
current_task: asyncio.Task | None = None
stop_flag = asyncio.Event()


@app.get("/config")
async def get_config():
    return {
        "personas": simulate.GLOBAL_CONFIG["personas"],
        "companions": simulate.GLOBAL_CONFIG["companions"],
        "runs_per_persona": simulate.GLOBAL_CONFIG["runs_per_persona"],
        "max_conversation_rounds": simulate.GLOBAL_CONFIG["max_conversation_rounds"],
        "max_parallel_simulations": simulate.GLOBAL_CONFIG["max_parallel_simulations"],
        "conversation_turn_delay": simulate.GLOBAL_CONFIG["conversation_turn_delay"],
    }


@app.post("/start")
async def start_run(request: Request):
    data = await request.json() if request.headers.get("content-length") not in (None, "0") else {}
    if "personas" in data:
        simulate.GLOBAL_CONFIG["personas"] = data["personas"]
    if "companions" in data:
        simulate.GLOBAL_CONFIG["companions"] = data["companions"]
    if "runs_per_persona" in data:
        simulate.GLOBAL_CONFIG["runs_per_persona"] = data["runs_per_persona"]
    if "max_conversation_rounds" in data:
        simulate.GLOBAL_CONFIG["max_conversation_rounds"] = data["max_conversation_rounds"]
    if "max_parallel_simulations" in data:
        simulate.GLOBAL_CONFIG["max_parallel_simulations"] = data["max_parallel_simulations"]
    if "conversation_turn_delay" in data:
        simulate.GLOBAL_CONFIG["conversation_turn_delay"] = data["conversation_turn_delay"]
    global current_task
    if current_task and not current_task.done():
        return JSONResponse({"status": "running"})
    stop_flag.clear()
    personas = list(simulate.GLOBAL_CONFIG["personas"])
    companions = list(simulate.GLOBAL_CONFIG["companions"])
    runs_per_persona = simulate.GLOBAL_CONFIG["runs_per_persona"]
    current_task = asyncio.create_task(run_live(personas, companions, runs_per_persona, bus.publish, stop_flag))
    return JSONResponse({"status": "started"})


@app.post("/stop")
async def stop_run():
    stop_flag.set()
    global current_task
    if current_task and not current_task.done():
        current_task.cancel()
        try:
            await current_task
        except asyncio.CancelledError:
            pass
    current_task = None
    return JSONResponse({"status": "stopped"})


@app.websocket("/ws")
async def websocket_endpoint(socket: WebSocket):
    await socket.accept()
    queue = bus.subscribe()
    try:
        while True:
            event = await queue.get()
            await socket.send_text(json.dumps(event))
    except WebSocketDisconnect:
        pass
    finally:
        bus.unsubscribe(queue)


frontend_path = Path(__file__).resolve().parent.parent / "frontend"
app.mount("/", StaticFiles(directory=frontend_path, html=True), name="dashboard")
