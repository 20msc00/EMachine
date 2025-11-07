import argparse, asyncio, webbrowser, sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import httpx
import uvicorn
import simulate


async def launch(args):
    if args.personas:
        simulate.GLOBAL_CONFIG["personas"] = args.personas
    if args.companions:
        simulate.GLOBAL_CONFIG["companions"] = args.companions
    if args.runs_per_persona is not None:
        simulate.GLOBAL_CONFIG["runs_per_persona"] = args.runs_per_persona
    if args.max_conversation_rounds is not None:
        simulate.GLOBAL_CONFIG["max_conversation_rounds"] = args.max_conversation_rounds
    if args.max_parallel_simulations is not None:
        simulate.GLOBAL_CONFIG["max_parallel_simulations"] = args.max_parallel_simulations
    config = uvicorn.Config("live_dashboard.backend.server:app", host="127.0.0.1", port=args.port, log_level="warning")
    server = uvicorn.Server(config)
    task = asyncio.create_task(server.serve())
    while not server.started:
        await asyncio.sleep(0.1)
    webbrowser.open(f"http://127.0.0.1:{args.port}")
    async with httpx.AsyncClient() as client:
        await client.post(f"http://127.0.0.1:{args.port}/start", json={})
    await task


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--personas", nargs="*")
    parser.add_argument("--companions", nargs="*")
    parser.add_argument("--runs-per-persona", type=int)
    parser.add_argument("--max-conversation-rounds", type=int)
    parser.add_argument("--max-parallel-simulations", type=int)
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    asyncio.run(launch(args))


if __name__ == "__main__":
    main()
