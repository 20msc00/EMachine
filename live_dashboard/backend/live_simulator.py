import asyncio, itertools, uuid, random, time, sys
from pathlib import Path
from typing import Callable
sys.path.append(str(Path(__file__).resolve().parents[2]))
import httpx
import simulate


async def run_conversation(persona_file: str, companion_model: str, seed: int, publish: Callable, companion_client: httpx.AsyncClient, stop_event: asyncio.Event):
    persona_prompt = simulate.load_persona_prompt(persona_file)
    persona_history = [simulate.persona_system_message(persona_prompt)]
    companion_history = [simulate.companion_system_prompt()]
    random.seed(seed)
    first_msg, persona_usage = await simulate.call_persona(persona_prompt, persona_history, seed)
    persona_history.append({"role": "assistant", "content": first_msg["content"]})
    companion_history.append({"role": "user", "content": first_msg["content"]})
    convo_id = str(uuid.uuid4())
    persona_tokens = simulate.total_tokens(persona_usage)
    companion_tokens = 0
    persona_turns = 1
    last_assistant = None
    await publish({"type": "conversation_start", "id": convo_id, "persona": persona_file, "companion": companion_model, "seed": seed, "timestamp": time.time()})
    await publish({"type": "message", "id": convo_id, "role": "user", "content": first_msg["content"], "turn": 0})
    for turn in range(simulate.GLOBAL_CONFIG["max_conversation_rounds"]):
        if stop_event.is_set():
            await publish({"type": "conversation_end", "id": convo_id, "end_reason": "stopped", "persona_tokens": persona_tokens, "companion_tokens": companion_tokens})
            return
        assistant_msg, companion_usage = await simulate.call_companion(companion_model, companion_history, companion_client, seed)
        companion_history.append(assistant_msg)
        companion_tokens += simulate.total_tokens(companion_usage)
        await publish({"type": "message", "id": convo_id, "role": "assistant", "content": assistant_msg["content"], "turn": turn})
        if simulate.repeat_guard(last_assistant, assistant_msg["content"].strip()):
            await publish({"type": "conversation_end", "id": convo_id, "end_reason": "stall", "persona_tokens": persona_tokens, "companion_tokens": companion_tokens})
            return
        last_assistant = assistant_msg["content"].strip()
        persona_history.append({"role": "user", "content": assistant_msg["content"]})
        if (persona_turns + 1) % simulate.GLOBAL_CONFIG["persona_reminder_every"] == 0:
            persona_history.append({"role": "system", "content": "Reminder: follow your persona instructions strictly."})
        if stop_event.is_set():
            await publish({"type": "conversation_end", "id": convo_id, "end_reason": "stopped", "persona_tokens": persona_tokens, "companion_tokens": companion_tokens})
            return
        persona_reply, persona_usage = await simulate.call_persona(persona_prompt, persona_history, seed)
        persona_history.append({"role": "assistant", "content": persona_reply["content"]})
        companion_history.append(persona_reply)
        persona_tokens += simulate.total_tokens(persona_usage)
        persona_turns += 1
        await publish({"type": "message", "id": convo_id, "role": "user", "content": persona_reply["content"], "turn": turn})
        if "[END_of_CONVERSATION]" in persona_reply["content"]:
            await publish({"type": "conversation_end", "id": convo_id, "end_reason": "persona_terminated", "persona_tokens": persona_tokens, "companion_tokens": companion_tokens})
            return
        if not persona_reply["content"].strip():
            await publish({"type": "conversation_end", "id": convo_id, "end_reason": "empty_persona_reply", "persona_tokens": persona_tokens, "companion_tokens": companion_tokens})
            return
        await asyncio.sleep(simulate.GLOBAL_CONFIG["conversation_turn_delay"])
    await publish({"type": "conversation_end", "id": convo_id, "end_reason": "max_conversation_rounds", "persona_tokens": persona_tokens, "companion_tokens": companion_tokens})


async def run_live(personas, companions, runs_per_persona, publish: Callable, stop_event: asyncio.Event):
    await publish({"type": "run_start", "total": len(personas) * len(companions) * runs_per_persona, "timestamp": time.time()})
    try:
        async with httpx.AsyncClient(timeout=30) as companion_client:
            semaphore = asyncio.Semaphore(simulate.GLOBAL_CONFIG["max_parallel_simulations"])
            tasks = []
            for companion_model, persona_file in itertools.product(companions, personas):
                for i in range(runs_per_persona):
                    seed = simulate.select_seed(simulate.GLOBAL_CONFIG["seeds"][i % len(simulate.GLOBAL_CONFIG["seeds"])], i)
                    tasks.append(asyncio.create_task(_wrapped_run(semaphore, persona_file, companion_model, seed, publish, companion_client, stop_event)))
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
    except asyncio.CancelledError:
        stop_event.set()
        raise
    finally:
        end_event = "run_stopped" if stop_event.is_set() else "run_complete"
        await publish({"type": end_event, "timestamp": time.time()})


async def _wrapped_run(semaphore, persona_file, companion_model, seed, publish, companion_client, stop_event):
    async with semaphore:
        await run_conversation(persona_file, companion_model, seed, publish, companion_client, stop_event)
