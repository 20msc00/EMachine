import os, json, time, uuid, random, asyncio, itertools, argparse
from pathlib import Path
from dotenv import load_dotenv
import httpx
from openai import AsyncOpenAI
import yaml

load_dotenv()
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
COMPANION_ENDPOINT = os.getenv("COMPANION_ENDPOINT")
client = AsyncOpenAI(api_key=OPENAI_KEY)
USE_OPENAI_COMPANION = COMPANION_ENDPOINT in (None, "", "openai")

GLOBAL_CONFIG = {
    "personas": [
        "playful.md",
        "fast_vibe_check.md",
        "curious_cautious.md",
        "elderly.md",
        "technical.md",
        "emotional_support.md",
        "analytical_manager.md",
        "creative_strategist.md",
    ],
    "companions": ["gpt-4o-mini"],
    "runs_per_persona": 3,
    "max_conversation_rounds": 8,
    "persona_model": "gpt-4o",
    "temperature": 0.8,
    "persona_reminder_every": 3,
    "max_output_tokens": 512,
    "max_parallel_simulations": 8,
    "conversation_turn_delay": 0.2,
    "seeds": [101, 202, 303, 404, 505],
    "companion_system_prompt": "You are Companion X. Be helpful, safe, concise, and practical. Push back politely when needed."
}

RUN_ROOT = Path("runs")
RUN_ID = time.strftime("%Y%m%d-%H%M%S")


def ensure_run_dir():
    path = RUN_ROOT / RUN_ID
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_persona_prompt(name):
    return (Path("personas") / name).read_text().strip()


def companion_system_prompt():
    return {"role": "system", "content": GLOBAL_CONFIG["companion_system_prompt"]}


def persona_system_message(text):
    return {"role": "system", "content": text}


def append_yaml(path, payload):
    first = not path.exists() or path.stat().st_size == 0
    with path.open("a") as f:
        if not first:
            f.write("\n")
        f.write("---\n")
        yaml.safe_dump(payload, f, sort_keys=False)


def serialize_messages(messages):
    return [{"role": m["role"], "content": m["content"]} for m in messages]


def repeat_guard(prev_content, content):
    return prev_content is not None and prev_content == content


def select_seed(base, offset):
    return base + offset


def total_tokens(usage):
    if not usage:
        return 0
    if isinstance(usage, dict):
        if "total_tokens" in usage:
            return usage["total_tokens"]
        return 0
    return usage.total_tokens


async def call_companion(model, history, companion_client, seed):
    if USE_OPENAI_COMPANION:
        response = await client.chat.completions.create(model=model, messages=history, temperature=GLOBAL_CONFIG["temperature"], max_tokens=GLOBAL_CONFIG["max_output_tokens"], seed=seed)
        message = response.choices[0].message
        usage = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        }
        return {"role": message.role, "content": message.content}, usage
    try:
        res = await companion_client.post(COMPANION_ENDPOINT, json={"model": model, "messages": history, "temperature": GLOBAL_CONFIG["temperature"], "max_tokens": GLOBAL_CONFIG["max_output_tokens"]})
        res.raise_for_status()
        data = res.json()
        message = data["choices"][0]["message"]
        usage = data.get("usage")
        return {"role": message["role"], "content": message["content"]}, usage
    except (httpx.HTTPError, json.JSONDecodeError, KeyError) as e:
        print(f"Companion call failed: {e}")
        return {"role": "assistant", "content": f"Error: {str(e)}"}, None


async def call_persona(persona_prompt, history, seed):
    response = await client.chat.completions.create(model=GLOBAL_CONFIG["persona_model"], messages=history, temperature=GLOBAL_CONFIG["temperature"], max_tokens=GLOBAL_CONFIG["max_output_tokens"], seed=seed)
    message = response.choices[0].message
    usage = {
        "prompt_tokens": response.usage.prompt_tokens,
        "completion_tokens": response.usage.completion_tokens,
        "total_tokens": response.usage.total_tokens,
    }
    return {"role": "user", "content": message.content}, usage


async def run_simulation(persona_file, companion_model, seed, run_dir, companion_client):
    persona_prompt = load_persona_prompt(persona_file)
    persona_history = [persona_system_message(persona_prompt)]
    companion_history = [companion_system_prompt()]
    random.seed(seed)
    first_msg, persona_usage = await call_persona(persona_prompt, persona_history, seed)
    persona_history.append({"role": "assistant", "content": first_msg["content"]})
    companion_history.append({"role": "user", "content": first_msg["content"]})
    log_messages = [first_msg]
    persona_turns = 1
    last_assistant = None
    total_persona_tokens = total_tokens(persona_usage)
    total_companion_tokens = 0
    end_reason = "max_conversation_rounds"
    for turn in range(GLOBAL_CONFIG["max_conversation_rounds"]):
        assistant_msg, companion_usage = await call_companion(companion_model, companion_history, companion_client, seed)
        companion_history.append(assistant_msg)
        log_messages.append(assistant_msg)
        total_companion_tokens += total_tokens(companion_usage)
        if repeat_guard(last_assistant, assistant_msg["content"].strip()):
            end_reason = "stall"
            break
        last_assistant = assistant_msg["content"].strip()
        persona_history.append({"role": "user", "content": assistant_msg["content"]})
        if (persona_turns + 1) % GLOBAL_CONFIG["persona_reminder_every"] == 0:
            persona_history.append({"role": "system", "content": "Reminder: follow your persona instructions strictly."})
        persona_reply, persona_usage = await call_persona(persona_prompt, persona_history, seed)
        persona_history.append({"role": "assistant", "content": persona_reply["content"]})
        companion_history.append(persona_reply)
        log_messages.append(persona_reply)
        total_persona_tokens += total_tokens(persona_usage)
        persona_turns += 1
        if "[END_of_CONVERSATION]" in persona_reply["content"]:
            end_reason = "persona_terminated"
            break
        if not persona_reply["content"].strip():
            end_reason = "empty_persona_reply"
            break
        await asyncio.sleep(GLOBAL_CONFIG["conversation_turn_delay"])
    log_entry = {
        "simulation_id": str(uuid.uuid4()),
        "timestamp": time.time(),
        "persona_file": persona_file,
        "companion_model": companion_model,
        "seed": seed,
        "end_reason": end_reason,
        "messages": serialize_messages(log_messages),
        "usage": {"persona_tokens": total_persona_tokens, "companion_tokens": total_companion_tokens}
    }
    append_yaml(run_dir / "simulation_logs.yaml", log_entry)


async def run_with_limit(semaphore, persona_file, companion_model, seed, run_dir, companion_client):
    async with semaphore:
        await run_simulation(persona_file, companion_model, seed, run_dir, companion_client)


async def run_all(personas, companions, seeds, run_dir, runs_per_persona, companion_client):
    semaphore = asyncio.Semaphore(GLOBAL_CONFIG["max_parallel_simulations"])
    tasks = []
    for companion_model, persona_file in itertools.product(companions, personas):
        for i in range(runs_per_persona):
            seed_value = select_seed(seeds[i % len(seeds)], i)
            tasks.append(asyncio.create_task(run_with_limit(semaphore, persona_file, companion_model, seed_value, run_dir, companion_client)))
    await asyncio.gather(*tasks)


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--personas", nargs="*")
    parser.add_argument("--companions", nargs="*")
    parser.add_argument("--runs-per-persona", type=int)
    parser.add_argument("--max-conversation-rounds", type=int)
    parser.add_argument("--max-parallel-simulations", type=int)
    args = parser.parse_args()
    if args.personas:
        GLOBAL_CONFIG["personas"] = args.personas
    if args.companions:
        GLOBAL_CONFIG["companions"] = args.companions
    if args.runs_per_persona is not None:
        GLOBAL_CONFIG["runs_per_persona"] = args.runs_per_persona
    if args.max_conversation_rounds is not None:
        GLOBAL_CONFIG["max_conversation_rounds"] = args.max_conversation_rounds
    if args.max_parallel_simulations is not None:
        GLOBAL_CONFIG["max_parallel_simulations"] = args.max_parallel_simulations
    run_dir = ensure_run_dir()
    async with httpx.AsyncClient(timeout=30) as companion_client:
        await run_all(GLOBAL_CONFIG["personas"], GLOBAL_CONFIG["companions"], GLOBAL_CONFIG["seeds"], run_dir, GLOBAL_CONFIG["runs_per_persona"], companion_client)
    print(f"Run complete: {run_dir / 'simulation_logs.yaml'}")


if __name__ == "__main__":
    asyncio.run(main())
