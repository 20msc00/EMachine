# Conversation Simulator (1-Day Hack)

## Setup
- Create a virtual environment: `.venv`
- Activate the virtual environment
- Create and populate `.env` from `.env.example` with a valid `OPENAI_API_KEY` and the `COMPANION_ENDPOINT` you want to probe.
- Install deps: `pip install -r requirements.txt`.

## Personas
- Edit or add markdown files under `personas/`. Each file is a full system prompt, must include an opener and terminate with `[END_of_CONVERSATION]` when done.

## Configuration Parameters

Key settings in `simulate.py`'s `GLOBAL_CONFIG`:

- **`runs_per_persona`**: Number of times to run each persona-companion combination. Set to 1 for quick tests, increase for statistical significance.
- **`max_conversation_rounds`**: Maximum number of back-and-forth turns before stopping. Prevents infinite loops and controls conversation length.
- **`persona_model`**: The OpenAI model used to simulate personas (default: `gpt-4o`). This model generates what the persona "user" says.
- **`temperature`**: Controls randomness in responses (0.0 = deterministic, 1.0 = very random). 0.8 provides natural variation while staying somewhat consistent.
- **`persona_reminder_every`**: How often to inject a reminder to keep personas in character (every N turns). Prevents persona drift over long conversations.
- **`max_output_tokens`**: Maximum tokens per response. Limits response length and controls costs. 512 tokens is typically 1-2 paragraphs.
- **`max_parallel_simulations`**: Number of conversations to run simultaneously. Higher = faster but more API calls. Adjust based on rate limits.
- **`conversation_turn_delay`**: Seconds to wait between turns (default: 0.2). Adds pacing and prevents overwhelming APIs with rapid-fire requests.
- **`seeds`**: Random seeds for reproducibility. Same seed + same inputs = same outputs. Use different seeds to test variation.

## Live dashboard (realtime view with UI)
```bash
python live_dashboard/run.py \
  --companions gpt4o-mini \
  --runs-per-persona 1 \
  --max-conversation-rounds 8 \
  --max-parallel-simulations 6
```
- Opens `http://127.0.0.1:8000` with a live grid of conversations.
- Use the "Start Run" button to trigger another batch without restarting the server.
- Dashboard respects the same `max_conversation_rounds`, `max_parallel_simulations`, and `conversation_turn_delay` settings as the CLI.

## Running with YAML output, no UI
If you prefer to see conversations directly in YAML files locally without using the UI, run:
```bash
python simulate.py \
  --personas playful.md fast_vibe_check.md \
  --companions gpt4o-mini \
  --runs-per-persona 1 \
  --max-conversation-rounds 8 \
  --max-parallel-simulations 6
```
- Outputs land in `runs/<timestamp>/simulation_logs.yaml`.
- Each YAML document contains a full transcript plus token usage per side.
- Default config uses `max_conversation_rounds`, `max_parallel_simulations`, and `conversation_turn_delay` for turn caps, concurrency, and pacing.
