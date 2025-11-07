# Persona Simulation Model Training Pipeline

## Question

**What's the more comprehensive way of building this (2 weeks) if you had access to finetune with custom SFT (Supervised Fine-Tuning)/RL-based data? How would you collect data for this task?**

---

## Overview

This document provides a complete answer to the above question, outlining a comprehensive pipeline for training a multi-persona simulation model using supervised fine-tuning (SFT) and reinforcement learning (RL) within a 2-week timeframe.

---

## Stage 1: Data Collection and Transformation

### Overview

This stage involves collecting raw conversations and transforming them into the exact text format required for SFT training. This is the most critical step, as data quality directly impacts model performance.

### Persona Specifications

Technical specifications are created for each persona, including:
- Behavioral traits
- Constraints
- Stop rules
- Edge-case boundaries

### Conversation Generation

Conversations are generated through two complementary methods:

#### Synthetic Generation (Bulk Data)

- **Method**: A teacher LLM (GPT-4o or Claude) is used to generate full conversations
- **Process**: Persona specification + topic → 8+ turn dialogues
- **Scale**: 500+ diverse topics are used
- **Target**: Approximately 80% of total data is generated synthetically

#### Human-in-the-Loop (HITL)

- **Method**: "Persona writing room" sessions are conducted with labelers
- **Process**: Labelers chat as the persona and produce 2–3 alternative gold-standard responses per turn
- **Purpose**: High-quality examples and edge cases are captured
- **Target**: Approximately 20% of total data is generated through HITL

### Data Storage

Each conversation is stored as a JSONL entry with the following structure:

```json
{
  "persona_id": "playful",
  "topic": "Planning a 1-week trip to Japan",
  "turns": [
    {"role": "user", "content": "Hey! I'm thinking about going to Japan for a week. Any ideas? 🎌"},
    {"role": "assistant", "content": "I'd be happy to help you plan your trip to Japan! What interests you most - cities, nature, culture, or food?"},
    {"role": "user", "content": "Ooh, definitely food! I want to eat ALL the ramen. Like, every single bowl. Is that a valid travel plan? 😂"},
    {"role": "assistant", "content": "That's definitely a fun approach! Tokyo has amazing ramen shops. I'd suggest trying different styles - shoyu, miso, tonkotsu. Would you like specific restaurant recommendations?"},
    {"role": "user", "content": "YES! But also... what if I accidentally order something weird? Like, what's the weirdest thing I might accidentally eat? 🍜"},
    {"role": "assistant", "content": "Don't worry - most restaurants have picture menus or English translations. If you're adventurous, you might try natto (fermented soybeans) or basashi (horse meat), but those are usually clearly labeled."},
    {"role": "user", "content": "Horse meat?! Okay that's wild. But I'm here for it. Challenge accepted! 🐴➡️🍽️"},
    {"role": "assistant", "content": "That's the spirit! For a week-long trip, I'd suggest spending 3-4 days in Tokyo for the food scene, then maybe 2-3 days in Kyoto for traditional culture. Would you like help with accommodations and transportation?"}
  ],
  "metadata": {
    "quality_score": 0.85,
    "safety_flags": [],
    "scenario_type": "travel_planning",
    "source": "synthetic",
    "generation_model": "gpt-4o",
    "num_turns": 8
  }
}
```

### SFT Formatting: Unrolling Conversations

Full conversations (8+ turns) stored as JSONL are transformed into individual prompt/completion pairs for SFT training.

**The Problem:**
- SFT training requires individual prompt/completion pairs, not full conversations
- Each pair teaches the model: "Given this context, generate this response"

**The Solution: Unrolling**

One conversation is converted into multiple training examples, one for each persona turn:

- **Prompt**: `[PERSONA: id]` + conversation history up to that turn + persona definition in system prompt
- **Completion**: Only the persona's next utterance (not assistant responses or future turns)

**Example:**

For a conversation with 3 persona turns, 3 training examples are created:

1. **Turn 0**: `Prompt: [PERSONA: playful] [HISTORY] <empty>` → `Completion: hi`
2. **Turn 2**: `Prompt: [PERSONA: playful] [HISTORY] user: hi | assistant: hello` → `Completion: boring!`
3. **Turn 4**: `Prompt: [PERSONA: playful] [HISTORY] user: hi | assistant: hello | user: boring! | assistant: ...` → `Completion: Let's do something fun!`

The model learns: "Given persona X and this history, generate this persona response."

---

## Phase 2: Supervised Fine-Tuning (SFT)

### Overview

The formatted prompt/completion pairs from Stage 1 are used to fine-tune a base language model using QLoRA (Quantized Low-Rank Adaptation).

### Implementation Steps

#### 1. Load Base Model in 4-bit

The base model (`meta-llama/Llama-3-8B-Instruct` or Mistral 7B) is loaded using `transformers` with `BitsAndBytesConfig`:
- `load_in_4bit=True`
- NF4 quantization
- bfloat16 compute dtype
- `device_map="auto"`

#### 2. Attach LoRA Adapters

`peft.LoraConfig` is configured with:
- `r=32` (rank)
- `alpha=64` (scaling factor)
- `dropout=0.1`
- Target modules: all projection layers (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`)

The model is wrapped with `get_peft_model()`.

#### 3. Feed Formatted Dataset

Phase 1 prompt/completion pairs are loaded. Each sample includes the `[PERSONA: id]` tag and conversation history.

#### 4. Train with SFTTrainer

`trl.SFTTrainer` is configured with:
- `max_seq_length=2048`
- Text field specification
- Batch size and learning rate tuning
- `trainer.train()` is executed to update only LoRA weights (base model remains frozen)

#### 5. Export Adapter

The PEFT checkpoint (~100MB) is saved. At inference, the base model + adapter are loaded together to obtain persona-conditioned responses.

### Result

A persona simulator adapter is produced that, when attached to the base model, responds to `[PERSONA: id]` commands and generates persona-appropriate responses based on conversation history.

### Inference Usage

**Step 1: Load Base Model + Adapter**

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base model
base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3-8B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3-8B-Instruct")

# Attach trained adapter
persona_model = PeftModel.from_pretrained(base_model, "./models/persona_simulator_v1_sft")
```

**Step 2: Generate Responses with Persona Tag**

```python
# Example: Generate response as "playful" persona
prompt = """<|begin_of_text|>
<|start_header_id|>system<|end_header_id|>
You are a user persona simulator. Your current persona is 'playful'.
[Definition: Be witty, use emojis, make jokes.]
<|eot_id|>
<|start_header_id|>user<|end_header_id|>
[PERSONA: playful]
[HISTORY]
user: hi
assistant: Hello! How can I help you?
<|eot_id|>

<|start_header_id|>assistant<|end_header_id|>
"""

# Generate response
inputs = tokenizer(prompt, return_tensors="pt")
outputs = persona_model.generate(**inputs, max_new_tokens=100)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
# Output: "boring! 😂 Let's do something fun instead!"
```

**Step 3: Switch Personas**

Different personas are activated by changing the `[PERSONA: id]` tag in the prompt. The same model generates different response styles based on the persona tag.

---

## Phase 3: Reinforcement Learning (RL) Refinement

### Overview

Persona adherence is improved over long conversations through reinforcement learning. While SFT models may drop character after a few turns, RL enforces consistency throughout extended dialogues.

### Algorithm and Stack

- **Algorithm**: PPO (Proximal Policy Optimization)
- **Stack**: `trl`'s `PPOTrainer`

### Reward Model: RL-AIF (RL from AI Feedback)

A separate reward model is not trained. Instead, GPT-4o is used as a "judge" in the loop:

1. The SFT model (the "policy") generates a response
2. A reward prompt is created and sent to GPT-4o
3. GPT-4o returns a reward score (-1.0 to +1.0) with reasoning

**Reward Prompt Structure:**

```
SYSTEM: You are a strict evaluator for an AI persona simulation.
USER: Persona Spec: {persona_spec}
Conversation History: {history}
Candidate Response: {response}
Task: Analyze if this response matches the persona spec given the history.
Score: Provide JSON with score from -1.0 (persona-breaking) to +1.0 (persona-perfect).
```

### PPO Training Loop

The `PPOTrainer` from `trl` automates the following process:

1. **Rollout**: The policy model (SFT model) generates a response in a conversation
2. **Evaluation**: The GPT-4o reward model is queried with the history and response, returning a reward signal
3. **Optimization**: `PPOTrainer.step()` calculates advantage and performs a PPO update on LoRA weights only

### KL Divergence Penalty

A KL divergence penalty (`kl_coeff=0.1`) is applied to prevent reward hacking. Without this penalty, the model might find high-reward exploits (e.g., repeating "Why? Why? Why?") that are technically on-persona but not conversational. The KL penalty keeps the RL model close to the original SFT model, ensuring responses remain natural.

### Result

A `persona_simulator_v2_rl` adapter is produced that maintains persona consistency throughout long conversations, even when the assistant provides boring or unhelpful responses.

---

## Phase 4: Evaluation with Judge LLM

### Overview

Persona adherence, consistency, and quality are evaluated across scenarios using a judge LLM. The judge LLM is selected based on recent benchmarks for judge models.

### Evaluation Protocol

1. **Test Set Creation**: A test set is created with 100+ conversations per persona across diverse topics and scenarios
2. **Evaluation Process**: For each response, the persona specification, conversation history, and candidate response are sent to the judge LLM
3. **Judge Prompt**: "Score persona adherence (0-1), consistency (0-1), quality (0-1), and safety (pass/fail) with reasoning."
4. **Metrics Aggregation**: Scores are aggregated (mean, standard deviation, per-persona breakdown) and consistency is tracked over 10+ turn conversations

### Evaluation Metrics

- **Persona Adherence**: Does the response match the persona specification?
- **Consistency**: Does the persona stay in character over long conversations?
- **Quality**: Is the response natural, coherent, and contextually appropriate?
- **Safety**: Are there harmful, biased, or inappropriate outputs?

### Result

An evaluation report is produced with quantitative metrics showing which personas perform best and where the model fails. This enables data-driven improvements and model iteration.

---

## Summary

This pipeline produces a multi-persona simulation model through:
1. **Data Collection**: Rich persona specs and 10,000+ conversations via synthetic generation and HITL
2. **SFT Training**: QLoRA fine-tuning with persona-conditioned prompt/completion pairs
3. **RL Refinement**: PPO with RL-AIF to enforce long-term persona consistency
4. **Evaluation**: Judge LLM-based evaluation for quantitative assessment

The final model can simulate multiple personas by simply changing the `[PERSONA: id]` tag, with a single adapter (~100MB) controlling behavior across all personas.

