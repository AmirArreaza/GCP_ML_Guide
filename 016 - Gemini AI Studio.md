# Gemini AI Studio
> **GCP Professional ML Engineer Study Guide**

---

## 1. What is Google AI Studio?

Google AI Studio is a **free, browser-based developer platform** for building, testing, and deploying applications with Gemini models. It provides a unified interface to experiment with prompts, tune parameters, work with multimodal inputs, and export production-ready code — all without installing anything.

**URL:** [aistudio.google.com](https://aistudio.google.com)

**Key characteristics:**
- Browser-based — no downloads, no setup, just a Google account
- Access to the **latest Gemini models** — often gets new models before Vertex AI
- **Free tier** with generous limits (rate-limited but no cost)
- "Get Code" button exports Python, Node.js, curl, or REST snippets instantly
- Direct path to production via Vertex AI or Google Cloud Run

> **AI Studio vs Vertex AI Studio:** AI Studio (aistudio.google.com) is the **developer-facing product** using Google AI API keys. Vertex AI Studio (console.cloud.google.com/vertex-ai/studio) is the **enterprise-facing product** requiring a GCP project, offering data residency, VPC, CMEK, and IAM controls. Both interface with Gemini models.

---

## 2. AI Studio Interface — Key Sections

| Section | Purpose |
|---------|---------|
| **Playground** | Unified workspace for text, multimodal, streaming, and live prompts |
| **Build** | AI-assisted app building — generates code and deploys to Cloud Run with one click |
| **Generate Media** | Experiment with Imagen (images) and Veo (video) generation |
| **My Library** | Saved prompts, conversations, and configurations |
| **API Keys** | Create and manage Gemini API keys for programmatic access |
| **Prompt Gallery** | Pre-built prompt templates for common use cases |

---

## 3. Prompt Types

### Chat Prompt (Multi-turn Conversation)

```
System: You are a helpful customer support agent for a tech company.
         You only answer questions about our products and escalate billing issues.

User: How do I reset my password?
Model: To reset your password, go to the login page and click "Forgot Password"...

User: What if I can't receive emails?
Model: In that case, you can verify your identity by...
```

### Single-turn (Free-form) Prompt

```
Prompt: Summarise the following meeting transcript into a bullet list of action items:

[transcript goes here]
```

### Structured Prompt (with Examples / Few-shot)

```
Input: Classify the sentiment of this review as Positive, Negative, or Neutral.
Examples:
  Input: "Great product, works perfectly!" → Output: Positive
  Input: "Completely broken, waste of money." → Output: Negative
  Input: "It arrived on time." → Output: Neutral
Test: "The colour is nice but the battery drains too fast." → Output:
```

---

## 4. System Instructions

System instructions (also called system prompts) tell the model how to behave **before** the user's message. They persist across all turns of a conversation.

```
System instruction:
  You are a senior financial analyst at a global investment bank.
  - Always format numbers with two decimal places and currency symbols.
  - Never provide personalised investment advice — always recommend consulting a licensed advisor.
  - Respond in a formal, professional tone.
  - If data is unavailable, say "Data not available" rather than estimating.
```

**Uses:**
- Set persona, tone, and style
- Define task scope and restrictions
- Enforce output format requirements
- Inject domain-specific rules and constraints

---

## 5. Model Parameters

All parameters are found in the **Run settings** panel in AI Studio.

### Temperature

Controls randomness in token selection.

| Value | Behaviour | Use When |
|-------|-----------|---------|
| `0.0` | Greedy — always picks most probable token | Factual Q&A, classification, data extraction |
| `0.1–0.4` | Low randomness — consistent, focused | Code generation, structured output, summaries |
| `0.5–0.7` | Balanced | General conversation, explanations |
| `0.8–1.0` | High randomness — diverse, creative | Creative writing, brainstorming, ideation |

> **Gemini 3 note:** For Gemini 3 models, Google recommends keeping temperature at **default 1.0**. Setting it below 1.0 can cause looping or degraded performance on reasoning tasks.

### Top-P (Nucleus Sampling)

Selects the smallest set of tokens whose **cumulative probability ≥ P**.

```
Tokens: A(0.4), B(0.3), C(0.2), D(0.1)
Top-P = 0.7: Selects from {A, B} (cumulative 0.7) — excludes C and D
Top-P = 1.0: All tokens are candidates (effectively disabled)
```

| Value | Effect |
|-------|--------|
| `0.1` | Very focused — only highest-prob tokens |
| `0.9–0.95` | Standard balanced setting |
| `1.0` | All tokens considered |

### Top-K

Limits candidate tokens to the **K most probable**.

| Value | Effect |
|-------|--------|
| `1` | Greedy decoding — always picks top-1 |
| `40` (default) | Standard setting |
| Higher | More variety in token selection |

### Max Output Tokens

Sets the **maximum number of tokens** the model can generate in a response. A token ≈ 4 characters ≈ 0.75 words.

```
100 tokens  ≈ 60–80 words
1,000 tokens ≈ 600–800 words
```

### Stop Sequences

Define strings that **stop generation** when encountered. Useful for controlling structured outputs.

```python
stop_sequences=["### END", "---", "\n\n"]
```

### Seed

Fix a seed value for **reproducible outputs** (best effort — not guaranteed with temperature > 0).

### Parameter Quick Reference

| Parameter | Range | Low = | High = |
|-----------|-------|-------|--------|
| `temperature` | 0.0–2.0 | Deterministic | Creative |
| `top_p` | 0.0–1.0 | Focused | Broad |
| `top_k` | 1–N | Greedy | Diverse |
| `max_output_tokens` | 1–context limit | Short | Long |

---

## 6. Built-in Tools

AI Studio exposes the following tools via the **Run settings** panel:

### Grounding with Google Search

Connects the model to live Google Search results — reduces hallucinations on recent or time-sensitive topics.

```python
from google import genai
from google.genai import types

client = genai.Client(api_key="YOUR_API_KEY")

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="What are the latest quarterly earnings for Alphabet?",
    config=types.GenerateContentConfig(
        tools=[types.Tool(google_search=types.GoogleSearch())],
    ),
)
print(response.text)
# response.candidates[0].grounding_metadata contains cited sources
```

### Grounding with Google Maps

Ground responses in real-world location data. Supported for Gemini 3+ models.

### URL Context

Provide URLs in your prompt; the model fetches and uses their content as context. Supports up to ~20 URLs.

```python
# Pass URL as context
contents = [
    types.Part.from_uri("https://example.com/article.html", mime_type="text/html"),
    "Summarise the key points of this article.",
]
```

### Code Execution

Allow the model to **write and run Python code** in a sandboxed environment — useful for calculations, data analysis, and logic problems.

```python
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Calculate the compound interest on £10,000 at 5% over 10 years.",
    config=types.GenerateContentConfig(
        tools=[types.Tool(code_execution=types.ToolCodeExecution())],
    ),
)
```

### Function Calling

Define custom functions (tools) that the model can invoke to interact with external APIs or services.

```python
# Define a function declaration
def get_weather(location: str, unit: str = "celsius") -> dict:
    """Get the current weather for a location."""
    # call real weather API here
    return {"temperature": 22, "condition": "sunny"}

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="What's the weather in Paris?",
    config=types.GenerateContentConfig(
        tools=[get_weather],    # SDK auto-generates the schema from the function
    ),
)
```

### File Search

Ground the model's responses in your own uploaded files or data store. Launched in public preview (late 2025).

```python
# File Search tool grounds responses in uploaded corpus
config = types.GenerateContentConfig(
    tools=[types.Tool(file_search=types.FileSearch(
        file_uris=["gs://my-bucket/documents/policy.pdf"],
    ))],
)
```

### Structured Output (JSON Mode)

Force the model to respond in a specific JSON schema — essential for downstream parsing.

```python
from pydantic import BaseModel

class MeetingAction(BaseModel):
    owner: str
    task: str
    due_date: str
    priority: str

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Extract action items from: [meeting transcript]",
    config=types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=list[MeetingAction],
    ),
)
import json
actions = json.loads(response.text)
```

---

## 7. Multimodal Inputs in AI Studio

Drag and drop or paste files directly into the AI Studio prompt editor:

| Input Type | What You Can Do |
|-----------|----------------|
| **Image** | Describe, compare, extract data, identify objects |
| **Video** | Summarise, extract timestamps, transcribe audio |
| **Audio** | Transcribe, summarise, detect tone/sentiment |
| **PDF / Document** | Summarise, extract, Q&A, compare versions |
| **Code files** | Review, explain, refactor, debug |

All multimodal inputs are processed within the model's context window.

---

## 8. Context Window & Thinking Mode

### Context Window

Gemini models support very large context windows:

| Model Family | Input Context | Output |
|-------------|--------------|--------|
| Gemini 2.5 Pro | Up to 1M tokens | 64K tokens |
| Gemini 2.5 Flash | Up to 1M tokens | 64K tokens |
| Gemini 3 (Flash/Pro) | 1M tokens | 64K tokens |

**1 million tokens ≈ ~750,000 words ≈ entire books, code repositories, or hours of audio.**

### Thinking Mode (Gemini 2.5+)

Gemini 2.5+ and Gemini 3 models support **adaptive thinking** — the model internally reasons before producing a final answer.

```python
# Control thinking depth with thinking_level
config = types.GenerateContentConfig(
    thinking_config=types.ThinkingConfig(
        thinking_level="HIGH",  # NONE, LOW, MEDIUM, HIGH
    ),
)
```

> **Note:** Gemini 3 uses `thinking_level`; older models used `thinking_budget`. Both are supported for backward compatibility but should not be used together.

---

## 9. Safety Settings

Safety settings allow you to control content filtering thresholds per harm category.

### Harm Categories

| Category | Description |
|----------|-------------|
| `HARM_CATEGORY_HARASSMENT` | Harassment and bullying |
| `HARM_CATEGORY_HATE_SPEECH` | Hate speech and discrimination |
| `HARM_CATEGORY_SEXUALLY_EXPLICIT` | Sexual content |
| `HARM_CATEGORY_DANGEROUS_CONTENT` | Harmful or dangerous instructions |

### Threshold Levels

| Level | Behaviour |
|-------|-----------|
| `BLOCK_LOW_AND_ABOVE` | Blocks the most — strictest |
| `BLOCK_MEDIUM_AND_ABOVE` | Default for most categories |
| `BLOCK_ONLY_HIGH` | Blocks the least — permissive |
| `BLOCK_NONE` | No blocking (requires allowlisting) |

```python
from google.genai import types

safety_settings = [
    types.SafetySetting(
        category="HARM_CATEGORY_DANGEROUS_CONTENT",
        threshold="BLOCK_ONLY_HIGH",
    ),
]

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Explain how fireworks work chemically.",
    config=types.GenerateContentConfig(safety_settings=safety_settings),
)
```

---

## 10. Prompt Engineering Techniques

### Zero-shot

```
Translate the following English text into French:
"The weather is beautiful today."
```

### One-shot / Few-shot

```
Classify the sentiment. Examples:
  "I love this!" → Positive
  "Terrible experience." → Negative
Classify: "It was okay, nothing special." →
```

### Chain-of-Thought (CoT)

```
Solve this step by step:
  A train travels at 60 mph for 2.5 hours, then 80 mph for 1 hour.
  What is the total distance travelled?

Let's think step by step:
```

### Role Prompting

```
System: You are a senior Python engineer with 15 years of experience.
        Review the following code and identify security vulnerabilities.
```

### Prompt Tips

- **Be specific and detailed** — leave minimal room for interpretation
- **Specify output format explicitly** — JSON, Markdown table, bulleted list
- **Put context before instructions** for long documents
- **Put the image first** for single-image prompts
- For Gemini 3: start questions with "Based on the information above..."

---

## 11. Getting Your API Key

```python
# In AI Studio: click "API Keys" → "Create API Key"

from google import genai

client = genai.Client(api_key="YOUR_GEMINI_API_KEY")

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Hello, world!",
)
print(response.text)
```

> **AI Studio API key vs Vertex AI:** AI Studio uses a simple API key with `client = genai.Client(api_key=...)`. Vertex AI uses Application Default Credentials (ADC) with `client = genai.Client(vertexai=True, project=..., location=...)`.

---

## 12. "Get Code" — Export to SDK

After building a prompt in AI Studio, click **"Get Code"** to export it as:
- Python (using `google-genai` SDK)
- JavaScript / Node.js
- REST / curl
- Go

The exported code includes your system instruction, model parameters, and multimodal file references — ready to paste into your application.

---

## 13. Available Models in AI Studio (2025–2026)

| Model | Best For | Context |
|-------|---------|---------|
| `gemini-2.5-pro` | Complex reasoning, code, analysis | 1M tokens |
| `gemini-2.5-flash` | Best value — speed + quality | 1M tokens |
| `gemini-3-flash-preview` | Latest fast model (free tier available) | 1M tokens |
| `gemini-3.1-pro-preview` | Most advanced reasoning (no free tier) | 1M tokens |
| `gemini-3.1-flash-lite-preview` | Cost-efficient workhorse | 1M tokens |
| `imagen-4` | Image generation (GA, 2K resolution) | N/A |
| `veo-3.1` | Video generation | N/A |

> **Note:** AI Studio often gets the newest Gemini models first. Check [ai.google.dev](https://ai.google.dev) for the latest model list as it evolves frequently.

---

## 14. AI Studio vs Vertex AI Studio — When to Use Which

| Feature | AI Studio (Google AI) | Vertex AI Studio |
|---------|----------------------|-----------------|
| Access | Google account only | GCP project + billing |
| Authentication | Simple API key | ADC / Service Account |
| Data residency | No | Yes (regional control) |
| Enterprise security | No VPC / CMEK | VPC, CMEK, IAM |
| Billing | Google AI billing | GCP billing |
| Production SLA | No | Yes |
| Model tuning | Limited | Full support |
| Context caching | Yes | Yes (implicit + explicit) |
| Best for | Prototyping, experiments | Production, enterprise |

---

## 15. One-Click Cloud Run Deployment (2026)

AI Studio now supports deploying applications directly to Cloud Run:

1. Build your prompt / app in **Build mode**
2. Click **"Download App"** — generates Python/JS code
3. Click **"Deploy to Cloud Run"** — handles containerisation, hosting, and scaling
4. A live HTTPS URL is created automatically

---

## 16. Exam-Relevant Tips

- AI Studio uses a **simple API key** — not ADC / service accounts (that's Vertex AI).
- **Temperature 0** = greedy decoding (always most probable token).
- **Gemini 3 models**: keep temperature at **default 1.0** — lowering it can degrade reasoning.
- `top_p` and `top_k` both narrow the candidate token pool — use one, not both.
- **System instructions** persist across all turns of a conversation.
- **Grounding with Google Search** reduces hallucinations — attaches real-time sources.
- **Structured output** uses `response_mime_type="application/json"` + `response_schema` — not just a prompt instruction.
- **Code Execution** runs Python in a sandboxed environment — no network, limited libraries.
- **Function Calling** lets the model trigger your external APIs — you handle the actual execution.
- `thinking_level` (Gemini 3) replaces `thinking_budget` (Gemini 2.5) — do not use both together.
- AI Studio exports **"Get Code"** snippets — always uses the modern `google-genai` SDK.
- `google-generativeai` (old SDK) is **deprecated** — use `google-genai` instead.
- Safety settings: `BLOCK_LOW_AND_ABOVE` = strictest; `BLOCK_ONLY_HIGH` = least restrictive.
- **1M token context window** ≈ entire books or code repositories in a single prompt.

---

## 17. Quick Reference Cheat Sheet

```
ACCESS
  aistudio.google.com          →  browser-based, free, Google account only
  API key                      →  genai.Client(api_key="YOUR_KEY")
  Vertex AI                    →  genai.Client(vertexai=True, project=..., location=...)

MODEL PARAMETERS
  temperature=0                →  deterministic (greedy)
  temperature=1.0              →  Gemini 3 default (recommended)
  top_p=0.95                   →  nucleus sampling (cumulative probability)
  top_k=40                     →  candidate token pool size
  max_output_tokens=           →  cap response length

TOOLS (via GenerateContentConfig)
  google_search                →  ground with live web data + citations
  google_maps                  →  location grounding (Gemini 3+)
  code_execution               →  run Python in sandbox
  function_calling             →  invoke your external APIs
  file_search                  →  ground in uploaded documents
  response_mime_type="application/json"  →  structured output (JSON)

SAFETY
  BLOCK_LOW_AND_ABOVE          →  strictest (most blocked)
  BLOCK_ONLY_HIGH              →  permissive (least blocked)

THINKING MODE
  thinking_level="HIGH"        →  Gemini 3 — depth of reasoning
  thinking_budget=N            →  Gemini 2.5 (backward compatible)

EXPORT
  "Get Code"                   →  Python / JS / curl from AI Studio
  SDK: google-genai            →  current (NOT google-generativeai — deprecated)
```

---

*Study tip: Focus on the three output control categories — parameters (temperature, top-p, top-k), tools (search, code execution, function calling, structured output), and safety settings. Also know the difference between AI Studio (API key, prototyping) and Vertex AI Studio (GCP, enterprise, production). These distinctions are frequently tested.*