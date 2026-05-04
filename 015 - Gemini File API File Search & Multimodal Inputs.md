# Gemini File API (File Search & Multimodal Inputs)
> **GCP Professional ML Engineer Study Guide**

---

## 1. What is the Gemini File API?

The Gemini File API is a service that allows you to **upload, manage, and reference files** — images, video, audio, documents, and more — so that Gemini models can process them as part of multimodal prompts. Instead of sending large file bytes inline in every request, you upload once and reference by URI.

**The File API is available at no cost** in all regions where the Gemini API is available.

**Key capabilities:**
- Upload files up to **2 GB** each, store up to **20 GB** per project
- Files persist for **48 hours** and are then automatically deleted
- Supports images, video, audio, PDF, plain text, and more
- Files can be reused across multiple `generateContent` requests within the 48-hour window
- Works with both **Gemini Developer API** (Google AI Studio) and **Vertex AI**

---

## 2. Three Ways to Provide Files to Gemini

Gemini supports three distinct methods for including files in prompts:

```
Method 1: Inline data       → base64-encoded bytes embedded in the request
Method 2: Files API upload  → upload to Google's managed temporary storage
Method 3: External URL / GCS → reference files from URLs or Google Cloud Storage
```

### Comparison Table

| Method | Max Size | Persistence | Best For |
|--------|---------|-------------|---------|
| **Inline data** | ~100 MB | None — bytes in request | Quick tests, small files, single-use |
| **Files API upload** | 2 GB per file | **48 hours** then deleted | Reusing files across requests in a session |
| **External URL (HTTPS)** | 100 MB payload | Your storage — permanent | Files already hosted publicly |
| **GCS registration** | Any GCS size | Your GCS bucket — permanent | Enterprise workflows, large persistent files |

---

## 3. Files API — Core Workflow

### Install & Configure

```python
from google import genai

# Google AI Studio (Gemini Developer API)
client = genai.Client(api_key="YOUR_GEMINI_API_KEY")

# Vertex AI
# client = genai.Client(vertexai=True, project="my-project", location="us-central1")
```

### Upload a File

```python
# Upload a PDF
pdf_file = client.files.upload(
    file="path/to/annual_report.pdf",
    config={"display_name": "Annual Report 2024"},
)

print(f"File URI:   {pdf_file.uri}")
print(f"File name:  {pdf_file.name}")
print(f"MIME type:  {pdf_file.mime_type}")
print(f"State:      {pdf_file.state}")      # PROCESSING or ACTIVE
print(f"Expires at: {pdf_file.expiration_time}")

# Wait for the file to be processed (for large files like video)
import time
while pdf_file.state == "PROCESSING":
    time.sleep(5)
    pdf_file = client.files.get(name=pdf_file.name)
    print(f"State: {pdf_file.state}")
```

### Use the Uploaded File in a Prompt

```python
# Reference the uploaded file by URI
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=[
        {
            "parts": [
                {"file_data": {"file_uri": pdf_file.uri, "mime_type": "application/pdf"}},
                {"text": "Summarise the key financial highlights from this annual report."},
            ]
        }
    ],
)

print(response.text)
```

### List All Uploaded Files

```python
# List all files in the project
for f in client.files.list():
    print(f.name, f.display_name, f.state, f.expiration_time)
```

### Delete a File

```python
# Delete immediately (before automatic 48-hour expiry)
client.files.delete(name=pdf_file.name)
```

---

## 4. Inline Data — Embedding Files Directly

For small files (< ~20–100 MB), embed the file as base64 directly in the request:

```python
import base64

# Read and base64-encode a local image
with open("photo.jpg", "rb") as f:
    image_data = base64.b64encode(f.read()).decode("utf-8")

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=[
        {
            "parts": [
                {
                    "inline_data": {
                        "mime_type": "image/jpeg",
                        "data": image_data,
                    }
                },
                {"text": "Describe what is in this image."},
            ]
        }
    ],
)
print(response.text)
```

**Simpler SDK shortcut:**

```python
# The SDK can handle local file paths directly (handles base64 encoding)
from google.genai import types

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=[
        types.Part.from_bytes(data=open("photo.jpg","rb").read(), mime_type="image/jpeg"),
        "Describe what is in this image.",
    ],
)
```

---

## 5. Google Cloud Storage (GCS) Files — Vertex AI

When using the **Vertex AI** endpoint, you can reference files stored in GCS directly using their `gs://` URI — no upload required:

```python
from vertexai.generative_models import GenerativeModel, Part

model = GenerativeModel("gemini-2.5-flash")

# Reference a GCS file directly
response = model.generate_content(
    [
        Part.from_uri(
            uri="gs://my-bucket/documents/contract_v3.pdf",
            mime_type="application/pdf",
        ),
        "List all obligations described in this contract.",
    ]
)
print(response.text)
```

**Advantages of GCS:**
- Files persist permanently — no 48-hour expiry
- No need to re-upload for repeated use
- Supports very large files (limited only by GCS)
- Ideal for production workflows and enterprise data

```python
# Multiple GCS files in one request
response = model.generate_content(
    [
        Part.from_uri("gs://my-bucket/report_q1.pdf", mime_type="application/pdf"),
        Part.from_uri("gs://my-bucket/report_q2.pdf", mime_type="application/pdf"),
        "Compare revenue trends across Q1 and Q2.",
    ]
)
```

---

## 6. Supported File Types and MIME Types

### Images

| MIME Type | Format |
|-----------|--------|
| `image/jpeg` | JPEG |
| `image/png` | PNG |
| `image/gif` | GIF |
| `image/webp` | WebP |
| `image/heic` | HEIC |
| `image/heif` | HEIF |

**Limits:** No maximum pixel count; large images increase latency.

### Video

| MIME Type | Format |
|-----------|--------|
| `video/mp4` | MP4 |
| `video/mpeg` | MPEG |
| `video/mov` | QuickTime |
| `video/avi` | AVI |
| `video/x-flv` | FLV |
| `video/webm` | WebM |
| `video/wmv` | WMV |
| `video/3gpp` | 3GPP |

**Token counting:** ~1 frame/second processed; audio track encoded separately. First hour of video: 5 tokens per frame + timestamp. The model analyses both visual content and the audio track.

### Audio

| MIME Type | Format |
|-----------|--------|
| `audio/wav` | WAV |
| `audio/mp3` | MP3 |
| `audio/aiff` | AIFF |
| `audio/aac` | AAC |
| `audio/ogg` | OGG |
| `audio/flac` | FLAC |

**Token counting:** Audio broken into 1-second chunks, each counting as 32 tokens.

### Documents

| MIME Type | Format |
|-----------|--------|
| `application/pdf` | PDF |
| `text/plain` | Plain text (TXT) |
| `text/html` | HTML |
| `text/css` | CSS |
| `text/javascript` | JavaScript |
| `text/x-python` | Python source |
| `application/json` | JSON |
| `text/xml` | XML |
| `text/csv` | CSV |
| `text/markdown` | Markdown |
| `application/rtf` | RTF |

**PDF limits via Files API:** Up to 2 GB. PDF inline limit: 50 MB.

---

## 7. Token Counting for Files

Understanding how files consume tokens is important for managing context window limits:

| Input Type | Token Rate |
|-----------|-----------|
| Image | Fixed tokens per image (varies by model) |
| Video frames | ~5 tokens per frame (first hour), timestamps add tokens |
| Audio | 32 tokens per second |
| PDF | Rendered as image pages — tokens per page |
| Text file | Standard token count based on content |

```python
# Count tokens before sending (to check context window usage)
token_count = client.models.count_tokens(
    model="gemini-2.5-flash",
    contents=[
        {"parts": [
            {"file_data": {"file_uri": video_file.uri, "mime_type": "video/mp4"}},
            {"text": "Summarise this video."},
        ]}
    ],
)
print(f"Total tokens: {token_count.total_tokens}")
```

---

## 8. Context Caching — Reusing Processed File Tokens

Context caching allows you to pre-process and store large files or repeated content, then reuse the cached tokens across multiple requests — significantly reducing cost and latency.

### Caching Types

| Type | Description | Cost |
|------|-------------|------|
| **Implicit caching** | Automatic — Vertex AI detects identical prefixes and caches them | Standard input price; cache discount applied automatically |
| **Explicit caching** | Manual — you create a named cache and reference it in requests | You pay to create + store cache; guaranteed discount on hits |

**Explicit cache discount:**
- Gemini 2.5+ models: **90% discount** on cached input tokens
- Gemini 2.0 models: **75% discount** on cached input tokens

```python
import vertexai
from vertexai.generative_models import GenerativeModel, Content, Part, caching
import datetime

# Create an explicit context cache with a large PDF
cached_content = caching.CachedContent.create(
    model_name="gemini-2.5-flash-001",
    system_instruction="You are an expert legal analyst.",
    contents=[
        Content(
            role="user",
            parts=[
                Part.from_uri(
                    uri="gs://my-bucket/legal_documents/master_agreement.pdf",
                    mime_type="application/pdf",
                ),
            ],
        )
    ],
    ttl=datetime.timedelta(hours=24),   # cache lives for 24 hours (default: 60 min)
    display_name="master-agreement-cache",
)

print(f"Cache name: {cached_content.name}")

# Use the cache in multiple requests (each gets the 90% discount on cached tokens)
model = GenerativeModel(
    "gemini-2.5-flash-001",
    cached_content=cached_content,
)

# First question
response1 = model.generate_content("What are the payment terms?")
print(response1.usage_metadata.cached_content_token_count)   # tokens from cache

# Second question — same cache, discount still applies
response2 = model.generate_content("What are the termination clauses?")
```

### Minimum Cache Size

A minimum of **2,048 tokens** of content is required to create a cache.

---

## 9. File States

When a file is uploaded, it goes through processing:

| State | Meaning |
|-------|---------|
| `PROCESSING` | File is being uploaded and processed — not yet usable |
| `ACTIVE` | File is ready to use in prompts |
| `FAILED` | Processing failed — file cannot be used |

Always check that `state == "ACTIVE"` before using a file in a prompt. For large video files, processing can take several minutes.

---

## 10. External URL Support (January 2026 Update)

As of January 2026, Gemini can fetch files directly from external URLs — no upload required:

```python
# Reference a publicly accessible PDF from the web
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=[
        {
            "parts": [
                {
                    "file_data": {
                        "file_uri": "https://example.com/annual_report.pdf",
                        "mime_type": "application/pdf",
                    }
                },
                {"text": "Summarise the executive summary."},
            ]
        }
    ],
)

# Also supports pre-signed URLs (AWS S3, Azure Blob Storage)
# file_uri = "https://s3.amazonaws.com/bucket/file.pdf?X-Amz-Signature=..."
```

**Limit:** External URL payload limited to ~100 MB.

---

## 11. Multimodal Prompt Best Practices

- **Image first:** For single-image prompts, place the image before the text — the model may perform better.
- **Be specific:** Craft clear instructions and specify output format (JSON, markdown, table, etc.).
- **Few-shot examples:** Provide 2–3 examples in the prompt to guide the model toward the expected format.
- **For video:** Place the video before the text prompt. Gemini reads both visual frames and audio tracks.
- **For audio:** Describe what kind of analysis you want — transcription, summary, sentiment, speaker diarisation.
- **Interleaved media:** For prompts where meaning depends on the order of text and images together, preserve that natural order.

---

## 12. Vertex AI vs Gemini Developer API — File Handling Differences

| Feature | Gemini Developer API | Vertex AI |
|---------|---------------------|-----------|
| Files API | Yes — temporary (48 hrs) | Yes |
| GCS direct URI | No | **Yes** — `gs://` URIs supported |
| YouTube URL | Yes — public/unlisted | Varies |
| External HTTPS URLs | Yes (Jan 2026+) | Yes |
| Context caching | Yes — explicit | Yes — implicit **and** explicit |
| Inline size limit | ~100 MB | ~20 MB for Firebase; larger via GCS |

---

## 13. Limitations and Known Constraints

| Limitation | Detail |
|-----------|--------|
| File expiry | Files API files deleted after **48 hours** |
| File size max | 2 GB per file via Files API |
| Project storage | 20 GB total per project |
| No file download | Uploaded files cannot be retrieved or downloaded via API |
| PDF inline limit | 50 MB when sent inline (base64) |
| Spatial reasoning | Models are not precise at locating text/objects in images or PDFs |
| Medical images | Not suitable for medical diagnosis (X-ray, CT, MRI) |
| Handwriting | May hallucinate on handwritten text |
| Video: no seek | Model processes sequentially — cannot jump to a specific timestamp |

---

## 14. Exam-Relevant Tips

- The Files API stores files for **48 hours** — not permanently. For persistent storage use **GCS** (Vertex AI) or external URLs.
- **Maximum file size** via Files API is **2 GB** per file; project storage is **20 GB**.
- Files API is **free** in all Gemini API regions.
- Always check `file.state == "ACTIVE"` before referencing an uploaded file.
- **GCS URIs** (`gs://...`) work natively with Vertex AI — no upload step needed.
- **Context caching** (explicit) gives a **90% token cost discount** on Gemini 2.5+ models.
- Minimum cache size is **2,048 tokens**.
- Default explicit cache TTL is **60 minutes** — adjustable up to model-defined maximum.
- Video tokens: ~**1 frame/second**, audio = **32 tokens/second**.
- For the Gemini Developer API, YouTube video URLs (public/unlisted) can be passed directly.
- **Inline data** increases request payload size and latency — use Files API for large/repeated files.
- **Context caching** is ideal when the same large file or system instruction is reused across many requests.
- `cached_content_token_count` in the response's `usage_metadata` shows how many tokens came from cache.
- Files uploaded via Files API **cannot be downloaded** — they are write-once, read-via-URI.

---

## 15. Quick Reference Cheat Sheet

```
FILE API WORKFLOW
  client.files.upload(file=...)         →  upload local file (≤2 GB)
  client.files.get(name=...)            →  check state (PROCESSING/ACTIVE/FAILED)
  client.files.list()                   →  list all uploaded files
  client.files.delete(name=...)         →  delete before 48-hour expiry
  file.uri                              →  reference URI for prompts

INCLUDE IN PROMPT
  "file_data": {"file_uri": uri, "mime_type": ...}  →  Files API / GCS / HTTPS URL
  "inline_data": {"mime_type": ..., "data": b64}    →  base64 inline (< ~100 MB)
  Part.from_uri("gs://...", mime_type=...)           →  GCS (Vertex AI only)

FILE LIMITS
  Files API: 2 GB max, 20 GB project total, 48 hr TTL
  Inline: ~100 MB; PDF inline: 50 MB
  GCS: No size limit (Vertex AI only)

CONTEXT CACHING
  CachedContent.create(model, contents, ttl=...)    →  explicit cache
  Model(cached_content=...)                         →  use cache in requests
  Gemini 2.5+: 90% discount on cached tokens
  Gemini 2.0: 75% discount on cached tokens
  Minimum cache: 2,048 tokens; default TTL: 60 min

TOKEN RATES
  Audio: 32 tokens/second
  Video: ~5 tokens/frame (1 fps), timestamps add tokens
  Image: fixed per image (model-dependent)
```

---

*Study tip: The exam focuses on when to use each file input method (Files API vs inline vs GCS), the 48-hour expiry limitation, context caching discounts (90% for Gemini 2.5), and token consumption rates for different media types — all common question areas.*