# Plan: task-016 — Migrate Google embedder to google-genai SDK

## Files to Modify
- `src/embedders/google_text.py` — replace old SDK with new Client-based SDK
- `requirements.txt` — remove google-generativeai and exclusive deps

## Approach
1. Replace `import google.generativeai as genai` with `from google import genai`
2. Add `from google.genai import types` for EmbedContentConfig
3. Replace `genai.configure(api_key=...)` with `self._client = genai.Client(api_key=...)`
4. Replace `genai.embed_content(...)` with `self._client.models.embed_content(...)`
5. Update response access: `result["embedding"]` → `result.embeddings[0].values`
6. Clean up requirements.txt — carefully remove only exclusive deps

## Ambiguities
- None — spec provides exact API migration map.
