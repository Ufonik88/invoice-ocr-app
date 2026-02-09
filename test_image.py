#!/usr/bin/env python3
"""Test LM Studio image processing to diagnose 400 error."""
import sys
sys.path.insert(0, ".")

import requests
import json
import base64
from PIL import Image, ImageDraw
from io import BytesIO

BASE = "http://localhost:1234"

# 1. Check models
print("=== Models ===")
resp = requests.get(f"{BASE}/v1/models", timeout=10)
for m in resp.json().get("data", []):
    print(f"  {m['id']}")

# 2. Create test image with content
print("\n=== Creating test image ===")
img = Image.new("RGB", (400, 200), color="white")
draw = ImageDraw.Draw(img)
draw.text((50, 50), "INVOICE #12345", fill="black")
draw.text((50, 80), "Total: R1,500.00", fill="black")
draw.text((50, 110), "Date: 2026-01-15", fill="black")

# 3. Test using our _prepare_image method
from app.llm.client import LMStudioClient
from app.config import get_config

config = get_config()
client = LMStudioClient(config.lm_studio)

# Test with PIL Image
b64, mime = client._prepare_image(img)
print(f"PIL Image -> {mime}, base64 length: {len(b64)}")

# Test with raw bytes (simulating uploaded file)
buf = BytesIO()
img.save(buf, format="JPEG")
raw_bytes = buf.getvalue()
b64_bytes, mime_bytes = client._prepare_image(raw_bytes)
print(f"Raw bytes -> {mime_bytes}, base64 length: {len(b64_bytes)}")

# 4. Send to LM Studio
print(f"\n=== Sending to LM Studio ({mime}) ===")
try:
    resp3 = requests.post(f"{BASE}/v1/chat/completions",
        json={
            "model": "qwen3-vl-4b-instruct",
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": "What text do you see in this image? Reply in 1 sentence."},
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}
            ]}],
            "temperature": 0.1,
            "max_tokens": 100,
        },
        timeout=120)
    print(f"Status: {resp3.status_code}")
    print(f"Body: {resp3.text[:500]}")
except Exception as e:
    print(f"Error: {e}")

