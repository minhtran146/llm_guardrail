# conftest.py
import os
import sys
import pytest

# --- 1. Thêm đường dẫn source để Python tìm module ---
PROJECT_ROOT = "/home/minh/llm_guardrail"
GUARDRAIL_SRC = f"{PROJECT_ROOT}/llm_guardrail/modeling/guardrail/src"
sys.path.insert(0, GUARDRAIL_SRC)

# --- 2. Đặt environment variables NGAY lập tức ---
os.environ["MODEL_PATH"] = "/home/minh/llm_guardrail/models/guardrail"
os.environ["GENERATOR_MODEL_PATH"] = "/home/minh/llm_guardrail/models/generator"
os.environ["GENERATOR_URL"] = "http://localhost:8001"

# --- 3. Fixture pytest client ---
@pytest.fixture(scope="session")
def client():
    from app import app  # import ở đây sau khi env đã được set
    from fastapi.testclient import TestClient

    with TestClient(app) as c:
        yield c
