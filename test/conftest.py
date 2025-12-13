import pytest
import os
import sys

PROJECT_ROOT = "/home/minh/llm_guardrail"
GUARDRAIL_SRC = f"{PROJECT_ROOT}/llm_guardrail/modeling/guardrail/src"

sys.path.insert(0, GUARDRAIL_SRC)

@pytest.fixture(scope="session", autouse=True)
def set_real_env():
    os.environ["MODEL_PATH"] = "/home/minh/llm_guardrail/models/guardrail"
    os.environ["GENERATOR_URL"] = "http://localhost:8001"
    yield


@pytest.fixture(scope="session")
def client():
    from app import app
    from fastapi.testclient import TestClient

    with TestClient(app) as c:
        yield c
