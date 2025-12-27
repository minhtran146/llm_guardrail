import pytest
import os
import sys

PROJECT_ROOT = "/home/minh/llm_guardrail"
GUARDRAIL_SRC = f"{PROJECT_ROOT}/llm_guardrail/modeling/guardrail/src"

sys.path.insert(0, GUARDRAIL_SRC)

@pytest.fixture(scope="session", autouse=True)
def set_real_env():
    # Guardrail model path
    os.environ["GUARDRAIL_MODEL_PATH"] = "/home/minh/llm_guardrail/models/guardrail"
    # Generator model path
    os.environ["GENERATOR_MODEL_PATH"] = "/home/minh/llm_guardrail/models/generator"
    os.environ["GENERATOR_URL"] = "http://localhost:8001"
    yield

@pytest.fixture(scope="session")
def guardrail_model():
    """Lazy-load Guardrail model/tokenizer một lần cho toàn bộ session"""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path = os.environ.get("GUARDRAIL_MODEL_PATH")
    if not model_path:
        raise ValueError("GUARDRAIL_MODEL_PATH chưa được thiết lập")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, local_files_only=True, trust_remote_code=False, repo_type="local"
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path, device_map="auto", local_files_only=True
    )
    return model, tokenizer
