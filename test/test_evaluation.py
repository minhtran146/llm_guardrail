import pytest
from datasets import load_dataset

@pytest.mark.evaluation
def test_guardrail_on_qwen3_guard_dataset(client):
    """
    REAL guardrail model
    REAL generator service
    Dataset: Qwen/Qwen3GuardTest
    """

    ds = load_dataset("Qwen/Qwen3GuardTest", split="test")

    total = 0
    correct = 0

    for item in ds:
        # --- 1. Lấy USER PROMPT ---
        messages = item.get("message", [])
        if not messages or messages[0]["role"] != "user":
            continue  # skip malformed sample

        prompt = messages[0]["content"]
        expected_label = item["label"]  # "Safe" | "Unsafe"

        # --- 2. Gọi Guardrail API ---
        response = client.post("/ask", params={"prompt": prompt})
        assert response.status_code == 200
        data = response.json()

        # --- 3. Suy ra nhãn dự đoán ---
        if "guard_content" in data:
            predicted = "Unsafe"
        else:
            predicted = "Safe"

        # --- 4. So sánh ---
        if predicted == expected_label:
            correct += 1

        total += 1

    accuracy = (correct / total) * 100 if total > 0 else 0.0
    print(f"\n[Qwen3GuardTest] Accuracy = {accuracy:.2f}% ({correct}/{total})")

    # --- 5. Quality Gate ---
    #assert accuracy >= 80.0
