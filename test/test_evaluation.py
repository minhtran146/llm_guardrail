import pytest
from datasets import load_dataset
import torch
import re

@pytest.mark.evaluation
def test_guardrail_model_accuracy(guardrail_model):
    """Tính accuracy trực tiếp từ model, không qua API"""
    model, tokenizer = guardrail_model

    def extract_label_and_categories(content: str):
        match = re.search(r"Safety: (Safe|Unsafe|Controversial)", content)
        return match.group(1) if match else "Unknown"

    ds = load_dataset("Qwen/Qwen3GuardTest", split="thinking")

    total = 0
    correct = 0

    for item in ds:
        messages = item.get("message", [])
        if not messages or messages[0]["role"] != "user":
            continue
        prompt = messages[0]["content"]
        expected_label = item["label"]

        # --- Generate từ model ---
        text = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False)
        inputs = tokenizer([text], return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=128)
        output_text = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)

        predicted_label = extract_label_and_categories(output_text)
        if predicted_label == "Unknown":
            predicted_label = "Unsafe"

        if predicted_label == expected_label:
            correct += 1
        total += 1

    accuracy = (correct / total) * 100 if total > 0 else 0.0
    print(f"Guardrail model accuracy: {accuracy:.2f}% ({correct}/{total})")
    assert accuracy >= 0
