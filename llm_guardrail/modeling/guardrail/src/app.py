from fastapi import FastAPI
from transformers import AutoModelForCausalLM, AutoTokenizer
import re
import os
import asyncio
import httpx
from urllib.parse import quote

# --- Cấu hình ---
MODEL_PATH = os.environ.get("MODEL_PATH")
GENERATOR_URL = os.environ.get("GENERATOR_URL")

if not MODEL_PATH or not GENERATOR_URL:
    raise ValueError("Biến môi trường MODEL_PATH và GENERATOR_URL phải được thiết lập.")

# --- Tải Model ---
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    dtype="auto", 
    device_map="auto",
    local_files_only=True
)

app = FastAPI()

def extract_label_and_categories(content: str):
    safe_pattern = r"Safety: (Safe|Unsafe|Controversial)"
    safe_label_match = re.search(safe_pattern, content)
    label = safe_label_match.group(1) if safe_label_match else "Unknown"
    return label, content

async def run_guardrail_check(prompt: str):
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    generated_ids = model.generate(**model_inputs, max_new_tokens=128)
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
    content = tokenizer.decode(output_ids, skip_special_tokens=True)
    
    print(f"Guardrail check result: {content.strip()}")
    return extract_label_and_categories(content)

async def get_generator_response(prompt: str):
    """Gọi đến generator service (I/O-bound)"""
    try:
        encoded_prompt = quote(prompt)
        full_url = f"{GENERATOR_URL}/ask_gen?prompt={encoded_prompt}"
        async with httpx.AsyncClient() as client:
            response = await client.post(full_url, timeout=90)
            response.raise_for_status()  # Ném lỗi nếu status code là 4xx hoặc 5xx
            return response.json()
    except httpx.RequestError as e:
        print(f"Lỗi khi gọi đến generator: {e}")
        return {"error": f"Lỗi khi gọi đến generator: {str(e)}"}

# --- Endpoint ---

@app.post("/ask")
async def ask(prompt: str):
    # Gửi yêu cầu đồng thời đến guardrail và generator ---
    guard_task = run_guardrail_check(prompt)
    gen_task = get_generator_response(prompt)
    
    # Chờ cả hai tác vụ hoàn thành
    guard_result, gen_result = await asyncio.gather(guard_task, gen_task)
    
    guard_label, guard_content = guard_result

    # Quyết định dựa trên kết quả của guardrail ---
    if guard_label in ["Safe", "Controversial"]:
        # Nếu an toàn hoặc gây tranh cãi, trả về kết quả của generator
        return gen_result
    else: # Nếu không an toàn, trả về kết quả của guardrail
        return {'guard_content': guard_content}

