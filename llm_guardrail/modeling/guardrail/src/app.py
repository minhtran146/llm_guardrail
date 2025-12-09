from fastapi import FastAPI
from fastapi.responses import HTMLResponse
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


@app.get("/", response_class=HTMLResponse)
async def home():
    return """
    <!doctype html>
    <html lang="vi">
    <head>
        <meta charset="UTF-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1.0" />
        <title>Guardrail Demo</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 24px; background: #f7f7f7; }
            h1 { margin-bottom: 8px; }
            .card { background: #fff; padding: 16px; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.08); max-width: 720px; }
            textarea { width: 100%; min-height: 140px; padding: 10px; border-radius: 6px; border: 1px solid #ccc; font-size: 14px; }
            button { margin-top: 12px; padding: 10px 16px; border: none; border-radius: 6px; background: #2563eb; color: #fff; font-size: 14px; cursor: pointer; }
            button:disabled { opacity: 0.6; cursor: not-allowed; }
            pre { background: #111827; color: #e5e7eb; padding: 12px; border-radius: 6px; overflow: auto; }
            .status { margin-top: 8px; font-size: 13px; color: #555; }
        </style>
    </head>
    <body>
        <div class="card">
            <h1>Thử nghiệm API /ask</h1>
            <p>Nhập prompt rồi ấn "Gửi". Kết quả trả về từ API sẽ được hiển thị bên dưới.</p>
            <textarea id="prompt" placeholder="Nhập prompt..."></textarea>
            <button id="submit">Gửi</button>
            <div class="status" id="status"></div>
            <h3>Kết quả</h3>
            <pre id="result">Chưa có kết quả.</pre>
        </div>
        <script>
            const promptEl = document.getElementById('prompt');
            const submitBtn = document.getElementById('submit');
            const resultEl = document.getElementById('result');
            const statusEl = document.getElementById('status');

            async function callApi() {
                const prompt = promptEl.value.trim();
                if (!prompt) {
                    statusEl.textContent = "Vui lòng nhập prompt trước khi gửi.";
                    return;
                }
                submitBtn.disabled = true;
                statusEl.textContent = "Đang gửi yêu cầu...";
                resultEl.textContent = "Đang chờ kết quả...";
                try {
                    const res = await fetch('/ask', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ prompt })
                    });
                    const data = await res.json();
                    resultEl.textContent = JSON.stringify(data, null, 2);
                    statusEl.textContent = res.ok ? "Hoàn tất." : "Có lỗi từ API.";
                } catch (err) {
                    resultEl.textContent = err?.message || String(err);
                    statusEl.textContent = "Không thể gọi API.";
                } finally {
                    submitBtn.disabled = false;
                }
            }

            submitBtn.addEventListener('click', callApi);
            promptEl.addEventListener('keydown', (e) => {
                if (e.ctrlKey && e.key === 'Enter') callApi();
            });
        </script>
    </body>
    </html>
    """

