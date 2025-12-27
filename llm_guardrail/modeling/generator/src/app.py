from fastapi import FastAPI
import re
import os

MODEL_PATH = os.environ.get("MODEL_PATH")
if not MODEL_PATH:
    raise ValueError("Biến môi trường MODEL_PATH chưa được thiết lập cho generator.")

# Lazy-load model/tokenizer
model = None
tokenizer = None

def load_model():
    global model, tokenizer
    if model is None or tokenizer is None:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            dtype="auto",
            device_map="auto",
            local_files_only=True
        )
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATH,
            local_files_only=True,
            trust_remote_code=False,
            repo_type="local"
        )

app = FastAPI()

@app.post("/ask_gen")
def ask(prompt: str):
    load_model()  # chỉ load khi endpoint được gọi
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.2,
    )
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
    content = tokenizer.decode(output_ids, skip_special_tokens=True)
    return {"gen_content": content}
