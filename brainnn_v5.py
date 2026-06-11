#!/root/miniconda3/bin/python3
"""BrainNN v5 — 14B Zoe LoRA 服务"""
import sys as _sys
_CONDA = "/root/miniconda3/lib/python3.12/site-packages"
if _CONDA not in _sys.path: _sys.path.insert(0, _CONDA)

import os, re
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import torch, time, json
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from flask import Flask, request, jsonify

app = Flask(__name__)

BASE = "/root/autodl-tmp/models/Qwen/Qwen2.5-14B-Instruct"
LORA = "/root/autodl-tmp/archive/zoe_14b_lora_v6"

print("加载模型...", flush=True)
tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
tok.pad_token = tok.eos_token
base = AutoModelForCausalLM.from_pretrained(BASE, device_map="cuda:0", torch_dtype=torch.float16, trust_remote_code=True)
model = PeftModel.from_pretrained(base, LORA, device_map="cuda:0")
model.eval()
print(f"✅ 14B加载完成, VRAM: {torch.cuda.memory_allocated()/1e9:.1f}GB", flush=True)

SYS = {
    "逻辑": "你是一个逻辑推理专家。数列、条件推理、证明。步骤清晰，推理严密。",
    "类比": "你是一个类比压缩专家。概念压缩、比喻、隐喻。深刻独到，不超过50字。",
    "知识": "你是一个知识问答专家。事实、概念、百科。准确全面，简洁明确。",
    "代码": "你是一个代码专家。代码生成、架构、算法。精确实用，可运行。",
    "创意": "你是一个创意写作专家。故事、诗歌、剧本。有创意有感染力。",
}

def thalamus(msg):
    s = {"逻辑": 0, "类比": 0, "知识": 0, "代码": 0, "创意": 0}
    t = msg.lower()
    if any(p in msg for p in ["如果", "推理", "先后", "证明"]): s["逻辑"] += 5
    if len([x for x in msg if x.isdigit()]) >= 3: s["逻辑"] += 3
    if any(p in msg for p in ["之于", "::", "比喻"]): s["类比"] += 5
    if any(p in t for p in ["是什么", "定义", "概念", "为什么"]): s["知识"] += 4
    if any(p in msg for p in ["def ", "class ", "```"]): s["代码"] += 6
    if any(p in t for p in ["写", "故事", "诗歌", "小说"]): s["创意"] += 5
    return max(s, key=s.get)

def gen(msg, max_new=400, temp=0.6):
    expert = thalamus(msg)
    p = f"<|im_start|>system\n{SYS[expert]}<|im_end|>\n<|im_start|>user\n{msg}<|im_end|>\n<|im_start|>assistant\n"
    inp = tok(p, return_tensors="pt", truncation=True, max_length=2048).to("cuda")
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=max_new, temperature=temp, top_p=0.9, pad_token_id=tok.eos_token_id)
    dt = time.time() - t0
    return tok.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True).strip(), expert, dt

@app.route('/health')
def health():
    return jsonify({"model": "BrainNN-v5-14B", "status": "ok", "vram_gb": round(torch.cuda.memory_allocated()/1e9, 1)})

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.get_json()
    msg = data.get('message', '')
    if not msg.strip():
        return jsonify({"error": "空消息"}), 400
    try:
        resp, expert, dt = gen(msg, data.get('max_tokens', 400), data.get('temperature', 0.6))
        return jsonify({"answer": resp, "expert": expert, "time_s": round(dt, 1)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/batch', methods=['POST'])
def batch():
    data = request.get_json()
    msgs = data.get('messages', [])
    results = []
    for m in msgs:
        r, e, dt = gen(m, 300)
        results.append({"message": m, "answer": r, "expert": e, "time_s": round(dt, 1)})
    return jsonify(results)

if __name__ == '__main__':
    print("🚀 BrainNN v5 API: http://0.0.0.0:7861")
    app.run(host='0.0.0.0', port=7861, debug=False, threaded=True)
