#!/usr/bin/env python3
"""BrainNN v9 — 模型驱动的概念学习，简化架构"""
import sys, os, json, re, time, math
import torch
from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ─── 路径配置 ──────────────────────────────────────────────────────────────────
BASE_PATH = "/root/autodl-tmp/models/Qwen/Qwen2.5-14B-Instruct/"
LORA_PATH = "/root/autodl-tmp/archive/zoe_14b_lora_v6/"

# ─── 简单记忆存储 ─────────────────────────────────────────────────────────────
class SimpleMemory:
    def __init__(self, path):
        self.path = path
        self.concepts = {}
        self.load()

    def load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path) as f:
                    data = json.load(f)
                self.concepts = data.get("concepts", {})
            except:
                self.concepts = {}

    def save(self):
        with open(self.path, "w") as f:
            json.dump({"concepts": self.concepts}, f, ensure_ascii=False, indent=2)

    def add_concept(self, name, definition, category="general"):
        if name not in self.concepts:
            self.concepts[name] = {
                "definition": definition[:300],
                "category": category,
                "learned_at": time.strftime("%Y-%m-%d %H:%M")
            }
            self.save()
            return True
        return False

    def get_context(self, max_c=30):
        if not self.concepts:
            return "（目前没有已学习的概念）"
        lines = []
        for i, (name, info) in enumerate(self.concepts.items()):
            if i >= max_c:
                lines.append(f"...（共{len(self.concepts)}个概念）")
                break
            lines.append(f"- **{name}**：{info['definition'][:80]}")
        return "\n".join(lines)

    def has_concept(self, name):
        return name in self.concepts

    def reset(self):
        """清空所有概念，重新开始"""
        self.concepts = {}
        self.save()


# ─── 工具函数 ─────────────────────────────────────────────────────────────────
def fuzzy_match(name, known_list, threshold=2):
    """检查 name 是否与已知列表中某个概念太相似（编辑距离 <= threshold）"""
    for k in known_list:
        if abs(len(name) - len(k)) > 2:
            continue
        # 简单的字符重叠检查
        common = sum(1 for a, b in zip(name, k) if a == b)
        if common >= min(len(name), len(k)) - threshold:
            return k
        # 编辑距离
        if levenshtein(name, k) <= threshold:
            return k
    return None

def levenshtein(a, b):
    """编辑距离"""
    if len(a) < len(b):
        return levenshtein(b, a)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        curr = [i + 1]
        for j, cb in enumerate(b):
            curr.append(min(prev[j + 1] + 1, curr[j] + 1, prev[j] + (ca != cb)))
        prev = curr
    return prev[-1]


# ─── Flask 应用 ────────────────────────────────────────────────────────────────
app = Flask(__name__)
brain_model = None
brain_tokenizer = None
memory = None


def init():
    global brain_model, brain_tokenizer, memory
    print("加载 tokenizer...", flush=True)
    tok = AutoTokenizer.from_pretrained(BASE_PATH, local_files_only=True, trust_remote_code=True)
    tok.pad_token = tok.eos_token
    print("加载 base model...", flush=True)
    m = AutoModelForCausalLM.from_pretrained(
        BASE_PATH, local_files_only=True, trust_remote_code=True,
        device_map="auto", torch_dtype=torch.bfloat16
    )
    print("加载 LoRA...", flush=True)
    m = PeftModel.from_pretrained(m, LORA_PATH)
    m.eval()
    brain_model = m
    brain_tokenizer = tok
    print("BrainNN 就绪 ✓", flush=True)
    memory = SimpleMemory("/root/autodl-tmp/brainnn_v9_mem.json")
    print(f"记忆加载: {len(memory.concepts)} 个概念", flush=True)


def call_model(prompt, max_new=256, temp=0.3):
    """用 chat template 调用模型"""
    messages = [{"role": "user", "content": prompt}]
    inputs = brain_tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True)
    inputs = {k: v.to(brain_model.device) for k, v in inputs.items()}
    with torch.no_grad():
        out = brain_model.generate(**inputs, max_new_tokens=max_new, do_sample=True, temperature=temp, repetition_penalty=1.1)
    return brain_tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model": "Qwen2.5-14B + Zoe LoRA v9",
        "concepts": list(memory.concepts.keys()) if memory else [],
        "count": len(memory.concepts) if memory else 0
    })


@app.route("/api/chat", methods=["POST"])
def chat():
    if not brain_model:
        return jsonify({"error": "model not ready"}), 503
    data = request.json or {}
    messages = data.get("messages", [])
    context = memory.get_context() if memory else ""

    system = {"role": "system", "content": f"你是 BrainNN，智能知识系统。以下是你已学习的概念：\n{context}"}
    full = [system]
    for m in messages[-6:]:
        if m.get("role") in ("user", "assistant"):
            full.append({"role": m["role"], "content": m["content"]})

    try:
        inputs = brain_tokenizer.apply_chat_template(full, return_tensors="pt", add_generation_prompt=True)
        inputs = {k: v.to(brain_model.device) for k, v in inputs.items()}
        with torch.no_grad():
            out = brain_model.generate(**inputs, max_new_tokens=512, do_sample=True, temperature=0.7, top_p=0.9)
        resp = brain_tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return jsonify({"answer": resp.strip()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/learn", methods=["POST"])
def learn():
    if not brain_model:
        return jsonify({"error": "model not ready"}), 503

    text = (request.json or {}).get("text", "").strip()
    if not text:
        return jsonify({"error": "no text"}), 400

    known = list(memory.concepts.keys())

    prompt = f"""从文本中提取概念，输出格式：概念名|一句话定义。只输出一行，不要markdown。

示例：
文本：机器学习是让计算机从数据中自动学习规律的技术。
概念：机器学习|让计算机从数据中自动学习规律的技术

文本：{text}
概念："""
    try:
        raw = call_model(prompt, max_new=128, temp=0.3)

        added = []
        skipped = []
        # 取第一行，跳过 markdown/table/空行
        first = None
        for line in raw.strip().split("\n"):
            line = line.strip()
            if not line or line.startswith("---") or line.startswith("|"):
                continue
            if "|" in line:
                first = line
                break
        if not first:
            return jsonify({"learned": {"new_concepts": [], "skipped": [{"raw": raw[:100]}]}, "total": len(memory.concepts)})

        parts = first.split("|", 1)
        name = parts[0].strip()
        # 去掉 "概念：" 前缀
        if name.startswith("概念：") or name.startswith("概念:"):
            name = name.split("：", 1)[-1].split(":", 1)[-1].strip()
        definition = parts[1].strip() if len(parts) > 1 else text[:200]

        # 过滤
        if len(name) < 2 or len(name) > 10:
            skipped.append({"name": name, "reason": "length"})
        elif any(c in name for c in "（），、；：""''"):
            skipped.append({"name": name, "reason": "punctuation"})
        elif name in ("概念", "文本", "输出", "回答"):
            skipped.append({"name": name, "reason": "garbage"})
        else:
            # 精确去重（主要逻辑）
            if name in known or name in added:
                skipped.append({"name": name, "reason": "duplicate"})
            else:
                if memory.add_concept(name, definition):
                    added.append(name)
                    known.append(name)
                    print(f"  [学习] {name}")

        return jsonify({
            "learned": {"new_concepts": added, "skipped": skipped},
            "raw": raw[:200],
            "total": len(memory.concepts)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/classify", methods=["POST"])
def classify():
    if not brain_model:
        return jsonify({"error": "model not ready"}), 503

    name = (request.json or {}).get("candidate", "").strip()
    if not name:
        return jsonify({"valid": False})

    prompt = f'判断："{name}" 是有效的科学概念名称吗？回答是或否：'
    try:
        resp = call_model(prompt, max_new=8, temp=0)
        answer = resp.strip()
        valid = any(k in answer for k in ("是", "√", "Y", "有", "1", "T"))
        return jsonify({"candidate": name, "valid": valid, "model_answer": answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/reset", methods=["POST"])
def reset():
    """清空记忆，重新开始"""
    if memory:
        memory.reset()
    return jsonify({"status": "reset", "concepts": []})


@app.route("/api/concepts", methods=["GET"])
def list_concepts():
    if not memory:
        return jsonify([])
    return jsonify([{"name": k, **v} for k, v in memory.concepts.items()])


# ─── 启动 ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    init()
    app.run(host="0.0.0.0", port=7862, threaded=True)