# BrainNN v9 — 模型驱动的概念学习系统

## 快速启动
```bash
pip install torch transformers peft flask
python brainnn_v9.py
```

## API
- POST /api/learn  — {"text": "概念定义句子"}
- GET  /api/chat   — {"messages": [{"role":"user","content":"..."}]}
- GET  /api/reset  — 清空记忆
- GET  /api/health — 健康检查

## 模型
- Base: Qwen2.5-14B-Instruct
- LoRA: Zoe 14B v6
- 端口: 7862
