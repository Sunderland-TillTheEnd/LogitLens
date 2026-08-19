#!/usr/bin/env python
"""chat.py — 本地模型聊天。

用法:
    python chat.py   # 与本地 Qwen2.5-3B-Instruct (HF safetensors) 对话

模型走 transformers（HF 格式），有 GPU 时自动用 GPU，否则回退 CPU。
可用环境变量 CHAT_MAX_TOKENS 限制单次回复长度。
"""

import os
import sys

# ---------------- 模型注册表 ----------------
MODELS = {
    "qwen3b": {
        "short": "Qwen3B",
        "name": "Qwen2.5-3B-Instruct (HF)",
        "path": "models/Qwen2.5-3B-Instruct",
        "backend": "transformers",
    },
}
DEFAULT_MODEL = "qwen3b"

MAX_HISTORY = 20
MAX_TOKENS = int(os.environ.get("CHAT_MAX_TOKENS", "512"))
TEMPERATURE = 0.7


# ---------------- 后端 ----------------
def build_transformers(path):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype="auto")
    if torch.cuda.is_available():
        model = model.to("cuda")
    return tokenizer, model


def reply_transformers(tok_model, messages):
    import torch

    tokenizer, model = tok_model
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    new = out[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new, skip_special_tokens=True).strip()


# ---------------- 主流程 ----------------
def main():
    key =  DEFAULT_MODEL
    if key not in MODELS:
        names = "、".join(MODELS)
        print(f"未知模型: {sys.argv[1]}\n可用模型: {names}")
        sys.exit(1)
    cfg = MODELS[key]

    print(f"正在加载 {cfg['name']} ...", flush=True)
    model = build_transformers(cfg["path"])
    chat = lambda messages: reply_transformers(model, messages)

    print(f"{cfg['name']} 本地聊天已启动，输入 exit / 退出 结束。\n")

    messages = []
    while True:
        try:
            prompt = input("你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break
        if not prompt:
            continue
        if prompt.lower() in {"exit", "quit", "退出"}:
            print("再见！")
            break

        messages.append({"role": "user", "content": prompt})
        try:
            text = chat(messages)
        except Exception as exc:
            print(f"生成失败: {exc}")
            messages.pop()
            continue
        print(f"{cfg['short']}: {text}\n")
        messages.append({"role": "assistant", "content": text})
        if len(messages) > MAX_HISTORY:
            messages = messages[-MAX_HISTORY:]


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"发生错误: {exc}", file=sys.stderr)
        sys.exit(1)
