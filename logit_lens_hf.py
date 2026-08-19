#!/usr/bin/env python
"""logit_lens_hf.py

Logit Lens（"对数透镜"）× 自回归生成：
把"预测下一个词并回填"的步骤循环执行，连续预测若干个 token，
每预测一个 token 就做一次完整的逐层投影，观察模型"逐层决定每个词"的过程。

每步流程（transformers 原生）：
  1) model(ids, output_hidden_states=True) 前向一次；
  2) 对 37 个隐藏状态逐层投影（最终层 RMSNorm + lm_head）得到词表概率；
  3) 取最终层 top-1 作为本轮生成的 token，拼回上下文，进入下一步；
  4) 每步用模型原生 logits 对拍：最后一层投影与 outputs.logits 应逐位一致。

模型：models/Qwen2.5-3B-Instruct（36 层，hidden=2048）。3B bf16 约 6.2GB
略超 6GB 显存，加载用 device_map="auto" 拆分 GPU/CPU；逐层投影固定在 CPU
上算（模型原生 logits 也是在 CPU 上算的，同设备同内核才能逐位一致）。

用法（在 to_git/ 目录下）: conda run -n llm_lab python logit_lens_hf.py [提示词] [生成token数]
默认提示词 "莫听穿林打叶声，"，默认生成 8 个 token。

产物（logs/）：
  1. logit_lens.png  总览热图：行 = 层（0=嵌入层，1..36），
                               列 = 若干个生成步；每格 = 该步最终选中词在
                               对应层的概率；右侧 = 各层 top-1 命中最终词的比例。
  2. logit_lens_step01.gif ... logit_lens_step08.gif
                               每个生成词一个逐层动画：一帧 = 某一层对该词
                               的 top-10 概率条形图（红色高亮最终选择），
                               fps=1 慢速播放，每秒一帧。
"""

import os
import sys
import tempfile

# matplotlib 的字体/绘图缓存默认写在 ~/.config/matplotlib，可能不可写，
# 提前指到 /tmp 可消除警告并加速字体缓存（必须在 import matplotlib 之前设置）。
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib_cache")

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import animation, font_manager

MODEL_PATH = "models/Qwen2.5-3B-Instruct"
PROMPT = sys.argv[1] if len(sys.argv) > 1 else "莫听穿林打叶声，"
N_STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 8
TOP_K = 10

# 所有文字/数字放大到原来的 1.5 倍。
FONT_SCALE = 1.5

# Qwen2.5 的结束符：<|endoftext|> / <|im_end|>。
STOP_IDS = {151643, 151645}

# matplotlib 自带的 DejaVu 字体没有中文字形，图里的汉字会变成方框，
# 所以按优先级尝试注册一份中文字体（找到第一个存在的即用）：
#   1) 脚本同级的 assets/fonts/（想自带字体就放这里，全平台生效）；
#   2) 系统字体目录（system_font_dirs()）：
#        Windows 原生 -> C:\Windows\Fonts
#        WSL2         -> /mnt/c/Windows/Fonts（挂载自 Windows，9P 协议较慢，
#                        找到后会先复制到本地 /tmp 再注册，避免慢速 IO）
#        Linux        -> /usr/share/fonts（一般没有 Windows 字体，此级多为空）
#   3) LINUX_CJK_FONT_PATHS：Linux 常见开源中文字体（Noto/文泉驿/文鼎），
#      作为没有 Windows 字体时的兜底。
# 注：优先级 2、3 都是按平台动态生成的系统绝对路径（字体不在项目目录里）。
CJK_FONT_NAMES = ("simhei.ttf", "msyh.ttc", "msyh.ttf", "simsun.ttc")


def system_font_dirs():
    """返回本机可能存在的系统字体目录列表（按运行平台区分）。"""
    if os.name == "nt":
        windir = os.environ.get("WINDIR", r"C:\Windows")
        return [os.path.join(windir, "Fonts")]
    return ["/mnt/c/Windows/Fonts", "/usr/share/fonts"]


# Linux 没有 Windows 字体，兜底到常见的开源中文字体完整路径
# （Noto Sans CJK / 文泉驿 / 文鼎），系统装了任意一个即可，找到即用。
LINUX_CJK_FONT_PATHS = (
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "/usr/share/fonts/truetype/arphic/uming.ttc",
)


_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# 汇总候选路径：项目自带字体 -> 系统字体目录 -> Linux 开源字体兜底。
CJK_FONT_CANDIDATES = [
    os.path.join(_SCRIPT_DIR, "assets", "fonts", name) for name in CJK_FONT_NAMES
] + [
    os.path.join(d, name) for d in system_font_dirs() for name in CJK_FONT_NAMES
] + (list(LINUX_CJK_FONT_PATHS) if os.name != "nt" else [])

_CJK_FONT_PATH = None
_FT2FONT = None  # FT2Font 实例缓存：整个进程只打开一次字体文件


def _local_font_copy(path):
    """若字体位于 WSL 挂载盘（/mnt/c 等，9P 协议慢速 IO），
    先复制到本地 /tmp，之后所有读取都走本地，避免反复访问 Windows 字体。"""
    if not path.startswith("/mnt/"):
        return path
    cache_dir = os.path.join(tempfile.gettempdir(), "cjk_fonts")
    os.makedirs(cache_dir, exist_ok=True)
    local = os.path.join(cache_dir, os.path.basename(path))
    try:
        import shutil
        if os.path.exists(local):
            # copy2 会保留 Windows 字体的只读属性，导致缓存文件变成只读、
            # 下次复制失败后静默回退到 /mnt 慢速路径，先解锁再覆盖。
            os.chmod(local, 0o644)
        shutil.copy2(path, local)
        os.chmod(local, 0o644)  # 同样解锁刚复制出的文件，保证下次可覆盖
        return local
    except Exception:
        return path


def setup_cjk_font():
    """注册一份可用的中文字体，否则图里的汉字显示为方框。

    按 CJK_FONT_CANDIDATES 顺序找第一个存在的字体；若位于 WSL 挂载盘
    （/mnt/...），先复制到本地 /tmp 缓存再注册，避免 GIF 渲染期间反复
    从 Windows 慢速读取（9P 协议）。注册成功后整个进程复用该字体。
    """
    global _CJK_FONT_PATH
    for path in CJK_FONT_CANDIDATES:
        if os.path.exists(path):
            try:
                local = _local_font_copy(path)
                font_manager.fontManager.addfont(local)
                name = font_manager.FontProperties(fname=local).get_name()
                plt.rcParams["font.family"] = [name, "DejaVu Sans"]
                plt.rcParams["axes.unicode_minus"] = False
                # 关闭字体微调：中文等宽笔画不受影响，但 FreeType 渲染快 ~30%。
                plt.rcParams["text.hinting"] = "none"
                _CJK_FONT_PATH = local
                return name
            except Exception:
                continue
    return None


def _get_ft2font():
    """懒加载并缓存 FT2Font 单例，避免每次字符过滤都重新打开字体文件。"""
    global _FT2FONT
    if _FT2FONT is None and _CJK_FONT_PATH:
        try:
            from matplotlib.ft2font import FT2Font
            _FT2FONT = FT2Font(_CJK_FONT_PATH)
        except Exception:
            _FT2FONT = False  # 加载失败，之后不再重试
    return _FT2FONT or None


def sanitize_label(text, fallback=""):
    """过滤掉 CJK 字体中不存在的字形，避免 missing glyph 警告和豆腐块。

    单 token 解码经常产生字节残留字符（如 U+1408 之类），当前中文字体与
    DejaVu 均无对应字形。按字体字符表过滤后，若剩余为空则回退为 fallback。
    """
    if not text:
        return fallback
    if _CJK_FONT_PATH is None:
        return text
    font = _get_ft2font()
    if font is None:
        return text
    kept = []
    for ch in text:
        try:
            if font.get_char_index(ord(ch)) != 0:
                kept.append(ch)
        except Exception:
            pass
    return "".join(kept) or fallback


def truncate_label(text, max_width=10):
    """按显示宽度截断标签：CJK/全角字符计 2 宽，其余计 1 宽，超宽截断加省略号。

    默认阈值 10 = 五个汉字，避免条形图的长 token 标签向左越界撞到热力图。
    """
    def _w(ch):
        return 2 if ord(ch) > 0x2E80 else 1

    if not text or sum(_w(c) for c in text) <= max_width:
        return text
    budget = max_width - 2  # 预留省略号“…”（按 2 宽计）
    out, used = [], 0
    for ch in text:
        cw = _w(ch)
        if used + cw > budget:
            break
        out.append(ch)
        used += cw
    return "".join(out) + "…"


def rmsnorm(x, w, eps):
    """与 transformers 的 Qwen2RMSNorm.forward 逐行一致（先升 fp32 再归一化）。"""
    input_dtype = x.dtype
    x = x.to(torch.float32)
    variance = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    return w * x.to(input_dtype)


def probe_logit_lens(model, hidden_states, norm_w, lm_w, top_k=TOP_K):
    """对 37 个隐藏状态逐层投影，统计"每个位置若在此层停下会猜什么"。

    transformers 5.x 的 hidden_states 语义（已用逐层 hook 实测核对）：
      hidden_states[0]      = 嵌入层输出
      hidden_states[1..N-2] = 第 1..N-2 层输出（最终 norm 之前）
      hidden_states[N-1]    = 最终 RMSNorm 之后的状态（== last_hidden_state）
    因此最后一层投影必须跳过 RMSNorm，否则双重归一化会造成 ~12 的 logits 偏差。

    只对最后一个位置做投影（生成只需"下一个词"），比全序列投影快一个量级。

    返回：
      topk_ids    (L+1, K)   每层对下一个词预测的 top-k id
      topk_probs  (L+1, K)   对应概率
      winner      int        最终层 top-1（本轮将生成的词）
      win_probs   (L+1,)     最终选中词在每层的概率
      top1        (L+1,)     每层 top-1 的 token id
      logits_L    (S, V)     最后一层完整投影 logits（用于与模型原生 logits 对拍）
    """
    eps = model.config.rms_norm_eps
    n_layers = len(hidden_states)
    vocab = lm_w.shape[0]
    topk_ids = np.zeros((n_layers, top_k), dtype=np.int64)
    topk_probs = np.zeros((n_layers, top_k), dtype=np.float32)
    probs_last = np.zeros((n_layers, vocab), dtype=np.float32)  # 每层最后位置的完整概率
    logits_L = None

    for l in range(n_layers):
        h = hidden_states[l].squeeze(0).to("cpu").to(lm_w.dtype)  # (S, D)
        if l < n_layers - 1:
            h = rmsnorm(h, norm_w, eps)
        p = torch.softmax((h[-1:] @ lm_w.t()).float(), dim=-1)[0]  # (V,)
        probs_last[l] = p.numpy()
        pk, pi = torch.topk(p, top_k)
        topk_ids[l] = pi.numpy()
        topk_probs[l] = pk.numpy()
        if l == n_layers - 1:
            logits_L = (h @ lm_w.t()).float().numpy()  # (S, V)

    winner = int(probs_last[-1].argmax())
    win_probs = probs_last[:, winner]
    top1 = probs_last.argmax(axis=1)
    return topk_ids, topk_probs, winner, win_probs, top1, logits_L


def tok_label(tokenizer, tid):
    """把 token id 解码成可读短标签，空串/特殊字符回退为 <id>。"""
    s = tokenizer.decode([int(tid)], skip_special_tokens=False).strip()
    return s if s else f"<{tid}>"


def save_steps_heatmap(tokenizer, step_texts, win_probs, hit_rate, path):
    """总览图：左 = 层 x 生成步 的热图（每格 = 该步选中词在对应层的概率）；
    右 = 各层 top-1 命中最终选中词的比例。"""
    n_steps, n_layers = win_probs.shape

    fig, (ax_hm, ax_acc) = plt.subplots(
        1, 2, figsize=(n_steps * 1.6 + 5.5, n_layers * 0.5 + 2.2),
        gridspec_kw={"width_ratios": [3.2, 1]},
    )

    im = ax_hm.imshow(win_probs.T, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    ax_hm.set_xticks(range(n_steps))
    ax_hm.set_xticklabels(
        [f"#{i + 1}\n{t}" for i, t in enumerate(step_texts)], fontsize=8
    )
    ax_hm.set_yticks(range(n_layers))
    ax_hm.set_yticklabels(
        ["嵌入" if i == 0 else str(i) for i in range(n_layers)], fontsize=6
    )
    ax_hm.set_xlabel("生成步（顶部标注 = 该步最终选中的词）", fontsize=9)
    ax_hm.set_ylabel("层（越往下越接近输出）", fontsize=9)
    ax_hm.set_title(
        f"Logit Lens × 生成 {n_steps} 个 token：模型在第几层\"决定\"了每个词？",
        fontsize=10,
    )
    fig.colorbar(im, ax=ax_hm, label="概率")

    xs = np.arange(n_layers)
    ax_acc.bar(xs, hit_rate * 100, color="#4c72b0")
    ax_acc.axhline(hit_rate[-1] * 100, color="#c0392b", ls="--", lw=1)
    ax_acc.set_xticks(xs)
    ax_acc.set_xticklabels(
        ["嵌入" if i == 0 else str(i) for i in range(n_layers)],
        fontsize=6, rotation=90,
    )
    ax_acc.set_ylim(0, 105)
    ax_acc.set_ylabel("各层 top-1 命中最终词的比例 (%)", fontsize=9)
    ax_acc.set_xlabel("层", fontsize=9)
    ax_acc.set_title(f"{n_steps} 步平均命中率\n（最终层 = {hit_rate[-1] * 100:.0f}%）", fontsize=9)

    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"已保存: {path}")


def save_logit_lens_gif(tokenizer, topk_ids, topk_probs, context_text,
                         attn_mean, seq_len, max_seq, token_labels, path):
    """逐层动画（新布局），fps=1。

    版式（三块，GIF 之间逐像素对齐，便于拼接成视频）：
      上   = 已输入上下文（整行，不会再与右侧图重叠）；
      下左 = 该层所有头平均的 token 间注意力热力图——所有 GIF 固定坐标范围
             [0, max_seq]、origin="lower"（左下角对齐），随生成步向右上扩张；
      下右 = 该层对下一个词的 top-10 概率条形图（红色高亮最终选择）。

    文字统一放大 FONT_SCALE 倍（红色"最终选择"行保持原字号）。
    """
    n_layers, k = topk_ids.shape
    winner_id = int(topk_ids[-1, 0])
    winner_lab = sanitize_label(tok_label(tokenizer, winner_id), f"<{winner_id}>")

    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 4.0],
                          width_ratios=[1, 1.0], hspace=0.22, wspace=0.55)
    ax_text = fig.add_subplot(gs[0, :])
    ax_hm = fig.add_subplot(gs[1, 0])
    ax_bar = fig.add_subplot(gs[1, 1])
    ax_text.axis("off")

    # 上下文折行：按顶部文本面板实际像素宽度（整行宽度，足够宽松）。
    renderer = fig.canvas.get_renderer()
    max_text_px = ax_text.get_window_extent(renderer).width * 0.98
    wrapped_ctx = ""
    for para in context_text.split("\n"):
        if not para:
            wrapped_ctx += "\n"
            continue
        cur = ""
        for word in para.split(" "):
            trial = f"{cur} {word}".strip()
            tmp = ax_text.text(0, 0, trial, fontsize=12 * FONT_SCALE,
                               ha="left", va="top")
            wpx = tmp.get_window_extent(renderer).width
            tmp.remove()
            if wpx <= max_text_px or not cur:
                cur = trial
            else:
                wrapped_ctx += cur + "\n"
                cur = word
        wrapped_ctx += cur + "\n"
    wrapped_ctx = wrapped_ctx.rstrip()

    # ---- 热力图静态部分：同一 GIF 内 seq_len 不变，坐标范围固定到 max_seq ----
    im = ax_hm.imshow(
        np.zeros((seq_len, seq_len)),
        cmap="viridis", origin="lower", aspect="equal",
        extent=[0, seq_len, 0, seq_len], vmin=0, vmax=1,
    )
    ax_hm.set_xlim(0, max_seq)
    ax_hm.set_ylim(0, max_seq)
    ax_hm.set_facecolor("#f0f0f0")  # 尚未生成到的区域留浅灰
    ax_hm.add_patch(plt.Rectangle((0, 0), seq_len, seq_len,
                                  fill=False, edgecolor="#333", lw=1.0))
    cb = fig.colorbar(im, ax=ax_hm, fraction=0.046, pad=0.02)
    cb.ax.tick_params(labelsize=10 * FONT_SCALE)
    tick_step = max(1, seq_len // 14)
    tick_idx = list(range(0, seq_len, tick_step))
    tick_pos = [i + 0.5 for i in tick_idx]
    tick_lab = [sanitize_label(token_labels[i]) for i in tick_idx]
    ax_hm.set_xticks(tick_pos)
    ax_hm.set_xticklabels(tick_lab, rotation=90, fontsize=6.5 * FONT_SCALE)
    ax_hm.set_yticks(tick_pos)
    ax_hm.set_yticklabels(tick_lab, fontsize=6.5 * FONT_SCALE)
    ax_hm.set_xlabel("key token（被关注的词）", fontsize=11 * FONT_SCALE)
    ax_hm.set_ylabel("query token（发出注意力）", fontsize=11 * FONT_SCALE)
    ax_hm.set_title("层 0（嵌入层）：无注意力热图", fontsize=11 * FONT_SCALE)

    # 顶部文本面板：静态内容只创建一次，动画中仅更新变化的文字，
    # 避免每帧 clear 后重建整段上下文（GIF 渲染慢的主要来源之一）。
    ax_text.axis("off")
    ax_text.set_title("基于Logit Lens的Qwen3B大模型推理可视化", fontsize=15 * FONT_SCALE)
    layer_text = ax_text.text(0.01, 0.90, f"Layer 0 / {n_layers - 1}",
                              va="top", ha="left", fontsize=12 * FONT_SCALE)
    ctx_text = ax_text.text(0.01, 0.62, wrapped_ctx,
                            va="top", ha="left", fontsize=12 * FONT_SCALE,
                            linespacing=1.5)
    win_text = ax_text.text(0.01, 0.02, f"最终选择：{winner_lab}",
                            va="bottom", ha="left", fontsize=20, color="#c0392b")

    def draw(layer):
        layer_text.set_text(f"Layer {layer} / {n_layers - 1}")

        if layer == 0:
            im.set_data(np.zeros_like(attn_mean[0]))  # 全 0 = viridis 0.0 紫色
            ax_hm.set_title("层 0（嵌入层）：无注意力热图", fontsize=11 * FONT_SCALE)
        else:
            im.set_data(attn_mean[layer - 1])
            ax_hm.set_title(f"层 {layer} 注意力（所有头平均）", fontsize=11 * FONT_SCALE)

        ids = topk_ids[layer]
        probs = topk_probs[layer]
        ys = np.arange(k)
        labels = [truncate_label(sanitize_label(tok_label(tokenizer, t), f"<{t}>"))
                  for t in ids]
        colors = [plt.cm.viridis(p) for p in probs]
        for j, tid in enumerate(ids):
            if tid == winner_id:
                colors[j] = "#c0392b"  # 红色高亮最终答案
        ax_bar.clear()  # 条形图每帧数据不同，只重建本面板，不碰其他 axes
        ax_bar.barh(ys, probs, color=colors, edgecolor="none")
        ax_bar.set_yticks(ys)
        ax_bar.set_yticklabels(labels, fontsize=9 * FONT_SCALE)
        ax_bar.invert_yaxis()
        ax_bar.set_xlim(0, 1.0)
        for y, p in zip(ys, probs):
            ax_bar.text(p + 0.015, y, f"{p:.3f}", va="center", fontsize=8 * FONT_SCALE)
        ax_bar.set_xlabel("概率", fontsize=10 * FONT_SCALE)
        ax_bar.set_title(f"层 {layer} 对下一个词的猜测（top-{k}）",
                         fontsize=11 * FONT_SCALE)
        return ax_text, ax_hm, ax_bar

    anim = animation.FuncAnimation(fig, draw, frames=n_layers,
                                   interval=1000, blit=False)
    anim.save(path, writer="pillow", fps=1)
    plt.close(fig)
    print(f"已保存: {path}")


def main():
    setup_cjk_font()
    torch.set_num_threads(8)
    torch.set_grad_enabled(False)

    import json

    from safetensors import safe_open
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    print("cuda available:", torch.cuda.is_available())
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
        torch_dtype="auto",
        low_cpu_mem_usage=True,
        device_map="auto" if torch.cuda.is_available() else None,
        # eager 才会真正计算并返回注意力权重（SDPA 会丢弃），
        # 供 GIF 里每帧的"该层 token 间注意力热力图"使用。
        attn_implementation="eager",
    )
    model.eval()
    print(f"模型: {type(model).__name__} | 层数: {model.config.num_hidden_layers} "
          f"| hidden: {model.config.hidden_size} | vocab: {model.config.vocab_size}")

    # 从 safetensors 分片直接加载最终层 RMSNorm 和 lm_head（= embed_tokens，二者绑定）
    # 的权重，绕开 device_map 卸载导致的 meta 占位张量。
    with open(os.path.join(MODEL_PATH, "model.safetensors.index.json")) as f:
        weight_map = json.load(f)["weight_map"]

    def load_tensor(name):
        shard = weight_map[name]
        with safe_open(os.path.join(MODEL_PATH, shard), framework="pt", device="cpu") as sf:
            return sf.get_tensor(name)

    proj_dtype = model.dtype  # bf16，与模型原生 logits 同设备同精度
    norm_w = load_tensor("model.norm.weight").to("cpu").to(proj_dtype)
    lm_w = load_tensor("model.embed_tokens.weight").to("cpu").to(proj_dtype)
    print(f"投影权重: norm {tuple(norm_w.shape)} / lm_head(绑定) {tuple(lm_w.shape)} -> cpu {proj_dtype}")

    stop_ids = STOP_IDS | {tokenizer.eos_token_id}
    ids = tokenizer(PROMPT, return_tensors="pt").input_ids
    print(f"prompt: {PROMPT!r} | 初始 token 数: {ids.shape[1]} | 计划生成: {N_STEPS} 个 token")

    step_texts = []
    step_win_probs = []
    step_hit = []
    step_topk_ids = []
    step_topk_probs = []
    step_winner_ids = []
    step_attns = []   # 每步 (n_layers, seq, seq) 所有头平均的注意力
    step_seqs = []    # 每步的序列长度
    step_token_labels = []  # 每步的 token 标签（用于热力图坐标轴）
    all_diffs = []
    full_ids = ids

    for s in range(N_STEPS):
        with torch.no_grad():
            outputs = model(full_ids, output_hidden_states=True, output_attentions=True)
        hidden = outputs.hidden_states
        topk_ids, topk_probs, winner, win_probs, top1, logits_L = probe_logit_lens(
            model, hidden, norm_w, lm_w
        )

        # 收集该步注意力：每层 (1, heads, seq, seq) -> 头平均 -> (seq, seq)。
        if outputs.attentions is None:
            raise RuntimeError("output_attentions=True 未返回注意力，"
                               "请确认模型以 attn_implementation='eager' 加载。")
        attn_mean = np.stack(
            [a[0].float().mean(dim=0).cpu().numpy() for a in outputs.attentions]
        )  # (n_layers, seq, seq)
        step_attns.append(attn_mean)
        step_seqs.append(int(full_ids.shape[1]))
        step_token_labels.append(
            [tokenizer.decode([int(t)], skip_special_tokens=False)
             for t in full_ids[0]]
        )

        # 一致性对拍：最后一层投影 vs 模型原生 logits（每步都校验）。
        ref = outputs.logits[0].float().numpy()
        diff = float(np.abs(logits_L - ref).max())
        all_diffs.append(diff)

        text = tok_label(tokenizer, winner)
        step_texts.append(text)
        step_win_probs.append(win_probs)
        step_hit.append(top1 == winner)
        step_topk_ids.append(topk_ids)
        step_topk_probs.append(topk_probs)
        step_winner_ids.append(winner)

        final_top = "  ".join(
            f"{tok_label(tokenizer, t)}({p:.2f})"
            for t, p in zip(topk_ids[-1, :5], topk_probs[-1, :5])
        )
        print(f"step {s + 1:>2}: 生成 {text!r} (id={winner}) | 一致性max差={diff:.2e} "
              f"| 最终层 top5: {final_top}")

        if winner in stop_ids:
            print(f"  输出结束符，停止生成。")
            break
        full_ids = torch.cat([full_ids, torch.tensor([[winner]])], dim=1)

    hit_rate = np.mean(np.stack(step_hit), axis=0)
    win_probs_mat = np.stack(step_win_probs)
    print(f"\n生成完毕：{len(step_texts)} 个 token，"
          f"每步一致性max差最大为 {max(all_diffs):.2e} "
          f"({'全部一致' if max(all_diffs) < 1e-1 else '存在不一致!'})")
    print(f"完整输出:\n{PROMPT} {' '.join(step_texts)}")

    out_dir = os.path.join("logs")
    os.makedirs(out_dir, exist_ok=True)
    save_steps_heatmap(
        tokenizer, step_texts, win_probs_mat, hit_rate,
        os.path.join(out_dir, "logit_lens.png"),
    )
    # 每个生成词一个逐层动画 GIF：一帧 = 某一层对该词的 top-10 概率条形图。
    # 上下文 = 用 tokenizer 解码完整 token 序列（token 自带前导空格，如
    # ' David'，直接解码即可得到自然的单词间距，无需手动拼接空格）。
    # 串行渲染：字体本地化后每个 GIF 约 10~20 秒，无需多进程；
    # 多进程在已加载大模型的进程上 fork 反而会拖慢（页缓存压力 + 字体 IO）。
    context_ids = ids
    max_seq = max(step_seqs)
    print(f"渲染 {len(step_topk_ids)} 个 GIF（串行，每个约 10~20 秒）...")
    for s, (tk_ids, tk_probs) in enumerate(zip(step_topk_ids, step_topk_probs)):
        save_logit_lens_gif(
            tokenizer, tk_ids, tk_probs,
            tokenizer.decode(context_ids[0], skip_special_tokens=False).strip(),
            step_attns[s], step_seqs[s], max_seq, step_token_labels[s],
            os.path.join(out_dir, f"logit_lens_step{s + 1:02d}.gif"),
        )
        context_ids = torch.cat(
            [context_ids, torch.tensor([[step_winner_ids[s]]])], dim=1
        )
    print("GIF 渲染完成。")


if __name__ == "__main__":
    main()
