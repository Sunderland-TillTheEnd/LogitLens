# Logit Lens × Qwen2.5-3B 推理可视化

我们都知道大模型有几十层，但实际使用时只会用最后一层的输出——中间层在想什么，就像黑箱。这个项目用 **Logit Lens（对数透镜）** 把每一层的“模型认为最可能的下文”都投影出来：每生成一个token，就把 37 个隐藏状态（嵌入层 + 36 层 Transformer）分别投影回词表，观察模型在每一层"当时最想猜哪个词"，看它在各层候选概率里如何逐步胜出，直到最后一层收敛。左侧注意力热图记录层内 token 的关注关系，右侧条形图展示各层 top-10 候选词。

本仓库是学习大模型原理的练习项目，包含两个脚本：

| 脚本 | 说明 |
| --- | --- |
| `chat.py` | 前序测试脚本：与本地 Qwen2.5-3B-Instruct 的简单命令行聊天 |
| `logit_lens_hf.py` | 主脚本：Logit Lens 逐层可视化，输出总览热图 + 逐 token 动画 |

## 目录结构

```text
to_git/
├── logit_lens_hf.py          # 主脚本：Logit Lens 逐层可视化
├── chat.py                   # 前序测试：命令行聊天
├── requirements.txt          # Python 依赖清单
├── README.md                 
├── LICENSE                   
├── models/                   # 模型目录（自行下载，不随仓库分发）
│   └── Qwen2.5-3B-Instruct/  # Qwen2.5-3B-Instruct 权重（约 6GB）
└── logs/                     # 运行产物（脚本自动创建，不随仓库分发）
    ├── logit_lens_step01.gif 
    ├── ......
    ├── logit_lens_step08.gif # 逐 token 动画 gif
    └── logit_lens.png        # logit_lens.png 总览热图 
```

## 环境要求

- Python 3.14+（作者在 3.14.6 上验证）
- 操作系统：Windows / WSL2 / Linux（不支持 macOS）
- 内存：建议 ≥ 16GB（模型权重约 6GB + 运行时开销）
- 磁盘：模型约 6GB，另需少量空间存放输出图
- 显卡（只影响速度，不是必需）：
  - 推荐：NVIDIA GPU，显存 ≥ 6GB（作者在 RTX 4050 上验证，全程约几十秒）
  - 最低：无 GPU 也能跑（自动回退 CPU，生成 + 渲染动画会慢很多，请耐心等待）

## 安装

```bash
# 1. 创建虚拟环境（可选，用 conda 或 venv 均可）
conda create -n llm_lab python=3.14 -y
conda activate llm_lab

# 2. 安装依赖（CPU 版 torch 已包含在 requirements.txt 中）
pip install -r requirements.txt

# 3.（有 NVIDIA 显卡时）把 torch 换成 CUDA 版，例如：
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

## 创建目录
参照上文给出的文件目录结构，在主文件夹内新建名为models的空文件夹。
（不必创建logs文件夹，脚本会自动创建）

## 下载模型

脚本通过 `transformers` 加载 **Qwen2.5-3B-Instruct**（safetensors 格式，约 6GB）。
模型不随仓库分发，请自行下载后放到 `models/Qwen2.5-3B-Instruct/`：

```bash
# 国内镜像下载（huggingface-cli 已弃用，用 hf 命令）
HF_ENDPOINT=https://hf-mirror.com hf download Qwen/Qwen2.5-3B-Instruct \
  --local-dir models/Qwen2.5-3B-Instruct
```

Windows 下若模型放在其他盘，可用目录联接避免复制（管理员终端）：

```bat
mklink /J models\Qwen2.5-3B-Instruct D:\LLM_models\Qwen2.5-3B-Instruct
```

> 注意：脚本使用 `local_files_only=True`，模型目录不存在时会直接报错，属预期行为。

## 使用

**命令行聊天（前序测试）**：

```bash
python chat.py     # 输入 exit / 退出 结束
```

**Logit Lens 可视化（主脚本）**：

```bash
python logit_lens_hf.py                    # 默认提示词，生成 8 个 token
python logit_lens_hf.py "床前明月光，" 5   # 自定义提示词 + 生成 5 个 token
```

产物输出到 `logs/`：

- `logit_lens.png`：总览热图，行 = 层（0=嵌入层，1..36），列 = 生成步；
  每格 = 该步最终选中词在对应层的概率；右侧为各层 top-1 命中率。
- `logit_lens_step01.gif ...`：每个生成词一个逐层动画，一帧 = 某一层对该词的
  top-10 概率条形图（红色高亮最终选择），并带该层注意力热力图。

## 常见问题

- **支持哪些操作系统？** 本项目在 Windows、WSL2、Linux 上验证可用，
  **暂不支持 macOS**。Windows 原生直接使用系统字体；WSL2/Linux 用户
  的字体与磁盘挂载路径会自动适配，无需手动配置。
- **图里中文变成方框？** 脚本会自动找中文字体：Windows 用系统字体，
  Linux 可安装开源字体：`sudo apt install fonts-noto-cjk`。
- **GIF 渲染很慢（每个要几分钟）？** 在 WSL2 下这通常是因为字体回退到了从
  Windows 字体目录（`/mnt/c/Windows/Fonts`）慢速读取（9P 协议）。脚本会把字体
  先复制到本地 `/tmp` 缓存再注册；早期版本用 `shutil.copy2` 复制时会保留
  Windows 字体的只读属性，导致缓存文件变只读、下次复制失败后静默回退到慢速
  路径。现已修复：复制前后显式 `os.chmod(..., 0o644)` 解锁缓存。若仍异常缓慢，
  可删除 `/tmp/cjk_fonts/` 缓存目录后重试。
- **显存不足？** 脚本用 `device_map="auto"` 把放不下的层拆到 CPU 上，
  6GB 显存即可运行；纯 CPU 也能跑，只是慢。
- **想换模型？** 修改脚本顶部 `MODEL_PATH` 和 `STOP_IDS`
  （Qwen2.5 系结束符为 151643/151645，其他模型需自行确认）。

## License

MIT
