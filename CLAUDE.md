# Utter

本地优先的双模语音平台：**听**（英语讲座/会议实时转录 + 译中）与 **说**（听写，转成文字注入光标）。macOS 为主，需可分发到 Windows / Intel Mac。

> 项目原名 **LiveScribe**，2026-08-09 更名 Utter。v1/v2 文档里的 "LiveScribe" 指本项目旧名。

## 任务性质

这是**工具开发项目，不是学术写作项目**。全局 CLAUDE.md 的 LTL 心智在此降级——以任务正确性与清晰度为主。但作者是翻译研究学者、非软件工程背景：**解释技术决策时要落到具体，不要堆缩写**。术语参见 [docs/概念说明.md](docs/概念说明.md)。

## 权威文档

| 文档 | 地位 |
|---|---|
| [docs/plans/2026-08-09-v3-design.md](docs/plans/2026-08-09-v3-design.md) | **唯一现行设计权威源** |
| [docs/教训与经验.md](docs/教训与经验.md) | **踩过的坑与已知不足**，动手前先看 |
| `docs/plans/2026-04-14-livescribe-v2-*.md` | 管线部分已作废；**会话模型与 UI 组件那六成仍有效**，P3 照它写。存废清单见 v3 设计 §7 |
| [docs/benchmarks/](docs/benchmarks/) | 实测数据，架构结论的依据 |
| [docs/plans/2026-08-10-p1-build-log.md](docs/plans/2026-08-10-p1-build-log.md) | P1 施工日志：自主决定、被推翻的旧说法、待办 |

**不要再提议"执行现成的 v2 计划"。** v2 的 partial 管线已被实测证伪。

v1 的计划文档与后端服务器已于 2026-08-10 删除（git 历史里还在）。

## 铁律

1. **实时管线的 temperature 阶梯必须有界，且第一级为 0.0。** 现值 `(0.0, 0.2)`，
   并关掉 `condition_on_previous_text`。
   **2026-08-10 修正**：原文是"必须 `temperature=0.0`"（标量）。那样确实把最坏情况从
   8–11 秒压到 1 秒，但漏了一件事——**那条阶梯的用途正是逃出复读循环**：Whisper 每轮解码后
   查 `compression_ratio`，输出退化就升温重试；标量意味着无处可退，模型一旦开始复读就一路
   复读到窗口结束。作者实测收到过同一短语重复五十遍。两级封顶把最坏情况定在约 2.2 秒，
   而正常情况零额外代价（第二级只在退化时触发，实测 3s/8s/27s 音频耗时与标量档持平）。
   **这条铁律的本意始终是"给最坏情况封顶"，不是"禁止重试"。**
2. **不做高频小增量重转。** Whisper 内部把 mel 补齐到 30 秒过 encoder，所以 2 秒块和 15 秒块几乎同价（都约 1 秒）。任何"每秒刷新"的设计都会掉队。
3. **音频永不落盘。** 只在内存活几秒，转完即弃。Utter 不产生录音文件。
4. **API key 只进 macOS Keychain**，不进 `.env`、不进 JSON、不进代码。
5. **不引入 torch。** VAD 用 silero 的 ONNX 权重 + onnxruntime（约 2M），不用拖 328M torch 的 pip 包。
6. **provider 抽象不可绕过。** 作者持有非 Apple Silicon 设备，MLX 只在 Apple Silicon 有 GPU 后端。任何直接调用 `mlx_whisper` 的代码都是 bug。
7. **不引入新脚手架。** 现有模块划分是干净的；模板残留（`productName: "frontend"` 等）是待清理项，不是待模仿项。

以下六条 2026-08-10 增补，全部服务于同一件事——**绝不静默丢字、绝不悄悄改字**。详见 v3 设计 §4.1。

8. **转录先于润色落定，且永不因润色失败而丢失。** 拿到转录就先进缓冲与存档；润色失败、超时、被限流，一律降级注入原始转录。宁可给一段没润色的原话，不给空白或错误提示。
9. **注入过的文字永不回改。** 一段只注入一次。禁止"先出草稿、LLM 回来再选中重打"——那会毁掉用户正在编辑的文档。
10. **润色提示词不得改写措辞、不得增删内容。** 只允许删口水词、补标点、分段。作者口述的是学术论证，被"润色"成通顺但走样的句子且事后难以察觉，是本项目最严重的失败模式。默认档位「轻」。原始转录必须与润色版一起存档。
11. **LLM 调用串行。** 同时只允许一个润色请求在飞，否则返回乱序会打乱注入顺序。积压时合并成一个更大的请求，不并发补发。
12. **长文本走剪贴板粘贴，不逐字模拟键盘。** 粘贴前备份用户原剪贴板，粘完还原。
13. **焦点离开目标窗口时缓冲，不强行注入。** 禁用 `CGEventPostToPid` 与辅助功能直改文本框值这两条路——它们在 Electron／浏览器／终端上随机失败，而**随机失败比明确失败更糟**：用户不会知道哪一段丢了。

## 借鉴来源

同类开源项目是这个领域的已知坑清单，遇到问题先查（全局规则 §八）。已确认可参考：

| 项目 | 已验证的做法 |
|---|---|
| [VoiceInk](https://github.com/Beingpax/VoiceInk) | 粘贴前查 `AXRole` 确认有输入框；剪贴板还原延迟**可配置**；type-out 作兜底；按 app 切配置 |
| [Handy](https://github.com/primaprashant/awesome-voice-typing) | 跨平台（Windows 参考） |
| openai/whisper [#976](https://github.com/openai/whisper/discussions/976) | code-switching 是设计限制：每 30s 窗口只认一个语言 |
| openai/whisper [#277](https://github.com/openai/whisper/discussions/277) | 繁简输出不稳定，已知问题 |

**但查开源不替代自己测量。** VoiceInk [#687](https://github.com/Beingpax/VoiceInk/issues/687) 至今开着的那个「间歇性空转录」，正是本项目量出来并修掉的首次开麦 713ms —— 他们复现不了，因为没埋点。

## 环境

- 开发机 Apple M2 / 16G。模型 `mlx-community/whisper-large-v3-turbo` 已在 `~/.cache/huggingface/`（1.6G），**不要重复下载**。
- 依赖环境按全局规则建在仓库外。**不要在项目内建 `.venv`**（本仓在 iCloud 同步区）：

```bash
uv venv ~/.venvs/utter
uv pip install --python ~/.venvs/utter/bin/python \
    --overrides requirements-overrides.txt -r requirements-dev.txt
uv pip install --python ~/.venvs/utter/bin/python --no-deps -e .
```

> 最后一行装的是本项目自身（editable），为的是 `utter` 成为一条真命令。
> **`--no-deps` 不能省**：本项目的 `pyproject.toml` 刻意不声明依赖，依赖只在
> `requirements.txt` 里，因为 mlx-whisper 谎报的 torch 必须靠 `--overrides` 挡掉。

> `--overrides` **不是可选的**。`mlx-whisper` 谎报依赖 torch，不加这个参数会拖进 476M
> 并直接违反铁律 5。证据见 `requirements-overrides.txt`。

- 查依赖用 `uv pip list --python ~/.venvs/utter/bin/python`。
  **`~/.venvs/utter/bin/pip list` 会假通过**——uv venv 不装 pip，管道 grep 的是空输出。
- 构建物（`target/`、`node_modules/`、`dist/`）随时可删，用完即清。

## 运行

装好后 `utter` 在任何目录都能用（**不要再写 `python -m backend.cli`，那个只在仓库目录内有效**）：

```bash
~/.venvs/utter/bin/utter doctor
~/.venvs/utter/bin/utter models list
~/.venvs/utter/bin/utter transcribe FILE.wav --mode listen --timing

# 听写。加 --target scratchpad 只暂存不注入
~/.venvs/utter/bin/utter dictate --timing
```

嫌路径长就 `export PATH="$HOME/.venvs/utter/bin:$PATH"`，之后直接敲 `utter`。

```bash
# 测试（要在仓库目录内跑）。默认跳过联网与真模型；全跑加 -m ""
~/.venvs/utter/bin/python -m pytest -q
```

**图形界面目前跑不起来，这是有意的。** v1 的后端服务器（`backend/main.py` 及
`session` / `transcriber` / `translator` / `audio_capture`）2026-08-10 已删除——它走的是被
实测证伪的旧管线，P3 无论如何都要在 v3 管线上重写。`frontend/src` 的界面组件保留，
v3 设计 §7 说那部分设计有效，是 P3 的起点。

听写的界面是 Python/AppKit 写的菜单栏 + 浮窗，随 `utter dictate` 一起起来，不需要 Tauri。

## 状态

当前在 `v3` 分支。551 个测试通过，无豁免、无 xfail。

- **P1 共享核心已验收**（2026-08-10）：provider 抽象、硬件探测、档位 catalog、模型下载器、
  VAD、配置与钥匙串、五槽管线、CLI。每句转录中位 1.18s、占空比 29%。
- **P2a 听写 6/8 完成**：麦克风、热键（左 Option 按住 / 双击左 Control 开关）、埋点计时、
  暂存模式、光标注入、常驻 daemon、菜单栏 + 浮窗，全部真机验证。端到端 968–1053ms。
  待办：Task 7 应用兼容表（需真机逐个 app 试注入）、Task 8 润色接线（**需作者先存 API key**；
  作者已定：用托管 API 的免费模型，不用本地 Ollama）。
- **模型选型已定**（2026-08-10）：默认 **Whisper turbo 自动档**——唯一中英文都拿得下的本地
  模型。SenseVoice 留作纯中文快档，但**它会吞掉中文句子里的英文**，名字里已写明。
  更大的开源模型（FireRedASR2、SenseVoice fp32）都实测更差，已删。

计划见 [P2a 实现计划](docs/plans/2026-08-10-v3-p2a-implementation.md)，
实测见 [P2a 延迟基准](docs/benchmarks/2026-08-10-p2a-dictation-latency.md)，
坑与不足见 [教训与经验](docs/教训与经验.md)。

无远端仓库（作者明示暂缓，勿反复劝）。
