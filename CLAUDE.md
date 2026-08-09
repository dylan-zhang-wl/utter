# Utter

本地优先的双模语音平台：**听**（英语讲座/会议实时转录 + 译中）与 **说**（听写，转成文字注入光标）。macOS 为主，需可分发到 Windows / Intel Mac。

> 项目原名 **LiveScribe**，2026-08-09 更名 Utter。v1/v2 文档里的 "LiveScribe" 指本项目旧名。

## 任务性质

这是**工具开发项目，不是学术写作项目**。全局 CLAUDE.md 的 LTL 心智在此降级——以任务正确性与清晰度为主。但作者是翻译研究学者、非软件工程背景：**解释技术决策时要落到具体，不要堆缩写**。术语参见 [docs/概念说明.md](docs/概念说明.md)。

## 权威文档

| 文档 | 地位 |
|---|---|
| [docs/plans/2026-08-09-v3-design.md](docs/plans/2026-08-09-v3-design.md) | **唯一现行设计权威源** |
| `docs/plans/2026-04-14-livescribe-v2-*.md` | **已作废**。存废清单见 v3 设计 §7 |
| `docs/plans/2026-04-13-livescribe-*.md` | v1 历史，仅供考古 |
| [docs/benchmarks/](docs/benchmarks/) | 实测数据，架构结论的依据 |

**不要再提议"执行现成的 v2 计划"。** v2 的 partial 管线已被实测证伪。

## 铁律

1. **实时管线必须 `temperature=0.0`。** 默认的 temperature fallback 阶梯会让最坏情况从 1 秒暴涨到 8–11 秒。实测见 benchmarks。
2. **不做高频小增量重转。** Whisper 内部把 mel 补齐到 30 秒过 encoder，所以 2 秒块和 15 秒块几乎同价（都约 1 秒）。任何"每秒刷新"的设计都会掉队。
3. **音频永不落盘。** 只在内存活几秒，转完即弃。Utter 不产生录音文件。
4. **API key 只进 macOS Keychain**，不进 `.env`、不进 JSON、不进代码。
5. **不引入 torch。** VAD 用 silero 的 ONNX 权重 + onnxruntime（约 2M），不用拖 328M torch 的 pip 包。
6. **provider 抽象不可绕过。** 作者持有非 Apple Silicon 设备，MLX 只在 Apple Silicon 有 GPU 后端。任何直接调用 `mlx_whisper` 的代码都是 bug。
7. **不引入新脚手架。** 现有模块划分是干净的；模板残留（`productName: "frontend"` 等）是待清理项，不是待模仿项。

## 环境

- 开发机 Apple M2 / 16G。模型 `mlx-community/whisper-large-v3-turbo` 已在 `~/.cache/huggingface/`（1.5G），**不要重复下载**。
- 依赖环境按全局规则建在仓库外：`uv venv ~/.venvs/utter`。**不要在项目内建 `.venv`**（本仓在 iCloud 同步区）。
- 构建物（`target/`、`node_modules/`、`dist/`）随时可删，用完即清。

## 运行

```bash
# 后端（终端 A）
~/.venvs/utter/bin/python -m backend.main
# 前端（终端 B）
cd frontend && npm run tauri dev
```

> **sidecar 链路目前是断的**，所以必须开两个终端。评估认为这就是 v1 建好后一次都没被用起来的直接原因。修复排在 P4，见 v3 设计 §6。

## 状态

当前在 `v3` 分支，处于**设计已定、P1 未动工**阶段。无远端仓库（作者明示暂缓，勿反复劝）。路线见 v3 设计 §8。
