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
| [docs/plans/2026-08-10-p1-build-log.md](docs/plans/2026-08-10-p1-build-log.md) | P1 施工日志：自主决定、被推翻的旧说法、待办 |

**不要再提议"执行现成的 v2 计划"。** v2 的 partial 管线已被实测证伪。

## 铁律

1. **实时管线必须 `temperature=0.0`。** 默认的 temperature fallback 阶梯会让最坏情况从 1 秒暴涨到 8–11 秒。实测见 benchmarks。
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

## 环境

- 开发机 Apple M2 / 16G。模型 `mlx-community/whisper-large-v3-turbo` 已在 `~/.cache/huggingface/`（1.6G），**不要重复下载**。
- 依赖环境按全局规则建在仓库外。**不要在项目内建 `.venv`**（本仓在 iCloud 同步区）：

```bash
uv venv ~/.venvs/utter
uv pip install --python ~/.venvs/utter/bin/python \
    --overrides requirements-overrides.txt -r requirements-dev.txt
```

> `--overrides` **不是可选的**。`mlx-whisper` 谎报依赖 torch，不加这个参数会拖进 476M
> 并直接违反铁律 5。证据见 `requirements-overrides.txt`。

- 查依赖用 `uv pip list --python ~/.venvs/utter/bin/python`。
  **`~/.venvs/utter/bin/pip list` 会假通过**——uv venv 不装 pip，管道 grep 的是空输出。
- 构建物（`target/`、`node_modules/`、`dist/`）随时可删，用完即清。

## 运行

```bash
# P1 的命令行（可用）
~/.venvs/utter/bin/python -m backend.cli doctor
~/.venvs/utter/bin/python -m backend.cli models list
~/.venvs/utter/bin/python -m backend.cli transcribe FILE.wav --mode listen --timing

# 测试。默认跳过联网与真模型；全跑加 -m ""
~/.venvs/utter/bin/python -m pytest -q
```

v1 的图形界面（P3 前仍是旧管线）：

```bash
# 后端（终端 A）
~/.venvs/utter/bin/python -m backend.main
# 前端（终端 B）
cd frontend && npm run tauri dev
```

> **sidecar 链路目前是断的**，所以必须开两个终端。评估认为这就是 v1 建好后一次都没被用起来的直接原因。修复排在 **P2b**（2026-08-10 从 P4 提前），见 v3 设计 §6、§8。

## 状态

当前在 `v3` 分支。**P1 共享核心已完成并验收（2026-08-10）**：provider 抽象、硬件探测、
档位 catalog、模型下载器、VAD、配置与钥匙串、五槽管线、CLI 全部就绪，295 个测试通过，
每句转录中位 1.18s、占空比 29%。

下一步 **P2a**：无界面的最小可用听写（热键、暂存模式、默认关润色）。规格见 v3 设计 §4.1，
排期见 §8，遗留待办见 [P1 施工日志](docs/plans/2026-08-10-p1-build-log.md) §六。

无远端仓库（作者明示暂缓，勿反复劝）。
