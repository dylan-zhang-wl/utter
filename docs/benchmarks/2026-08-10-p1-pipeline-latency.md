# P1 管线端到端实测（2026-08-10）

**机器：** Apple M2 / 16G / macOS，`mlx_whisper 0.4.3`，`mlx 0.32.0`
**模型：** `mlx-community/whisper-large-v3-turbo`（均衡档，1614 MB，本机已缓存）
**素材：** `docs/benchmarks/make-fixture.sh` 生成，macOS `say -v Daniel`，46.9s，16kHz 单声道

> **这是延迟基准，不是准确率基准。** 素材是合成语音——无口音、无口水词、无环境噪声。
> `temperature=0.0` 关掉了 fallback 重试，且 Whisper 无论内容一律把 mel 补到 30s，
> 所以**单块耗时几乎与内容无关**，这正是要测的东西。准确率必须用作者本人的真实录音，
> 排在 P2b（设计 §9 待决 3）。

## 复现

```bash
docs/benchmarks/make-fixture.sh
~/.venvs/utter/bin/python -m backend.cli transcribe "$TMPDIR/utter-fixtures/bench-say-en.wav" \
    --mode listen --language en --timing
```

## 结果

VAD 从 46.9s 音频里切出 **10 段**，与素材的 10 个句子一一对应。

| 指标 | 实测 | 设计 §3 的预期 | 判定 |
|---|---|---|---|
| 每句转录（中位） | **1.18s** | 1.0–1.2s | ✅ |
| 每句转录（最小） | 1.14s | — | — |
| 每句转录（最大） | 2.76s | — | ⚠️ 见下 |
| 占空比 | **29%** | 20–25% | ✅ 同量级，远低于 100% |
| VAD 自身开销 | 0.18s / 46.9s = **0.4%** | 未预估 | ✅ 几乎免费 |
| 全程墙钟 | 13.6s / 46.9s 音频 | — | 3.4x 实时 |

**最大值 2.76s 是第一句**，含模型加载与 Metal kernel 首次编译；第二句起即回落到 1.20s。
真实使用中这一次性开销应在启动时预热掉，不该落在用户的第一句话上——**P2a 待办**。

**占空比 29% 对比 v2 的 172–221%**（设计 §3b）。差别不在调参，在于 v2 每秒对累积音频重转一次，
而这里每段音频只过一次模型。

## 附带发现：词表在源头修正术语，零延迟代价

第 5 句里 "foreignisation" 被听成了 **"Thorinization"**——一个翻译研究的核心术语被听成了托尔金。
把术语表作为 `initial_prompt` 喂进去后：

| | 耗时 | 输出 |
|---|---|---|
| 不带词表 | 1.15s | **Thorinization**, by contrast, keeps the reader aware… |
| 带词表 | 1.14s | **foreignisation**, by contrast, keeps the reader aware… |

词表内容：`Venuti, domestication, foreignisation, translation studies`

**这条把设计 §4.1g 第 1 层从主张变成了结论**：术语纠正应该发生在**转录时**而不是交给下游 LLM。
它零延迟、在关闭润色时照常生效，而且修的是 LLM 根本无从修复的错误——"Thorinization" 与
"foreignisation" 在字面上相去太远，下游模型没有任何线索能猜回去。

## 与 §3 原始基准的差异

§3 测的是**裸模型**（直接调 `mlx_whisper.transcribe`），本次测的是**整条管线**（含 VAD、
分段、pipeline 调度）。

| 块长 | §3 裸模型 | 本次管线 |
|---|---|---|
| 3s | 0.94–1.00s | ~1.07s |
| 8s | 1.03–1.04s | ~1.21s |
| 15s | 1.11–1.20s | ~1.34s |

管线高出约 10–15%。VAD 只占 0.4%，所以差额主要不是本项目的代码开销；更可能是
运行时状态差异（本次为长时间会话，设备发热与 Metal 上下文状态与 §3 的短脚本不同）。
**未进一步定位**——量级正确、离 8s 失败模式很远，不值得在 P1 深挖。若 P2 实测发现
延迟继续爬升，从这里开始查。
