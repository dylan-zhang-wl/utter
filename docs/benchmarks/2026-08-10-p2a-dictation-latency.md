# P2a 听写延迟实测（2026-08-10）

机器 Apple M2 / 16G，`mlx-community/whisper-large-v3-turbo`（均衡档）。
目标：不开润色 ≤1500ms。

## 结论：达标，但有两个坑是只有真机跑才暴露的

| 按住时长 | 合计延迟 | 判定 |
|---|---|---|
| 2s | 1041 ms | ✅ |
| 4s | 968 ms | ✅ |
| 8s | 1053 ms | ✅ |

启动预热 3.3s（一次性，把 P1 记录的"第一句 2.76s"移出了用户路径）。

## 坑一：自动语言检测让延迟翻倍

8 秒音频，各测 3 次取中位：

| 设置 | 耗时 |
|---|---|
| `language="en"` | **1067 ms** |
| `language="zh"` | **1057 ms** |
| `language=None`（自动检测） | **1927 ms** |

**差额 860ms 是 Whisper 为了猜语言额外过的一遍 encoder。** 它比整个延迟预算的一半还多，
自动检测状态下无论如何撞不到 1500ms 目标。

配置项 `dictate_language` 默认仍是 `None`——作者中英双语写作，猜错语言比慢一点更糟——
但 `utter dictate` 启动时会明说这个代价。**作者确定某次是单语口述时，设一行配置省一半延迟。**

> 排查过程记一笔：最初怀疑是 pynput 的事件监听线程抢 CPU（macOS 事件 tap 跑高优先级 runloop）。
> 对照实验证否——有无监听器都是 1070ms。**不要凭直觉归咎于最可疑的组件。**

## 坑二：对着静音按热键，Whisper 会凭空编出句子

实测：房间无人说话时按住 2.5 秒，转录结果是 `"you"`，第二次是 `"Good job."`。

按住说模式的分段器是手指，不经过 VAD，所以这个洞一直开着。
**往作者文档里注入一句它自己编的话，是本项目能犯的最严重的错误**——比任何延迟问题都严重。

修法：转录前先用 silero 过一遍（`backend/vad.py: has_speech`）。代价 1–136ms，
静音时直接丢弃、模型根本不启动：

```
  hotkey → buffer closed          3 ms
  speech check                  100 ms
  transcription            (no speech)
  ────────────────────────────────────
  total                         103 ms      ← 而不是 1956ms + 一句假话
```

## 坑三：PortAudio 关流 130ms 挡在关键路径上

拆解 `end_utterance`：排空队列 0.1ms，拼接数组 0.1ms，**关闭音频流 130.6ms**。

改成先取音频入队、再异步关流后，`hotkey → buffer closed` 从 119ms 降到 **3ms**。

## 热键：右 Option 单键

按住 350ms 测得 334ms，开销约 8–16ms。左 Option 正确不触发。

pynput 有两个坑：`HotKey.parse("<alt_r>")` 产出裸键码而监听器报 `Key.alt_r`，两者永不相等；
且 `canonical()` 会把 `Key.alt_r` 折叠成 `Key.alt`。都已在 `backend/hotkey.py` 绕过。

## 剪贴板保真

五种 flavour（纯文本 / RTF / HTML / 二进制 PNG / macOS 派生的 UTF-16）写入再还原，
**逐字节一致**。铁律 12 的"粘完还原"成立。
