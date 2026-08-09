# LiveScribe — 实时语音转录与翻译工具 设计文档

## 背景

在英国留学，需要一个本地运行的桌面工具，实时将英语语音转录为文字并翻译为中文，辅助课堂学习。

## 需求总结

- 实时语音转文字（主要是英语）
- 音频来源可切换：麦克风（线下课）/ 系统音频（线上会议）
- 实时翻译：英→中，支持 OpenAI API 和 Google Translate
- 显示模式可切换：纯英文 / 英中并排 / 纯中文
- 后台运行：最小化到菜单栏，继续录音转录
- 自动保存 + 多格式导出（txt / markdown / srt）
- macOS M2 芯片，本地运行

## 架构

```
┌─────────────────────────────────┐
│     Tauri v2 前端 (Web UI)       │
│  HTML/CSS/JS - 实时显示转录/翻译   │
└──────────┬──────────────────────┘
           │ WebSocket
┌──────────▼──────────────────────┐
│     Python FastAPI 后端          │
│  ┌─────────────┐ ┌────────────┐ │
│  │ 语音识别模块  │ │  翻译模块   │ │
│  │ faster-whisper│ │ OpenAI API │ │
│  │ / mlx-whisper│ │ Google翻译  │ │
│  └─────────────┘ └────────────┘ │
│  ┌─────────────────────────────┐│
│  │ 音频捕获模块                  ││
│  │ 麦克风 / 系统音频(BlackHole)  ││
│  └─────────────────────────────┘│
└─────────────────────────────────┘
```

### 通信方式

Tauri 启动时以 sidecar 方式启动 Python 后端进程。前后端通过 WebSocket 实时通信：
- 后端 → 前端：推送转录文本、翻译结果、状态更新
- 前端 → 后端：控制指令（开始/停止/切换音源/切换翻译引擎）

## 技术选型

| 组件 | 技术 | 理由 |
|------|------|------|
| 前端框架 | Tauri v2 + React | 轻量（~30-50MB），现代美观，系统 WebView |
| 后端框架 | Python 3.11+ FastAPI | Whisper/AI 生态最成熟 |
| 语音识别 | faster-whisper（默认）/ mlx-whisper（备选） | M2 上高效，支持实时流式 |
| 音频捕获 | PyAudio（麦克风）+ BlackHole（系统音频） | 成熟稳定 |
| 语音活动检测 | Silero VAD | 准确检测语音段，减少无效识别 |
| 翻译引擎 | OpenAI API / googletrans | 双引擎可切换，用户自选 |
| 前后端通信 | WebSocket | 实时双向 |
| 数据存储 | 本地 JSON 文件 | 简单可靠，方便导出 |

## 核心模块设计

### 1. 音频捕获模块 (audio_capture.py)

- 抽象 AudioSource 接口，两个实现：MicrophoneSource / SystemAudioSource
- 统一输出 16kHz 单声道 PCM 音频流
- 支持运行时切换音源
- 系统音频需要用户预装 BlackHole，应用首次启动时检测并引导安装

### 2. 语音识别模块 (transcriber.py)

- 使用 faster-whisper 加载模型（推荐 base 或 small 模型，平衡速度和准确率）
- 结合 Silero VAD 做语音段检测，只在有语音时触发识别
- 流式输出：先输出部分结果（partial），语音段结束后输出最终结果（final）
- 模型首次使用时自动下载，缓存到本地

### 3. 翻译模块 (translator.py)

- 抽象 Translator 接口，两个实现：OpenAITranslator / GoogleTranslator
- OpenAI：使用 gpt-4o-mini（便宜快速），可配置 API key
- Google：使用 googletrans 库，免费但有速率限制
- 只翻译 final 结果，不翻译 partial（节省 API 调用）

### 4. 会话管理模块 (session.py)

- 每次录音为一个 Session，包含时间戳、音源类型、所有转录记录
- 自动保存为 JSON 文件到 ~/LiveScribe/sessions/
- 导出功能：txt（纯文本）、markdown（带格式）、srt（字幕格式）

### 5. 前端 UI

- 主界面：实时滚动显示转录文字，底部控制栏
- 控制栏：开始/停止按钮、音源切换、翻译引擎切换、显示模式切换
- 菜单栏图标：显示录音状态，点击可快速控制
- 设置页面：API key 配置、Whisper 模型选择、保存路径设置

## 项目结构

```
live-voice-transform/
├── docs/plans/                  # 设计文档
├── backend/                     # Python 后端
│   ├── main.py                  # FastAPI 入口 + WebSocket
│   ├── audio_capture.py         # 音频捕获
│   ├── transcriber.py           # 语音识别
│   ├── translator.py            # 翻译
│   ├── session.py               # 会话管理与导出
│   ├── config.py                # 配置管理
│   └── requirements.txt
├── frontend/                    # Tauri + React 前端
│   ├── src/                     # React 源码
│   ├── src-tauri/               # Tauri Rust 配置
│   └── package.json
└── README.md
```

## 参考项目

- KoljaB/RealtimeSTT — 实时语音转文字核心逻辑
- dieharders/example-tauri-v2-python-server-sidecar — Tauri + Python 集成模板
- solaoi/lycoris — Tauri + Whisper macOS 应用参考
- ExistentialAudio/BlackHole — macOS 系统音频捕获

## 未来扩展（不在 MVP 范围内）

- 说话人识别（区分老师和同学发言）
- AI 摘要（课后自动生成课程笔记）
- 多语言支持
