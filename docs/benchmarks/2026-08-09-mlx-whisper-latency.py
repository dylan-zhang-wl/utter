"""LiveScribe v3 可行性基准：mlx-whisper large-v3-turbo 的分块转录成本。

实测环境：Apple M2 / 16G / macOS，mlx_whisper 0.4.3，
模型 mlx-community/whisper-large-v3-turbo（1.5G，已在 ~/.cache/huggingface）。
结果与解读见 docs/plans/2026-08-09-v3-design.md §3。

要回答的问题：实时管线每次只送 N 秒音频，短块是否更便宜？
——不是。Whisper 内部一律把 mel 补齐到 30s 再过 encoder，所以 2s 与 15s 几乎同价。
这一条判了 v2「每 1s 重转累积句」的 partial 设计死刑。

复现：
    ffmpeg -i <任意英语讲话> -ss 120 -t 60 -ar 16000 -ac 1 -c:a pcm_s16le bench60.wav
    python 2026-08-09-mlx-whisper-latency.py
"""
import time

import mlx_whisper
import soundfile as sf

REPO = "mlx-community/whisper-large-v3-turbo"
audio, sr = sf.read("bench60.wav", dtype="float32")
assert sr == 16000, f"需要 16kHz，实际 {sr}"


def _timed(fn) -> float:
    t0 = time.perf_counter()
    fn()
    return time.perf_counter() - t0

# 预热：首次调用含模型加载与 Metal kernel 编译（实测约 7.7s），不计入
mlx_whisper.transcribe(audio[: sr * 3], path_or_hf_repo=REPO, language="en")

print(f"{'块长':>6} | {'耗时':>7} | {'实时倍率':>8}")
print("-" * 32)
for chunk_s in (2, 3, 5, 8, 15, 30):
    seg = audio[: sr * chunk_s]
    best = min(
        _timed(lambda: mlx_whisper.transcribe(seg, path_or_hf_repo=REPO, language="en"))
        for _ in range(3)
    )
    print(f"{chunk_s:>5}s | {best:>6.2f}s | {chunk_s / best:>7.1f}x")

# v2 partial 策略压力测试：模拟一句 8 秒的话，每 1s 重转一次累积音频
total = sum(
    _timed(
        lambda: mlx_whisper.transcribe(
            audio[: sr * t], path_or_hf_repo=REPO, language="en"
        )
    )
    for t in range(1, 9)
)
print(f"\nv2 partial 占空比：{total:.1f}s GPU / 8s 说话 = {total / 8 * 100:.0f}%（>100% 即掉队）")

# temperature fallback：默认在难切块上会走 0.0→1.0 六档重试，单次可达 8s
for label, kw in [("默认（带 fallback）", {}), ("temperature=0.0", {"temperature": 0.0})]:
    seg = audio[: sr * 3]
    d = _timed(
        lambda: mlx_whisper.transcribe(seg, path_or_hf_repo=REPO, language="en", **kw)
    )
    print(f"{label:<24} 3s 块 → {d:.2f}s")
