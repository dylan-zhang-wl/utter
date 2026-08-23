#!/bin/bash
# 在一台新 Mac 上从零装好 Utter。
#
#     bash packaging/setup_new_mac.sh
#
# 在仓库目录里跑（git clone 或整个文件夹拷过来都行）。每一步都做过就跳过，
# 所以中断了重跑、装坏了重跑，都安全。
#
# 它不做的三件事，结束时会提醒：
#   * API key —— 只进钥匙串（铁律 4），必须在新机器上重新输一次
#   * 系统权限 —— 辅助功能/麦克风/系统录音，macOS 只认人手点的
#   * 联网 —— 首次要下载约 1.7G 模型，在酒店 wifi 之前先在家跑完
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
VENV="${UTTER_VENV:-$HOME/.venvs/utter}"   # 仓库外、iCloud 外（全局规则 §七）
# 下面一律在仓库目录内用相对路径。不是风格问题：这个仓库的路径里有空格
# （「cc cowork」），而 uv 的 -r/--overrides 参数会把带空格的绝对路径从空格处
# 劈开（实测 error: File not found: `/Users/d/Desktop/华工大/cc`）。
cd "$REPO"
step() { printf '\n\033[1m== %s\033[0m\n' "$1"; }

if [ "$(uname)" != "Darwin" ]; then
    echo "这个脚本只管 macOS。Windows 版还没有做。" >&2; exit 1
fi
ARCH="$(uname -m)"
step "这台机器：$ARCH $(sw_vers -productVersion)"
if [ "$ARCH" = "arm64" ]; then
    echo "   Apple Silicon —— 转录走 GPU（MLX），和开发机同一条路。"
else
    echo "   Intel —— 转录走 CPU（faster-whisper），能用但明显更慢。"
fi
case "$(sw_vers -productVersion)" in
    1[0-3].*) echo "   ⚠ 系统低于 14.4：听记的「系统音频」不可用，麦克风与听写不受影响。";;
    14.[0-3]|14.[0-3].*) echo "   ⚠ 系统低于 14.4：听记的「系统音频」不可用，麦克风与听写不受影响。";;
esac

step "uv（Python 环境管理器）"
if ! command -v uv >/dev/null 2>&1 && [ ! -x "$HOME/.local/bin/uv" ]; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
command -v uv >/dev/null 2>&1 || export PATH="$HOME/.local/bin:$PATH"
echo "   $(uv --version)"

step "依赖环境 → $VENV"
[ -x "$VENV/bin/python" ] || uv venv "$VENV"
# --overrides 不是可选的：mlx-whisper 谎报依赖 torch，不挡会拖进 476M（铁律 5）
uv pip install --quiet --python "$VENV/bin/python" \
    --overrides requirements-overrides.txt -r requirements.txt
uv pip install --quiet --python "$VENV/bin/python" --no-deps -e .
echo "   $("$VENV/bin/python" --version)，utter 命令就绪"

step "语音模型（已下载的自动跳过）"
"$VENV/bin/python" - <<'PY'
from backend.models import ensure_model
for tier, why in (("balanced", "听写与听记的定稿（1.6G）"),
                  ("minimal", "听记的灰色预览（80M）")):
    print(f"   {tier} —— {why}")
    ensure_model(tier, on_progress=lambda p: print(f"      {p.message}"))
PY

step "签名证书（没有它，每次重装权限都会丢）"
echo "   macOS 可能弹窗要登录密码 —— 那是钥匙串在问，不是我们。"
"$VENV/bin/python" packaging/build_app.py --make-cert

step "打包到 /Applications/Utter.app（开机自启）"
"$VENV/bin/python" packaging/build_app.py --login-item

step "装完了。还剩三件事必须人来做："
cat <<'DONE'

   1. API key（翻译、润色、纪要要用；转录本身不用）：
          ~/.venvs/utter/bin/utter key openai
      钥匙串不跟机器走，旧电脑上的 key 拷不过来，重新输一次。

   2. 系统权限（系统设置 → 隐私与安全性，各自把 Utter 打开）：
          辅助功能     —— 听写热键
          输入监控     —— 听写热键
          麦克风       —— 听写 + 听记
          屏幕与系统录音 —— 听记录电脑里的声音（线上会议）
      前两个不开，按住 Option 没反应也不报错 —— 就是这里。

   3. 验证：
          ~/.venvs/utter/bin/utter doctor
      然后双击 /Applications/Utter.app，试一句听写。

DONE
