#!/bin/bash
#
# perf-demo 设备端安装(在 SDK 仓的 scripts 约定之上):
#   本地构建 wheel → 部署 app + wheel 到设备 venv → 追加 daemon yaml
#   injection 段(备份) → 安装 systemd 服务并启动
#
# 用法:
#   ./install.sh <ssh-target> [ssh-port] --model <设备侧模型路径> [-- extra-args]
#   例: ./install.sh root@<设备IP> 22 \
#         --model /data/aipc-data/containerd/.../hailo_yolov8n_384_640.hef \
#         -- --a-stream third --b-stream sub
#
# SSHPASS 非空时经 sshpass -e 认证(备用设备公钥不通)。
# 回滚: ./uninstall.sh <ssh-target> [ssh-port]  (yaml/服务/app 全还原;
#   venv wheel 回滚另行执行: /data/venv-sdk/bin/pip install --no-deps neoruntime-ipc-sdk==0.6.0)
#
# 退出码: 0 = 安装并启动成功

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"   # .../sdks/python
SDK_DIR="$PROJECT_ROOT"
OUTPUT_DIR="$PROJECT_ROOT/dist"

TARGET="${1:-}"
MODEL_PATH=""
EXTRA_ARGS=()
if [ -z "$TARGET" ]; then
    echo "用法: $0 <ssh-target> [ssh-port] --model <设备侧模型路径> [-- extra-args]" >&2
    exit 1
fi
shift
PORT=22
if [ "$#" -ge 1 ] && [[ "$1" =~ ^[0-9]+$ ]]; then PORT="$1"; shift; fi
while [ $# -gt 0 ]; do
    case "$1" in
        --model) MODEL_PATH="$2"; shift 2 ;;
        --) shift; EXTRA_ARGS=("$@"); break ;;
        *) echo "未知参数: $1" >&2; exit 1 ;;
    esac
done

if [ -z "$TARGET" ] || [ -z "$MODEL_PATH" ]; then
    echo "用法: $0 <ssh-target> [ssh-port] --model <设备侧模型路径> [-- app args]" >&2
    exit 1
fi

APP_DIR=/data/aipc/perf-demo
VENV=/data/venv-sdk
YAML=/data/aipc/etc/camera-daemon.yaml
YAML_BAK="$YAML.perf-demo.bak"

SSH_OPTS=(-p "$PORT" -o ConnectTimeout=8 -o StrictHostKeyChecking=no)
if [ -n "${SSHPASS:-}" ]; then
    command -v sshpass >/dev/null 2>&1 || { echo "错误: SSHPASS 已设置但缺 sshpass" >&2; exit 1; }
    SSH=(sshpass -e ssh "${SSH_OPTS[@]}")
    RSYNC_SSH="sshpass -e ssh -p $PORT -o ConnectTimeout=8"
else
    SSH=(ssh "${SSH_OPTS[@]}")
    RSYNC_SSH="ssh -p $PORT -o ConnectTimeout=8"
fi
RSYNC=(rsync -e "$RSYNC_SSH")

echo "== [1/6] 构建 SDK wheel =="
mkdir -p "$OUTPUT_DIR"
( cd "$SDK_DIR" && python3 -m build --wheel --outdir "$OUTPUT_DIR" --skip-dependency-check >/dev/null )
WHEEL="$(ls -t "$OUTPUT_DIR"/neoruntime_ipc_sdk-*.whl | head -1)"
echo "   wheel: $(basename "$WHEEL")"

echo "== [2/6] 部署 app + wheel =="
"${SSH[@]}" "$TARGET" "mkdir -p $APP_DIR"
"${RSYNC[@]}" -a --delete --exclude deploy --exclude __pycache__ \
    "$SCRIPT_DIR"/../ "$TARGET:$APP_DIR/"
"${RSYNC[@]}" -a "$WHEEL" "$TARGET:$APP_DIR/"
"${SSH[@]}" "$TARGET" "test -f '$MODEL_PATH'" || {
    echo "错误: 设备侧模型不存在: $MODEL_PATH" >&2; exit 1; }
"${SSH[@]}" "$TARGET" "$VENV/bin/pip install --quiet --no-deps --force-reinstall \
    $APP_DIR/$(basename "$WHEEL")"

echo "== [3/6] daemon yaml 追加 injection 段(备份后) =="
"${SSH[@]}" "$TARGET" "
set -e
if ! grep -q '^injection:' $YAML; then
    cp -n $YAML $YAML_BAK
    printf '\ninjection:\n  enabled: true\n' >> $YAML
    echo '   yaml: injection 段已追加(备份为 $YAML_BAK)'
    systemctl restart camera-daemon
    sleep 3
else
    echo '   yaml: injection 段已存在,跳过'
fi
"

echo "== [4/6] 安装 systemd 服务 =="
sed -e "s|@APP_DIR@|$APP_DIR|g" \
    -e "s|@PYTHON@|$VENV/bin/python3|g" \
    -e "s|@MODEL_PATH@|$MODEL_PATH|g" \
    -e "s|@EXTRA_ARGS@|${EXTRA_ARGS[*]-}|g" \
    "$SCRIPT_DIR/perf-demo.service" > /tmp/perf-demo.service
"${RSYNC[@]}" -a /tmp/perf-demo.service "$TARGET:/etc/systemd/system/perf-demo.service"

echo "== [5/6] 启动 =="
"${SSH[@]}" "$TARGET" "systemctl daemon-reload && systemctl enable --now perf-demo"

echo "== [6/6] 健康检查 =="
sleep 5
"${SSH[@]}" "$TARGET" "systemctl --no-pager -l status perf-demo | head -12; \
    test -f /run/aipc/perf-demo.json && echo '   JSON: /run/aipc/perf-demo.json 已生成' || true"
echo "完成。日志: ssh $TARGET 'journalctl -u perf-demo -f'"
