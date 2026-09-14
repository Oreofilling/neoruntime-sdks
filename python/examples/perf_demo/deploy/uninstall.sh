#!/bin/bash
#
# perf-demo 完整回滚(install.sh 的逆操作):
#   停止并卸载 systemd 服务 → 还原 daemon yaml(从备份) → 删除 app 目录
#   venv 里的 SDK wheel 不动(需要钉回旧版时手动执行:
#   /data/venv-sdk/bin/pip install --no-deps neoruntime-ipc-sdk==0.6.0)
#
# 用法: ./uninstall.sh <ssh-target> [ssh-port]
# SSHPASS 非空时经 sshpass -e 认证。

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

TARGET="${1:-}"
PORT="${2:-22}"
if [ -z "$TARGET" ]; then
    echo "用法: $0 <ssh-target> [ssh-port]" >&2
    exit 1
fi

APP_DIR=/data/aipc/perf-demo
YAML=/data/aipc/etc/camera-daemon.yaml
YAML_BAK="$YAML.perf-demo.bak"

SSH_OPTS=(-p "$PORT" -o ConnectTimeout=8 -o StrictHostKeyChecking=no)
if [ -n "${SSHPASS:-}" ]; then
    command -v sshpass >/dev/null 2>&1 || { echo "错误: SSHPASS 已设置但缺 sshpass" >&2; exit 1; }
    SSH=(sshpass -e ssh "${SSH_OPTS[@]}")
else
    SSH=(ssh "${SSH_OPTS[@]}")
fi

echo "== [1/4] 停止并卸载服务 =="
"${SSH[@]}" "$TARGET" "systemctl stop perf-demo 2>/dev/null || true; \
    systemctl disable perf-demo 2>/dev/null || true; \
    rm -f /etc/systemd/system/perf-demo.service; systemctl daemon-reload"

echo "== [2/4] 还原 daemon yaml =="
"${SSH[@]}" "$TARGET" "
if [ -f $YAML_BAK ]; then
    mv -f $YAML_BAK $YAML
    echo '   yaml: 已从备份还原'
    systemctl restart camera-daemon
    sleep 3
else
    if grep -q '^injection:' $YAML; then
        echo '   警告: 发现 injection 段但无备份 —— 手动检查 $YAML' >&2
    else
        echo '   yaml: 本就无 injection 段,跳过'
    fi
fi
"

echo "== [3/4] 删除 app 目录 =="
"${SSH[@]}" "$TARGET" "rm -rf $APP_DIR /run/aipc/perf-demo.json"

echo "== [4/4] 完成 =="
echo "venv SDK 未回滚;需要钉回旧版时执行:"
echo "  ssh -p $PORT $TARGET '/data/venv-sdk/bin/pip install --no-deps neoruntime-ipc-sdk==0.6.0'"
