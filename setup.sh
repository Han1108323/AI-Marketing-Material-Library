#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_ROOT"

MODE="${1:-portfolio}"
case "$MODE" in
    portfolio) REQUIREMENTS="requirements.txt" ;;
    full) REQUIREMENTS="requirements-full.txt" ;;
    *) echo "用法: ./setup.sh [portfolio|full]"; exit 2 ;;
esac

echo "🚀 配置 TrendCrafter AI：$MODE"

# 1. Check if python3 exists
if ! command -v python3 &> /dev/null; then
    echo "❌ 未找到 Python 3。"
    exit 1
fi

# 2. Create Virtual Environment
if [ ! -d ".venv" ]; then
    echo "📦 创建虚拟环境 .venv..."
    python3 -m venv .venv
else
    echo "✅ 已检测到虚拟环境 .venv。"
fi

# 3. Activate and Install
echo "⬇️  Installing dependencies..."
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r "$REQUIREMENTS"

echo "🎉 安装完成。"
echo "👉 运行 ./run.sh $MODE 启动。"
