#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_ROOT"

MODE="${1:-portfolio}"
case "$MODE" in
    portfolio) ENTRYPOINT="src/portfolio_app.py" ;;
    full) ENTRYPOINT="src/app.py" ;;
    *) echo "用法: ./run.sh [portfolio|full]"; exit 2 ;;
esac

if [ ! -x ".venv/bin/python" ]; then
    echo "未检测到虚拟环境，请先运行 ./setup.sh（完整版本使用 ./setup.sh full）。"
    exit 1
fi

echo "🚀 启动 TrendCrafter AI：$MODE"
exec .venv/bin/python -m streamlit run "$ENTRYPOINT"
