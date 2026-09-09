#!/bin/bash
echo "🚀 启动 MuseFlow AI 营销素材中台..."

# Activate virtual environment
if [ -d ".venv" ]; then
    source .venv/bin/activate
else
    echo "⚠️ 未检测到 .venv，尝试直接运行..."
fi

# Run Streamlit
echo "🌐 正在启动 Web 服务..."
streamlit run src/portfolio_app.py
