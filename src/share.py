import os
import sys
import time
import subprocess
from pyngrok import ngrok

def share_app():
    print("🚀 正在启动 AI 素材库分享模式...")
    
    # 1. Start Streamlit in background
    # We use sys.executable to ensure we use the same python interpreter (venv)
    cmd = [sys.executable, "-m", "streamlit", "run", "src/app.py", "--server.headless", "true", "--browser.gatherUsageStats", "false"]
    
    print("   启动本地 Streamlit 服务...")
    streamlit_process = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL, # Silence stdout to keep terminal clean
        stderr=subprocess.PIPE     # Capture stderr in case of error
    )
    
    # Wait a bit for Streamlit to initialize
    time.sleep(3)
    
    if streamlit_process.poll() is not None:
        print("❌ Streamlit 启动失败。请检查错误输出：")
        print(streamlit_process.stderr.read().decode())
        return

    # 2. Setup Ngrok Tunnel
    print("🔗 正在建立公网隧道 (Ngrok)...")
    
    try:
        # Open a HTTP tunnel on the default port 8501
        # Pyngrok will automatically download the ngrok binary if needed
        public_url = ngrok.connect(8501).public_url
        
        print("\n" + "="*50)
        print(f"🎉 分享成功！你的应用已在公网可访问")
        print("="*50)
        print(f"\n🌍 专属访问链接: \033[1;32m{public_url}\033[0m")
        print(f"\n📋 操作指南:")
        print(f"1. 复制上面的链接发给你的朋友")
        print(f"2. 他们只需浏览器打开即可使用，无需任何配置")
        print(f"3. 保持此终端窗口开启，关闭窗口链接将失效")
        print("\n(按 Ctrl+C 结束分享)")
        print("="*50)
        
        # Keep the script running
        streamlit_process.wait()
        
    except KeyboardInterrupt:
        print("\n🛑 停止分享...")
    except Exception as e:
        print(f"\n❌ 隧道建立失败: {e}")
        if "ERR_NGROK_4018" in str(e) or "authentication" in str(e).lower() or "your account" in str(e).lower():
            print("\n⚠️ 提示: Ngrok 现在通常需要免费注册账户才能使用。")
            print("👉 请访问 https://dashboard.ngrok.com/signup 注册 (免费)")
            print("👉 复制 Authtoken 并运行: ngrok config add-authtoken <YOUR_TOKEN>")
            print("👉 然后再次运行此脚本即可")
    finally:
        # Cleanup
        if streamlit_process.poll() is None:
            streamlit_process.terminate()
        ngrok.kill()
        print("✅ 服务已关闭")

if __name__ == "__main__":
    share_app()
