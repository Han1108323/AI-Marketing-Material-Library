
import os
import streamlit as st
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

# Global variable for non-Streamlit contexts (Singleton)
_GLOBAL_CLIENT = None

@st.cache_resource
def get_cached_qdrant_client():
    """
    Streamlit cached wrapper for Qdrant client.
    Ensures only one connection is created across re-runs.
    """
    return _create_qdrant_client()

def get_qdrant_client():
    """
    Unified entry point for getting Qdrant Client.
    Uses Streamlit cache if running in Streamlit, otherwise uses global singleton.
    """
    try:
        # Check if running in Streamlit
        import streamlit.runtime.scriptrunner
        if streamlit.runtime.exists():
             return get_cached_qdrant_client()
    except ImportError:
        pass
        
    # Fallback for CLI/Script usage (Singleton pattern)
    global _GLOBAL_CLIENT
    if _GLOBAL_CLIENT is None:
        _GLOBAL_CLIENT = _create_qdrant_client()
    return _GLOBAL_CLIENT

def _create_qdrant_client():
    """
    Internal factory method to create the client.
    Prioritizes Cloud connection from env vars.
    Falls back to local disk storage './local_qdrant_db' if Cloud is unreachable.
    """
    url = os.getenv("QDRANT_URL")
    api_key = os.getenv("QDRANT_API_KEY")
    
    # 1. Try Cloud
    # FORCE LOCAL MODE: The user is experiencing instability with Cloud.
    # We will temporarily bypass Cloud to ensure stability.
    # if url and "localhost" not in url and "127.0.0.1" not in url:
    #     try:
    #         print(f"🔌 Connecting to Qdrant Cloud: {url}...")
    #         # Increase timeout for stability (was 3s)
    #         client = QdrantClient(url=url, api_key=api_key, timeout=20) 
    #         # Test connection
    #         client.get_collections()
    #         print("✅ Connected to Qdrant Cloud")
    #         return client
    #     except (Exception, ResponseHandlingException, UnexpectedResponse) as e:
    #         print(f"⚠️ Cloud Connection Failed: {e}")
    #         print("🔄 Falling back to Local Storage...")
    #         # st.toast removed to avoid Streamlit caching errors
            
    # 2. Local Fallback
    local_path = "./local_qdrant_db"
    if not os.path.exists(local_path):
        os.makedirs(local_path)
        
    print(f"📂 Using Local Qdrant at: {local_path}")
    # Force single-instance mode for local file
    client = QdrantClient(path=local_path)
    return client
