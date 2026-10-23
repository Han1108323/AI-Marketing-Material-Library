
import json
import os
import time
from typing import Optional, Dict, List
import hashlib
import threading
import secrets

USERS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "users.json")

class AuthManager:
    def __init__(self):
        self._lock = threading.Lock()
        self.users = self._load_users()
        self._ensure_super_admin()

    def _load_users(self) -> Dict:
        if not os.path.exists(USERS_FILE):
            return {}
        try:
            with open(USERS_FILE, 'r') as f:
                return json.load(f)
        except:
            return {}

    def _save_users(self):
        with self._lock:
            try:
                with open(USERS_FILE, 'w') as f:
                    json.dump(self.users, f, indent=2)
            except Exception:
                return

    def increment_upload_count(self, phone: str):
        """Increments the upload count for a user in a thread-safe way."""
        if phone in self.users:
            with self._lock:
                self.users[phone]["upload_count"] = self.users[phone].get("upload_count", 0) + 1
            self._save_users()

    def _ensure_super_admin(self):
        """Ensures the super admin exists."""
        admin_phone = "13800138000"
        # Never ship a reusable admin password. In full-pipeline deployments,
        # provide ADMIN_PASSWORD through the environment. A random secret keeps
        # the account non-loginable by default while demo auto-login still works.
        default_pwd = os.getenv("ADMIN_PASSWORD") or secrets.token_urlsafe(32)
        
        if admin_phone not in self.users:
            self.users[admin_phone] = {
                "phone": admin_phone,
                "password": self._hash_password(default_pwd), 
                "role": "admin",
                "whitelist": True,
                "created_at": 1765555200.0, # 2025-12-12
                "upload_count": 0
            }
            self._save_users()
        else:
            # Migrate the legacy development password if this database predates
            # the environment-based configuration.
            old_weak_hash = self._hash_password("admin123")
            current_hash = self.users[admin_phone].get("password")
            
            needs_save = False
            legacy_demo_hash = self._hash_password("Admin@2025_Secure!")
            if current_hash in {old_weak_hash, legacy_demo_hash}:
                self.users[admin_phone]["password"] = self._hash_password(default_pwd)
                needs_save = True
            
            if self.users[admin_phone].get("role") != "admin":
                self.users[admin_phone]["role"] = "admin"
                self.users[admin_phone]["whitelist"] = True
                needs_save = True
            
            # Fix date if it's the old one (optional, but good for consistency)
            # Just force update created_at for admin to requested date
            self.users[admin_phone]["created_at"] = 1765555200.0
            needs_save = True
            
            if needs_save:
                self._save_users()

    def _hash_password(self, password: str) -> str:
        return hashlib.sha256(password.encode()).hexdigest()

    def register(self, phone: str, password: str) -> tuple[bool, str]:
        """Registers a new user."""
        if not phone or not password:
            return False, "手机号或密码不能为空"
        
        if phone in self.users:
            return False, "该手机号已注册"

        self.users[phone] = {
            "phone": phone,
            "password": self._hash_password(password),
            "role": "user",
            "whitelist": False, # Default to False, needs admin approval
            "created_at": time.time(),
            "upload_count": 0
        }
        self._save_users()
        return True, "注册成功，请联系管理员添加白名单后登录"

    def login(self, phone: str, password: str) -> tuple[Optional[Dict], str]:
        """
        Login with phone and password.
        Returns (user_dict, error_message).
        """
        if not phone or not password:
            return None, "手机号或密码不能为空"
            
        user = self.users.get(phone)
        if not user:
            return None, "用户不存在"
        
        if user.get("password") != self._hash_password(password):
            return None, "密码错误"
            
        if not user.get("whitelist", False):
            # Super admin is always whitelisted by _ensure_super_admin
            return None, "您的账号未在白名单中，请联系管理员"
        
        return user, "登录成功"

    def toggle_whitelist(self, target_phone: str, status: bool) -> bool:
        """Admin function to toggle whitelist status."""
        if target_phone in self.users:
            self.users[target_phone]["whitelist"] = status
            self._save_users()
            return True
        return False

    def get_all_users(self) -> List[Dict]:
        """Returns all users with masked phones."""
        safe_users = []
        for p, data in self.users.items():
            safe_user = data.copy()
            del safe_user["password"] # Remove password
            safe_user['masked_phone'] = self.mask_phone(p)
            safe_users.append(safe_user)
        return safe_users

    def mask_phone(self, phone: str) -> str:
        if not phone: return ""
        if len(phone) >= 7:
            return f"{phone[:3]}xxxx{phone[-4:]}"
        return phone

    def increment_upload_count(self, phone: str):
        # Reload to minimize race conditions
        self.users = self._load_users()
        if phone in self.users:
            self.users[phone]['upload_count'] = self.users[phone].get('upload_count', 0) + 1
            self._save_users()

    def delete_user(self, target_phone: str, operator_role: str) -> tuple[bool, str]:
        if operator_role != 'admin':
            return False, "无权限"
        if target_phone == "13800138000":
            return False, "不能删除超级管理员账号"
        if target_phone not in self.users:
            return False, "用户不存在"
        del self.users[target_phone]
        self._save_users()
        return True, "删除成功"

    def reset_password(self, target_phone: str, new_password: str, operator_role: str) -> tuple[bool, str]:
        if operator_role != 'admin':
            return False, "无权限"
        if target_phone not in self.users:
            return False, "用户不存在"
        if not new_password or len(new_password) < 6:
            return False, "密码长度至少 6 位"
        self.users[target_phone]["password"] = self._hash_password(new_password)
        self._save_users()
        return True, "密码重置成功"

    def change_role(self, target_phone: str, new_role: str, operator_role: str) -> tuple[bool, str]:
        if operator_role != 'admin':
            return False, "无权限"
        if new_role not in {"user", "designer", "admin"}:
            return False, "角色不合法，可选: user / designer / admin"
        if target_phone not in self.users:
            return False, "用户不存在"
        self.users[target_phone]["role"] = new_role
        if new_role == "admin":
            self.users[target_phone]["whitelist"] = True
        self._save_users()
        return True, "角色修改成功"

    def update_profile(self, phone: str, shop_name: str = None, category: str = None,
                       channel_prefs: list = None) -> tuple[bool, str]:
        if phone not in self.users:
            return False, "用户不存在"
        u = self.users[phone]
        if shop_name is not None:
            u["shop_name"] = shop_name
        if category is not None:
            u["category"] = category
        if channel_prefs is not None:
            u["channel_prefs"] = channel_prefs
        if "updated_at" not in u:
            u["created_at"] = u.get("created_at", time.time())
        u["updated_at"] = time.time()
        self._save_users()
        return True, "商家配置保存成功"

    def get_user(self, phone: str) -> Optional[Dict]:
        u = self.users.get(phone)
        if not u:
            return None
        safe = u.copy()
        if "password" in safe:
            del safe["password"]
        return safe
