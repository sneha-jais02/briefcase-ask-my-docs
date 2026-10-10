import base64
import hashlib
import hmac
import json
import secrets
import time
from dataclasses import dataclass
from typing import Annotated, Any

from fastapi import Depends, Header, HTTPException

from app import store
from app.config import get_settings

TOKEN_LIFETIME_SECONDS = 8 * 60 * 60
PASSWORD_SCRYPT_N = 2**14
PASSWORD_SCRYPT_R = 8
PASSWORD_SCRYPT_P = 1


@dataclass(frozen=True)
class UserContext:
    id: str
    workspace_id: str
    email: str
    role: str


def _b64encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64decode(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


def _jwt_secret() -> bytes:
    configured = get_settings().jwt_secret or get_settings().api_key
    if len(configured) < 32:
        raise RuntimeError("Set JWT_SECRET to a random secret of at least 32 characters.")
    return configured.encode("utf-8")


def hash_password(password: str) -> str:
    if len(password) < 12:
        raise ValueError("Passwords must contain at least 12 characters.")
    salt = secrets.token_bytes(16)
    derived = hashlib.scrypt(password.encode("utf-8"), salt=salt, n=PASSWORD_SCRYPT_N, r=PASSWORD_SCRYPT_R, p=PASSWORD_SCRYPT_P)
    return f"scrypt${PASSWORD_SCRYPT_N}${PASSWORD_SCRYPT_R}${PASSWORD_SCRYPT_P}${_b64encode(salt)}${_b64encode(derived)}"


def verify_password(password: str, encoded: str) -> bool:
    try:
        algorithm, n, r, p, salt, expected = encoded.split("$")
        if algorithm != "scrypt":
            return False
        actual = hashlib.scrypt(
            password.encode("utf-8"),
            salt=_b64decode(salt),
            n=int(n),
            r=int(r),
            p=int(p),
        )
        return hmac.compare_digest(actual, _b64decode(expected))
    except (ValueError, TypeError):
        return False


def issue_token(user: UserContext) -> str:
    now = int(time.time())
    header = _b64encode(json.dumps({"alg": "HS256", "typ": "JWT"}, separators=(",", ":")).encode())
    payload = _b64encode(json.dumps({
        "sub": user.id,
        "workspace_id": user.workspace_id,
        "role": user.role,
        "iat": now,
        "exp": now + TOKEN_LIFETIME_SECONDS,
    }, separators=(",", ":")).encode())
    signing_input = f"{header}.{payload}".encode("ascii")
    signature = hmac.new(_jwt_secret(), signing_input, hashlib.sha256).digest()
    return f"{header}.{payload}.{_b64encode(signature)}"


def _decode_token(token: str) -> dict[str, Any]:
    try:
        header, payload, signature = token.split(".")
        signing_input = f"{header}.{payload}".encode("ascii")
        expected = hmac.new(_jwt_secret(), signing_input, hashlib.sha256).digest()
        if not hmac.compare_digest(expected, _b64decode(signature)):
            raise ValueError
        claims = json.loads(_b64decode(payload))
        if not isinstance(claims, dict) or int(claims["exp"]) <= int(time.time()):
            raise ValueError
        return claims
    except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=401, detail="The access token is invalid or expired.") from exc


def current_user(authorization: str | None = Header(default=None)) -> UserContext:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Sign in to access this workspace.")
    claims = _decode_token(authorization[7:])
    user = store.get_user_by_id(str(claims.get("sub", "")))
    if (
        not user
        or not user["active"]
        or str(user["workspace_id"]) != claims.get("workspace_id")
        or user["role"] != claims.get("role")
    ):
        raise HTTPException(status_code=401, detail="This account or workspace access is no longer active.")
    return UserContext(
        id=str(user["id"]),
        workspace_id=str(user["workspace_id"]),
        email=user["email"],
        role=user["role"],
    )


AuthenticatedUser = Annotated[UserContext, Depends(current_user)]


def require_roles(*allowed_roles: str):
    def dependency(user: AuthenticatedUser) -> UserContext:
        if user.role not in allowed_roles:
            raise HTTPException(status_code=403, detail="Your workspace role does not allow this action.")
        return user

    return dependency
