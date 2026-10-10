import uuid
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from app import auth, main
from app.auth import UserContext


def configure_test_app(monkeypatch, role: str = "platform_admin") -> None:
    settings = SimpleNamespace(
        api_key="test-secret",
        jwt_secret="j" * 48,
        max_upload_bytes=10_000,
        bootstrap_admin_password="a-strong-bootstrap-password",
    )
    monkeypatch.setattr(main, "get_settings", lambda: settings)
    monkeypatch.setattr(auth, "get_settings", lambda: settings)
    monkeypatch.setattr(main.store, "initialize_schema", lambda: None)
    user = UserContext(
        id="3e5b7a24-41c7-4628-8ac1-b1da30b759ee",
        workspace_id="f3fd5a23-166c-4616-9b73-fc50f3d892fe",
        email="admin@example.test",
        role=role,
    )
    main.app.dependency_overrides[auth.current_user] = lambda: user


def test_health_is_public_and_documents_require_a_bearer_user(monkeypatch) -> None:
    configure_test_app(monkeypatch)
    main.app.dependency_overrides.clear()
    def unauthorized():
        raise HTTPException(status_code=401, detail="Sign in.")

    main.app.dependency_overrides[auth.current_user] = unauthorized
    monkeypatch.setattr(main.store, "list_documents", lambda workspace_id: [])

    with TestClient(main.app) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/api/documents").status_code == 401


def test_upload_is_workspace_scoped_and_viewers_are_read_only(monkeypatch) -> None:
    configure_test_app(monkeypatch, role="viewer")
    monkeypatch.setattr(main.embeddings, "embed_texts", lambda texts: [[0.0] * 384 for _ in texts])
    captured: dict = {}
    monkeypatch.setattr(main.store, "ingest_document", lambda **kwargs: captured.update(kwargs) or {
        "document_id": "doc-1", "version": 1, "unchanged": False
    })

    with TestClient(main.app) as client:
        response = client.post(
            "/api/documents",
            data={"title": "Test Case", "source_url": "https://example.test/case"},
            files={"file": ("case.txt", b"SECTION 1.1 Notice\nThirty days' written notice is required.", "text/plain")},
        )

    assert response.status_code == 403
    assert captured == {}


def test_editors_can_upload_only_into_their_workspace(monkeypatch) -> None:
    configure_test_app(monkeypatch, role="editor")
    captured: dict = {}
    monkeypatch.setattr(main.embeddings, "embed_texts", lambda texts: [[0.0] * 384 for _ in texts])

    def fake_ingest(**kwargs):
        captured.update(kwargs)
        return {"document_id": "doc-1", "version": 1, "unchanged": False}

    monkeypatch.setattr(main.store, "ingest_document", fake_ingest)
    with TestClient(main.app) as client:
        response = client.post(
            "/api/documents",
            data={"title": "Test Case", "source_url": "https://example.test/case"},
            files={"file": ("case.txt", b"SECTION 1.1 Notice\nThirty days' written notice is required.", "text/plain")},
        )

    assert response.status_code == 200
    assert str(captured["workspace_id"]) == "f3fd5a23-166c-4616-9b73-fc50f3d892fe"
    assert captured["document_date"] is None


def test_admin_can_create_editor_but_platform_admin_is_never_a_workspace_role(monkeypatch) -> None:
    configure_test_app(monkeypatch, role="admin")
    captured: dict = {}
    monkeypatch.setattr(main.store, "create_user", lambda *args: captured.update({"args": args}) or {
        "id": "new-user", "workspace_id": str(args[0]), "email": args[1], "role": args[3]
    })

    with TestClient(main.app) as client:
        created = client.post(
            "/api/users",
            json={"email": "editor@example.test", "password": "a-strong-test-password", "role": "editor"},
        )
        forbidden = client.post(
            "/api/users",
            json={"email": "root@example.test", "password": "a-strong-test-password", "role": "platform_admin"},
        )

    assert created.status_code == 201
    assert created.json()["role"] == "editor"
    assert captured["args"][3] == "editor"
    assert forbidden.status_code == 422


def test_platform_admin_can_create_a_separate_client_workspace(monkeypatch) -> None:
    configure_test_app(monkeypatch, role="platform_admin")
    captured: dict = {}
    monkeypatch.setattr(main.store, "create_workspace", lambda *args: captured.update({"args": args}) or {
        "id": "new-workspace", "name": args[0], "admin_email": args[1]
    })

    with TestClient(main.app) as client:
        response = client.post(
            "/api/workspaces",
            json={
                "name": "Client Legal",
                "admin_email": "client@example.test",
                "admin_password": "a-strong-client-password",
            },
        )

    assert response.status_code == 201
    assert response.json()["name"] == "Client Legal"
    assert captured["args"][1] == "client@example.test"


def test_ask_history_is_bound_to_authenticated_user_and_workspace(monkeypatch) -> None:
    configure_test_app(monkeypatch, role="viewer")
    saved: dict = {}
    monkeypatch.setattr(main.store, "get_history", lambda *args: [{"role": "user", "content": "Earlier question"}])
    monkeypatch.setattr(main.store, "save_turn", lambda *args: saved.update({"turn": args}))
    monkeypatch.setattr(
        main.rag_graph,
        "invoke",
        lambda state: {"result": {"answer": "Thirty days [C1].", "citations": [{"id": "C1"}]}},
    )

    with TestClient(main.app) as client:
        response = client.post(
            "/api/ask",
            json={"question": "How much notice?", "session_id": "ebf26b70-5721-4df0-8f62-c1d46cced9b1"},
        )

    assert response.status_code == 200
    assert saved["turn"][1:3] == (
        uuid.UUID("f3fd5a23-166c-4616-9b73-fc50f3d892fe"),
        uuid.UUID("3e5b7a24-41c7-4628-8ac1-b1da30b759ee"),
    )


def test_login_returns_a_signed_token_for_a_valid_password(monkeypatch) -> None:
    configure_test_app(monkeypatch)
    password_hash = auth.hash_password("a-strong-login-password")
    monkeypatch.setattr(main.store, "get_user_by_email", lambda email: {
        "id": "3e5b7a24-41c7-4628-8ac1-b1da30b759ee",
        "workspace_id": "f3fd5a23-166c-4616-9b73-fc50f3d892fe",
        "email": email,
        "password_hash": password_hash,
        "role": "admin",
        "active": True,
    })

    with TestClient(main.app) as client:
        response = client.post(
            "/api/auth/login",
            json={"email": "admin@example.test", "password": "a-strong-login-password"},
        )

    assert response.status_code == 200
    assert response.json()["token_type"] == "bearer"
    assert response.json()["user"]["role"] == "admin"


def test_signed_tokens_reject_tampering_and_expiration(monkeypatch) -> None:
    configure_test_app(monkeypatch)
    user = UserContext(
        id="3e5b7a24-41c7-4628-8ac1-b1da30b759ee",
        workspace_id="f3fd5a23-166c-4616-9b73-fc50f3d892fe",
        email="admin@example.test",
        role="admin",
    )
    monkeypatch.setattr(auth.time, "time", lambda: 1000)
    token = auth.issue_token(user)
    assert auth._decode_token(token)["sub"] == user.id

    header, payload, signature = token.split(".")
    with pytest.raises(HTTPException, match="invalid or expired"):
        changed_signature = ("A" if signature[0] != "A" else "B") + signature[1:]
        auth._decode_token(f"{header}.{payload}.{changed_signature}")

    monkeypatch.setattr(auth.time, "time", lambda: 1000 + auth.TOKEN_LIFETIME_SECONDS)
    with pytest.raises(HTTPException, match="invalid or expired"):
        auth._decode_token(token)


def test_authentication_rechecks_current_workspace_role_and_active_status(monkeypatch) -> None:
    configure_test_app(monkeypatch)
    context = UserContext(
        id="3e5b7a24-41c7-4628-8ac1-b1da30b759ee",
        workspace_id="f3fd5a23-166c-4616-9b73-fc50f3d892fe",
        email="admin@example.test",
        role="admin",
    )
    token = auth.issue_token(context)
    user = {
        "id": context.id,
        "workspace_id": context.workspace_id,
        "email": context.email,
        "role": context.role,
        "active": True,
    }
    monkeypatch.setattr(auth.store, "get_user_by_id", lambda user_id: user)
    assert auth.current_user(f"Bearer {token}") == context

    user["active"] = False
    with pytest.raises(HTTPException) as error:
        auth.current_user(f"Bearer {token}")
    assert error.value.status_code == 401


def test_workspace_viewers_cannot_create_accounts(monkeypatch) -> None:
    configure_test_app(monkeypatch, role="viewer")
    with TestClient(main.app) as client:
        response = client.post(
            "/api/users",
            json={"email": "new@example.test", "password": "a-strong-test-password", "role": "viewer"},
        )
    assert response.status_code == 403
