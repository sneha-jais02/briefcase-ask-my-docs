import hashlib
import io
import uuid
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path
from typing import Annotated
from zipfile import BadZipFile

from docx import Document as DocxDocument
from docx.opc.exceptions import PackageNotFoundError
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from psycopg.errors import UniqueViolation
from pydantic import BaseModel, ConfigDict, Field
from pypdf import PdfReader
from pypdf.errors import PdfReadError

from app import answering, auth, embeddings, graph, store
from app.auth import UserContext
from app.chunking import legal_chunks
from app.config import get_settings


@asynccontextmanager
async def lifespan(_: FastAPI):
    settings = get_settings()
    if not settings.api_key and not settings.jwt_secret:
        raise RuntimeError("Set JWT_SECRET (recommended) or API_KEY before starting the application.")
    if not settings.bootstrap_admin_password and len(settings.api_key) < 12:
        raise RuntimeError("Set BOOTSTRAP_ADMIN_PASSWORD to a password with at least 12 characters.")
    store.initialize_schema()
    try:
        yield
    finally:
        store.close_pool()


app = FastAPI(title="Ask My Docs", version="1.0.0", lifespan=lifespan)
rag_graph = graph.build_graph()


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    session_id: uuid.UUID | None = None
    jurisdiction: str | None = Field(default=None, max_length=200)


class LoginRequest(BaseModel):
    email: str = Field(min_length=3, max_length=320)
    password: str = Field(min_length=1, max_length=256)


class CreateUserRequest(BaseModel):
    email: str = Field(pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$", max_length=320)
    password: str = Field(min_length=12, max_length=256)
    role: str = Field(pattern=r"^(admin|editor|viewer)$")


class CreateWorkspaceRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    name: str = Field(min_length=2, max_length=200)
    admin_email: str = Field(pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$", max_length=320)
    admin_password: str = Field(min_length=12, max_length=256)


@app.post("/api/auth/login")
def login(request: LoginRequest) -> dict:
    user = store.get_user_by_email(request.email)
    if not user or not user["active"] or not auth.verify_password(request.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Email or password is incorrect.")
    context = UserContext(
        id=user["id"],
        workspace_id=user["workspace_id"],
        email=user["email"],
        role=user["role"],
    )
    return {
        "access_token": auth.issue_token(context),
        "token_type": "bearer",
        "expires_in": auth.TOKEN_LIFETIME_SECONDS,
        "user": {
            "id": context.id,
            "workspace_id": context.workspace_id,
            "email": context.email,
            "role": context.role,
        },
    }


@app.post("/api/workspaces", status_code=201)
def create_workspace(
    request: CreateWorkspaceRequest,
    user: Annotated[UserContext, Depends(auth.require_roles("platform_admin"))],
) -> dict:
    try:
        return store.create_workspace(
            request.name,
            request.admin_email,
            auth.hash_password(request.admin_password),
        )
    except UniqueViolation as exc:
        raise HTTPException(status_code=409, detail="A user with that email already exists.") from exc


@app.post("/api/users", status_code=201)
def create_workspace_user(
    request: CreateUserRequest,
    user: Annotated[UserContext, Depends(auth.require_roles("admin", "platform_admin"))],
) -> dict:
    try:
        return store.create_user(
            uuid.UUID(user.workspace_id),
            request.email,
            auth.hash_password(request.password),
            request.role,
        )
    except UniqueViolation as exc:
        raise HTTPException(status_code=409, detail="A user with that email already exists.") from exc


def extract_text(filename: str, data: bytes) -> str:
    suffix = Path(filename).suffix.casefold()
    if suffix == ".pdf":
        try:
            reader = PdfReader(io.BytesIO(data))
        except PdfReadError as exc:
            raise HTTPException(status_code=422, detail="The uploaded PDF could not be read.") from exc
        return "\n\n".join(page.extract_text() or "" for page in reader.pages)
    if suffix == ".docx":
        try:
            document = DocxDocument(io.BytesIO(data))
        except (PackageNotFoundError, BadZipFile) as exc:
            raise HTTPException(status_code=422, detail="The uploaded DOCX file could not be read.") from exc
        return "\n\n".join(paragraph.text for paragraph in document.paragraphs)
    if suffix in {".txt", ".md"}:
        try:
            return data.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise HTTPException(status_code=422, detail="Text documents must use UTF-8 encoding.") from exc
    raise HTTPException(status_code=415, detail="Supported document types are PDF, DOCX, TXT, and Markdown.")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(Path(__file__).parent / "static" / "index.html")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/documents")
def documents(user: Annotated[UserContext, Depends(auth.current_user)]) -> list[dict]:
    return store.list_documents(uuid.UUID(user.workspace_id))


@app.post("/api/documents")
async def upload_document(
    file: Annotated[UploadFile, File()],
    title: Annotated[str, Form(min_length=1, max_length=500)],
    source_url: Annotated[str, Form(min_length=1, max_length=2000)],
    user: Annotated[UserContext, Depends(auth.require_roles("admin", "editor", "platform_admin"))],
    jurisdiction: Annotated[str, Form()] = "",
    court: Annotated[str, Form()] = "",
    case_number: Annotated[str, Form()] = "",
    document_date: Annotated[date | None, Form()] = None,
) -> dict:
    data = await file.read(get_settings().max_upload_bytes + 1)
    if len(data) > get_settings().max_upload_bytes:
        limit_mb = get_settings().max_upload_bytes // (1024 * 1024)
        raise HTTPException(status_code=413, detail=f"The document exceeds the {limit_mb} MB upload limit.")
    if not data:
        raise HTTPException(status_code=400, detail="The uploaded document is empty.")
    text = extract_text(file.filename or "", data)
    chunks = legal_chunks(text)
    if not chunks:
        raise HTTPException(status_code=422, detail="No extractable text was found in the document.")
    vectors = embeddings.embed_texts([
        f"{section_title}\n{content}" if section_title else content
        for section_title, content in chunks
    ])
    result = store.ingest_document(
        workspace_id=uuid.UUID(user.workspace_id),
        source_url=source_url,
        title=title,
        jurisdiction=jurisdiction,
        court=court,
        case_number=case_number,
        document_date=document_date,
        document_hash=hashlib.sha256(data).hexdigest(),
        chunks=chunks,
        embeddings=vectors,
    )
    return {**result, "title": title, "chunks_indexed": len(chunks)}


@app.post("/api/ask")
def ask(request: AskRequest, user: Annotated[UserContext, Depends(auth.current_user)]) -> dict:
    session_id = request.session_id or uuid.uuid4()
    try:
        history = store.get_history(
            session_id,
            uuid.UUID(user.workspace_id),
            uuid.UUID(user.id),
        )
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    try:
        result = rag_graph.invoke(
            {
                "question": request.question,
                "history": history,
                "workspace_id": user.workspace_id,
                "jurisdiction": request.jurisdiction,
            }
        )["result"]
    except answering.SynthesisError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    try:
        store.save_turn(
            session_id,
            uuid.UUID(user.workspace_id),
            uuid.UUID(user.id),
            request.question,
            result["answer"],
            result["citations"],
        )
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    return {"session_id": str(session_id), **result}
