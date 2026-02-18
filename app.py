# app.py — Chatbot interne (RAG) sur base Cloudbox/Nextcloud via WebDAV
# -------------------------------------------------------------------
# Objectif: un assistant interne fiable, qui répond UNIQUEMENT avec sources.
# - Sync depuis WebDAV (Nextcloud) -> cache local
# - Indexation TF-IDF (local, sans API externe)
# - Chat + citations (fichier + extrait)
#
# Déps:
#   pip install streamlit requests pypdf python-docx scikit-learn lxml
#
# Lancement:
#   streamlit run app.py

import os
import re
import io
import json
import time
import pickle
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import requests
import streamlit as st
from pypdf import PdfReader
from docx import Document as DocxDocument

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# =========================
# CONFIG
# =========================
APP_TITLE = "Chatbot interne — Base Cloudbox (RAG)"
DEFAULT_WEBDAV_BASE = "https://cloudbox.institutimagine.org/remote.php/dav/files/ambroise.leleve"
CACHE_DIR = Path.home() / ".cloudbox_kb_cache"
FILES_DIR = CACHE_DIR / "files"
INDEX_PATH = CACHE_DIR / "index.pkl"

ALLOWED_EXT = {".pdf", ".docx", ".txt", ".md"}  # volontairement strict au début
IGNORE_DIR_NAMES = {"99_ARCHIVES", "ARCHIVES", ".trash", ".Trash", ".snapshot", ".snapshots"}

# Chunking (simple, robuste)
CHUNK_TARGET_CHARS = 1400
CHUNK_OVERLAP_CHARS = 180
MIN_CHUNK_CHARS = 350

# Retrieval / Answer
TOP_K = 6              # docs/chunks récupérés
MIN_SCORE = 0.18       # seuil de pertinence (TF-IDF) — ajustable

# Safety: réponses sourcées seulement
REQUIRE_SOURCES = True

# =========================
# UTILS
# =========================
def norm(s: str) -> str:
    if s is None:
        return ""
    s = str(s).replace("\u00A0", " ").replace("\t", " ").replace("\r", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def sha1_text(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()

def safe_rel_path(p: str) -> str:
    # Normalize a webdav href/path into a relative path
    p = p.replace("\\", "/")
    p = re.sub(r"^https?://[^/]+", "", p)  # strip host if any
    return p.lstrip("/")

def ensure_dirs():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    FILES_DIR.mkdir(parents=True, exist_ok=True)

def looks_like_binary_or_empty(text: str) -> bool:
    if not text:
        return True
    # too many replacement chars / no letters
    letters = sum(ch.isalpha() for ch in text)
    return letters < 20

# =========================
# FILE TEXT EXTRACTORS
# =========================
def extract_pdf_text(path: Path) -> str:
    try:
        with path.open("rb") as f:
            reader = PdfReader(f)
            parts = []
            for p in reader.pages:
                parts.append(p.extract_text() or "")
            return "\n".join(parts)
    except Exception:
        return ""

def extract_docx_text(path: Path) -> str:
    try:
        doc = DocxDocument(str(path))
        parts = []
        for para in doc.paragraphs:
            t = para.text.strip()
            if t:
                parts.append(t)
        return "\n".join(parts)
    except Exception:
        return ""

def extract_text(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".pdf":
        return extract_pdf_text(path)
    if ext == ".docx":
        return extract_docx_text(path)
    if ext in (".txt", ".md"):
        try:
            return path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return ""
    return ""

# =========================
# CHUNKING
# =========================
def chunk_text(text: str, target_chars: int = CHUNK_TARGET_CHARS, overlap_chars: int = CHUNK_OVERLAP_CHARS) -> List[str]:
    text = text.replace("\u00A0", " ")
    # Split on paragraphs / bullets first
    blocks = [b.strip() for b in re.split(r"\n{2,}|\r\n{2,}", text) if b.strip()]
    if not blocks:
        blocks = [text.strip()]

    chunks = []
    cur = ""
    for b in blocks:
        b = re.sub(r"[ \t]+", " ", b).strip()
        if not b:
            continue
        if len(cur) + len(b) + 1 <= target_chars:
            cur = (cur + "\n" + b).strip()
        else:
            if len(cur) >= MIN_CHUNK_CHARS:
                chunks.append(cur)
            cur = b

    if cur and len(cur) >= MIN_CHUNK_CHARS:
        chunks.append(cur)

    # overlap by chars (simple)
    if overlap_chars > 0 and len(chunks) > 1:
        out = []
        for i, c in enumerate(chunks):
            if i == 0:
                out.append(c)
            else:
                prev = out[-1]
                ov = prev[-overlap_chars:] if len(prev) > overlap_chars else prev
                out.append((ov + "\n" + c).strip())
        chunks = out

    return chunks

# =========================
# WEBDAV (Nextcloud) sync
# =========================
@dataclass
class RemoteEntry:
    href: str
    is_dir: bool
    etag: str
    size: int

def webdav_propfind_list(session: requests.Session, base: str, remote_path: str, depth: int = 1, timeout: int = 30) -> List[RemoteEntry]:
    """
    List children in a WebDAV folder using PROPFIND.
    remote_path is relative to base, no leading slash required.
    """
    # Build URL
    base = base.rstrip("/")
    remote_path = remote_path.strip("/")
    url = f"{base}/{remote_path}" if remote_path else base + "/"

    headers = {
        "Depth": str(depth),
        "Content-Type": "application/xml; charset=utf-8",
    }
    body = """<?xml version="1.0" encoding="utf-8" ?>
<d:propfind xmlns:d="DAV:">
  <d:prop>
    <d:resourcetype />
    <d:getetag />
    <d:getcontentlength />
  </d:prop>
</d:propfind>"""

    r = session.request("PROPFIND", url, headers=headers, data=body.encode("utf-8"), timeout=timeout)
    r.raise_for_status()

    # Parse XML with minimal regex fallback (avoid heavy dependency)
    # But lxml is installed via deps recommendation; still keep robust.
    try:
        from lxml import etree
        root = etree.fromstring(r.content)
        ns = {"d": "DAV:"}

        entries = []
        for resp in root.findall("d:response", namespaces=ns):
            href_el = resp.find("d:href", namespaces=ns)
            href = href_el.text if href_el is not None else ""
            propstat = resp.find("d:propstat", namespaces=ns)
            if propstat is None:
                continue
            prop = propstat.find("d:prop", namespaces=ns)
            if prop is None:
                continue

            rtype = prop.find("d:resourcetype", namespaces=ns)
            is_dir = False
            if rtype is not None and rtype.find("d:collection", namespaces=ns) is not None:
                is_dir = True

            etag_el = prop.find("d:getetag", namespaces=ns)
            etag = (etag_el.text or "").strip('"') if etag_el is not None else ""

            size_el = prop.find("d:getcontentlength", namespaces=ns)
            try:
                size = int(size_el.text) if size_el is not None and size_el.text else 0
            except Exception:
                size = 0

            entries.append(RemoteEntry(href=href, is_dir=is_dir, etag=etag, size=size))

        return entries
    except Exception:
        # Fallback: if XML parsing fails, return nothing
        return []

def should_ignore_remote(path_rel: str) -> bool:
    parts = [p for p in path_rel.split("/") if p]
    for p in parts:
        if p in IGNORE_DIR_NAMES:
            return True
    return False

def webdav_recursive_list_files(session: requests.Session, base: str, root_remote_folder: str, max_files: int = 3000) -> List[RemoteEntry]:
    """
    BFS listing files under root_remote_folder (relative).
    """
    base = base.rstrip("/")
    root_remote_folder = root_remote_folder.strip("/")
    queue = [root_remote_folder]
    files: List[RemoteEntry] = []
    seen_dirs = set()

    while queue:
        cur = queue.pop(0)
        if cur in seen_dirs:
            continue
        seen_dirs.add(cur)

        entries = webdav_propfind_list(session, base, cur, depth=1)
        # First entry is usually the folder itself; we keep children only
        for e in entries:
            href_rel = safe_rel_path(e.href)

            # Nextcloud returns href like /remote.php/dav/files/user/path...
            # We map to path relative to base root (after base's path)
            # Derive base path part:
            base_rel = safe_rel_path(base)
            # If base includes scheme/host, base_rel will start with remote.php/...
            # Remove base_rel prefix if present
            if href_rel.startswith(base_rel):
                rel = href_rel[len(base_rel):].lstrip("/")
            else:
                # fallback: try to find files/<user>/ segment
                m = re.search(r"remote\.php/dav/files/[^/]+/(.*)$", href_rel)
                rel = m.group(1) if m else href_rel

            # skip the directory itself
            if rel.rstrip("/") == cur.rstrip("/"):
                continue

            if should_ignore_remote(rel):
                continue

            if e.is_dir:
                queue.append(rel.rstrip("/"))
            else:
                p = Path(rel)
                if p.suffix.lower() in ALLOWED_EXT:
                    files.append(RemoteEntry(href=rel, is_dir=False, etag=e.etag, size=e.size))
                    if len(files) >= max_files:
                        return files

    return files

def webdav_download_file(session: requests.Session, base: str, rel_path: str, dest: Path, timeout: int = 60):
    base = base.rstrip("/")
    rel_path = rel_path.strip("/")
    url = f"{base}/{rel_path}" if rel_path else base + "/"
    r = session.get(url, stream=True, timeout=timeout)
    r.raise_for_status()
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("wb") as f:
        for chunk in r.iter_content(chunk_size=1024 * 256):
            if chunk:
                f.write(chunk)

def sync_from_webdav(
    base_url: str,
    username: str,
    password: str,
    remote_folder: str,
    progress_cb=None
) -> Dict[str, Dict]:
    """
    Sync remote files -> local cache.
    Returns manifest dict: {rel_path: {etag, size, local_path}}
    """
    ensure_dirs()
    manifest_path = CACHE_DIR / "manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            manifest = {}
    else:
        manifest = {}

    sess = requests.Session()
    sess.auth = (username, password)

    remote_files = webdav_recursive_list_files(sess, base_url, remote_folder)

    total = max(len(remote_files), 1)
    updated = 0
    for i, rf in enumerate(remote_files, start=1):
        rel = rf.href
        etag = rf.etag or ""
        size = rf.size or 0

        local_path = FILES_DIR / rel
        key = rel

        needs = True
        if key in manifest:
            old = manifest[key]
            if old.get("etag") == etag and old.get("size") == size and Path(old.get("local_path", "")).exists():
                needs = False

        if needs:
            try:
                webdav_download_file(sess, base_url, rel, local_path)
                manifest[key] = {"etag": etag, "size": size, "local_path": str(local_path)}
                updated += 1
            except Exception as e:
                # Keep going
                manifest.setdefault(key, {"etag": etag, "size": size, "local_path": str(local_path)})
        if progress_cb:
            progress_cb(i / total, f"Sync {i}/{total} — {Path(rel).name}")

    # remove entries that no longer exist
    remote_set = {rf.href for rf in remote_files}
    for k in list(manifest.keys()):
        if k not in remote_set:
            # keep file but mark stale; or delete
            manifest.pop(k, None)

    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest

# =========================
# INDEX
# =========================
@dataclass
class Chunk:
    doc_path: str
    chunk_id: int
    text: str

@dataclass
class KBIndex:
    vectorizer: TfidfVectorizer
    matrix
    chunks: List[Chunk]
    created_at: float
    manifest_hash: str

def build_manifest_hash(manifest: Dict[str, Dict]) -> str:
    # stable hash based on paths + etags + sizes
    items = []
    for k in sorted(manifest.keys()):
        v = manifest[k]
        items.append(f"{k}|{v.get('etag','')}|{v.get('size',0)}")
    return sha1_text("\n".join(items))

def index_documents(manifest: Dict[str, Dict], progress_cb=None) -> KBIndex:
    chunks: List[Chunk] = []
    files = list(manifest.items())
    total = max(len(files), 1)

    for i, (rel, meta) in enumerate(files, start=1):
        lp = Path(meta.get("local_path", ""))
        if not lp.exists():
            continue
        try:
            text = extract_text(lp)
            if looks_like_binary_or_empty(text):
                continue
            # light cleanup
            text = re.sub(r"[ \t]+", " ", text).strip()
            cks = chunk_text(text)
            for j, ck in enumerate(cks):
                chunks.append(Chunk(doc_path=rel, chunk_id=j, text=ck))
        except Exception:
            continue
        if progress_cb:
            progress_cb(i / total, f"Index {i}/{total} — {lp.name}")

    if not chunks:
        # empty index
        vectorizer = TfidfVectorizer(stop_words=None, max_features=50000, ngram_range=(1, 2))
        matrix = vectorizer.fit_transform([""])
        return KBIndex(vectorizer=vectorizer, matrix=matrix, chunks=[], created_at=time.time(), manifest_hash=build_manifest_hash(manifest))

    corpus = [c.text for c in chunks]
    vectorizer = TfidfVectorizer(stop_words=None, max_features=70000, ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(corpus)

    return KBIndex(
        vectorizer=vectorizer,
        matrix=matrix,
        chunks=chunks,
        created_at=time.time(),
        manifest_hash=build_manifest_hash(manifest),
    )

def save_index(idx: KBIndex):
    ensure_dirs()
    with INDEX_PATH.open("wb") as f:
        pickle.dump(idx, f)

def load_index() -> Optional[KBIndex]:
    if INDEX_PATH.exists():
        try:
            with INDEX_PATH.open("rb") as f:
                return pickle.load(f)
        except Exception:
            return None
    return None

# =========================
# RETRIEVAL + ANSWER
# =========================
def retrieve(idx: KBIndex, query: str, top_k: int = TOP_K) -> List[Tuple[Chunk, float]]:
    if not idx.chunks:
        return []
    q = norm(query)
    if not q:
        return []
    qv = idx.vectorizer.transform([q])
    sims = cosine_similarity(qv, idx.matrix).ravel()
    # top indices
    top_idx = sims.argsort()[::-1][:top_k]
    out = []
    for i in top_idx:
        score = float(sims[i])
        if score >= MIN_SCORE:
            out.append((idx.chunks[i], score))
    return out

def build_answer(query: str, hits: List[Tuple[Chunk, float]]) -> str:
    # Simple “extractive + structured” answer (no LLM), but useful + safe.
    # If you later want LLM, you can swap this block.
    if not hits:
        return "Je n’ai pas trouvé d’élément suffisamment sourcé dans la base. Essaie une formulation plus précise, ou ajoute le document manquant dans la KB."

    # Merge snippets
    bullets = []
    sources = []
    for ch, sc in hits:
        snippet = ch.text
        snippet = re.sub(r"\s+", " ", snippet).strip()
        snippet = snippet[:420] + ("…" if len(snippet) > 420 else "")
        bullets.append(f"- {snippet}")
        sources.append(f"{ch.doc_path}#chunk{ch.chunk_id}")

    # Provide a “best effort” summary without inventing:
    # We present “points pertinents” as extracted bullets.
    answer = []
    answer.append("Voici ce que la base contient de plus pertinent (extraits) :")
    answer.extend(bullets)
    answer.append("")
    answer.append("Sources :")
    for s in sources:
        answer.append(f"- {s}")
    return "\n".join(answer)

# =========================
# STREAMLIT UI
# =========================
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)
st.caption("Connexion WebDAV → sync → index → réponses sourcées (fichier + extrait).")

with st.expander("⚙️ Connexion Cloudbox (WebDAV)", expanded=True):
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        webdav_base = st.text_input("URL WebDAV (base)", value=DEFAULT_WEBDAV_BASE)
        remote_folder = st.text_input("Dossier à indexer (dans ton WebDAV)", value="KB_CentreSeminaire_OFFICIEL")
        st.caption("Astuce: crée un dossier KB_CentreSeminaire_OFFICIEL dans Cloudbox et mets-y tes docs 'OFFICIEL'.")
    with col2:
        username = st.text_input("Identifiant", value="ambroise.leleve")
    with col3:
        password = st.text_input("Mot de passe (idéal: mot de passe d’application)", type="password")

    colA, colB, colC = st.columns([1, 1, 2])
    do_sync = colA.button("📥 Mettre à jour depuis Cloudbox", use_container_width=True)
    do_reindex = colB.button("🧠 Réindexer", use_container_width=True)
    show_cache = colC.checkbox("Afficher infos cache/index", value=False)

ensure_dirs()

# Load manifest + index
manifest_path = CACHE_DIR / "manifest.json"
manifest = {}
if manifest_path.exists():
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        manifest = {}

idx = load_index()

def progress_ui(p, label):
    st.session_state["_p"] = p
    st.session_state["_label"] = label

if do_sync:
    if not webdav_base or not username or not password or not remote_folder:
        st.error("Renseigne URL WebDAV, identifiant, mot de passe et dossier.")
    else:
        prog = st.progress(0, text="Sync…")
        def cb(frac, label):
            prog.progress(min(max(frac, 0.0), 1.0), text=label)
        try:
            manifest = sync_from_webdav(webdav_base, username, password, remote_folder, progress_cb=cb)
            st.success(f"Sync terminée. Fichiers suivis: {len(manifest)}")
        except Exception as e:
            st.error(f"Erreur sync WebDAV: {e}")

        # auto reindex after sync to keep it simple
        prog = st.progress(0, text="Index…")
        def cb2(frac, label):
            prog.progress(min(max(frac, 0.0), 1.0), text=label)
        try:
            idx = index_documents(manifest, progress_cb=cb2)
            save_index(idx)
            st.success("Index mis à jour.")
        except Exception as e:
            st.error(f"Erreur indexation: {e}")

if do_reindex:
    if not manifest:
        st.warning("Pas de manifest. Fais d’abord 'Mettre à jour depuis Cloudbox'.")
    else:
        prog = st.progress(0, text="Index…")
        def cb2(frac, label):
            prog.progress(min(max(frac, 0.0), 1.0), text=label)
        try:
            idx = index_documents(manifest, progress_cb=cb2)
            save_index(idx)
            st.success("Index mis à jour.")
        except Exception as e:
            st.error(f"Erreur indexation: {e}")

if show_cache:
    st.write("Cache:", str(CACHE_DIR))
    st.write("Fichiers:", str(FILES_DIR))
    st.write("Manifest entries:", len(manifest))
    if idx:
        st.write("Index chunks:", len(idx.chunks))
        st.write("Index created:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(idx.created_at)))
        st.write("Index manifest_hash:", idx.manifest_hash)

st.divider()

# Chat UI
st.subheader("💬 Chat")
if not idx or not idx.chunks:
    st.info("Aucun index disponible. Clique sur 'Mettre à jour depuis Cloudbox' puis réessaie.")
    st.stop()

if "chat" not in st.session_state:
    st.session_state.chat = []

for role, content in st.session_state.chat:
    with st.chat_message(role):
        st.markdown(content)

q = st.chat_input("Pose une question (ex: 'livraison traiteur la veille', 'capacités salle du conseil', 'annulation devis')")

if q:
    st.session_state.chat.append(("user", q))
    with st.chat_message("user"):
        st.markdown(q)

    hits = retrieve(idx, q, top_k=TOP_K)

    if REQUIRE_SOURCES and not hits:
        a = "Je ne peux pas répondre de façon fiable: je n’ai pas trouvé de source pertinente dans la base. Ajoute/actualise le document (ou reformule)."
    else:
        a = build_answer(q, hits)

    st.session_state.chat.append(("assistant", a))
    with st.chat_message("assistant"):
        st.markdown(a)

st.caption("Note: ce bot répond à partir des documents indexés. Pour améliorer les réponses: ajoute des docs 'OFFICIEL' bien titrés et clique 'Mettre à jour'.")
