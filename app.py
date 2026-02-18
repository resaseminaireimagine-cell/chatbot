# app.py — Chatbot interne (RAG) sur Cloudbox/Nextcloud via WebDAV (robuste)
# ------------------------------------------------------------------------
# - Sync WebDAV -> cache local
# - Index TF-IDF local (pas d'API externe)
# - Chat: réponses UNIQUEMENT sourcées (citations fichier + extrait)
#
# Déps:
#   pip install streamlit requests pypdf python-docx scikit-learn lxml
#
# Run:
#   streamlit run app.py

import json
import time
import pickle
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

import streamlit as st
from pypdf import PdfReader
from docx import Document as DocxDocument
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from urllib.parse import unquote, quote


# =========================
# CONFIG
# =========================
APP_TITLE = "Chatbot interne — Base Cloudbox (Nextcloud) via WebDAV"
DEFAULT_WEBDAV_BASE = "https://cloudbox.institutimagine.org/remote.php/dav/files/ambroise.leleve"

CACHE_DIR = Path.home() / ".cloudbox_kb_cache"
FILES_DIR = CACHE_DIR / "files"
INDEX_PATH = CACHE_DIR / "index.pkl"
MANIFEST_PATH = CACHE_DIR / "manifest.json"

ALLOWED_EXT = {".pdf", ".docx", ".txt", ".md"}
IGNORE_DIR_NAMES = {"99_ARCHIVES", "ARCHIVES", ".trash", ".Trash", ".snapshot", ".snapshots"}

CHUNK_TARGET_CHARS = 1400
CHUNK_OVERLAP_CHARS = 180
MIN_CHUNK_CHARS = 350

TOP_K = 6
MIN_SCORE = 0.18

REQUIRE_SOURCES = True


# =========================
# UTILS
# =========================
def norm(s: str) -> str:
    s = "" if s is None else str(s)
    s = s.replace("\u00A0", " ").replace("\t", " ").replace("\r", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def sha1_text(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()

def ensure_dirs():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    FILES_DIR.mkdir(parents=True, exist_ok=True)

def should_ignore_remote(rel_path: str) -> bool:
    rel_path = rel_path.replace("\\", "/").strip("/")
    parts = [p for p in rel_path.split("/") if p]
    return any(p in IGNORE_DIR_NAMES for p in parts)

def looks_like_binary_or_empty(text: str) -> bool:
    if not text:
        return True
    letters = sum(ch.isalpha() for ch in text)
    return letters < 20


# =========================
# REQUESTS SESSION (retries)
# =========================
def make_session(username: str, password: str) -> requests.Session:
    sess = requests.Session()
    sess.auth = (username, password)

    retry = Retry(
        total=5,
        connect=5,
        read=5,
        backoff_factor=0.6,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=frozenset(["GET", "PROPFIND", "HEAD", "OPTIONS"]),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=10, pool_maxsize=10)
    sess.mount("https://", adapter)
    sess.mount("http://", adapter)
    return sess

def check_connectivity(sess: requests.Session, base: str, timeout_s: int = 20) -> Tuple[bool, str]:
    """
    Simple HEAD/GET to see if the host is reachable (network), before PROPFIND.
    """
    try:
        r = sess.request("OPTIONS", base.rstrip("/") + "/", timeout=timeout_s)
        # even 401/403 means reachable; timeout/conn error means not reachable
        return True, f"Reachable (HTTP {r.status_code})"
    except requests.exceptions.ConnectTimeout:
        return False, "ConnectTimeout (réseau / firewall / VPN ?)"
    except requests.exceptions.ConnectionError as e:
        return False, f"ConnectionError ({e})"
    except Exception as e:
        return False, f"Erreur connexion ({e})"


# =========================
# TEXT EXTRACTORS
# =========================
def extract_pdf_text(path: Path) -> str:
    try:
        with path.open("rb") as f:
            reader = PdfReader(f)
            return "\n".join([(p.extract_text() or "") for p in reader.pages])
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
def chunk_text(text: str) -> List[str]:
    text = text.replace("\u00A0", " ")
    blocks = [b.strip() for b in re.split(r"\n{2,}", text) if b.strip()]
    if not blocks:
        blocks = [text.strip()]

    chunks = []
    cur = ""
    for b in blocks:
        b = re.sub(r"[ \t]+", " ", b).strip()
        if not b:
            continue
        if len(cur) + len(b) + 1 <= CHUNK_TARGET_CHARS:
            cur = (cur + "\n" + b).strip()
        else:
            if len(cur) >= MIN_CHUNK_CHARS:
                chunks.append(cur)
            cur = b

    if cur and len(cur) >= MIN_CHUNK_CHARS:
        chunks.append(cur)

    if CHUNK_OVERLAP_CHARS > 0 and len(chunks) > 1:
        out = [chunks[0]]
        for i in range(1, len(chunks)):
            prev = out[-1]
            ov = prev[-CHUNK_OVERLAP_CHARS:] if len(prev) > CHUNK_OVERLAP_CHARS else prev
            out.append((ov + "\n" + chunks[i]).strip())
        chunks = out

    return chunks


# =========================
# WEBDAV (Nextcloud)
# =========================
@dataclass
class RemoteEntry:
    rel_path: str
    is_dir: bool
    etag: str
    size: int

def _dav_url(base: str, rel_path: str) -> str:
    base = base.rstrip("/")
    rel_path = rel_path.strip("/")
    if not rel_path:
        return base + "/"
    parts = [quote(p) for p in rel_path.split("/")]
    return base + "/" + "/".join(parts)

def webdav_propfind(session: requests.Session, url: str, depth: int = 1, timeout_s: int = 60) -> bytes:
    headers = {"Depth": str(depth), "Content-Type": "application/xml; charset=utf-8"}
    body = """<?xml version="1.0" encoding="utf-8" ?>
<d:propfind xmlns:d="DAV:">
  <d:prop>
    <d:resourcetype />
    <d:getetag />
    <d:getcontentlength />
  </d:prop>
</d:propfind>"""
    # timeout can be (connect, read)
    r = session.request("PROPFIND", url, headers=headers, data=body.encode("utf-8"), timeout=(30, timeout_s))
    r.raise_for_status()
    return r.content

def webdav_list_children(session: requests.Session, base: str, folder_rel: str) -> List[RemoteEntry]:
    from lxml import etree

    url = _dav_url(base, folder_rel)
    xml = webdav_propfind(session, url, depth=1)

    root = etree.fromstring(xml)
    ns = {"d": "DAV:"}

    base_path = re.sub(r"^https?://[^/]+", "", base).rstrip("/")
    entries: List[RemoteEntry] = []

    for resp in root.findall("d:response", namespaces=ns):
        href_el = resp.find("d:href", namespaces=ns)
        href_raw = href_el.text if href_el is not None else ""
        href = unquote(href_raw or "")

        if href.startswith(base_path):
            rel = href[len(base_path):].lstrip("/")
        else:
            m = re.search(r"/remote\.php/dav/files/[^/]+/(.*)$", href)
            rel = m.group(1) if m else href.lstrip("/")

        rel_norm = rel.rstrip("/")
        folder_norm = folder_rel.strip("/").rstrip("/")
        if rel_norm == folder_norm:
            continue

        propstat = resp.find("d:propstat", namespaces=ns)
        if propstat is None:
            continue
        prop = propstat.find("d:prop", namespaces=ns)
        if prop is None:
            continue

        rtype = prop.find("d:resourcetype", namespaces=ns)
        is_dir = rtype is not None and rtype.find("d:collection", namespaces=ns) is not None

        etag_el = prop.find("d:getetag", namespaces=ns)
        etag = (etag_el.text or "").strip('"') if etag_el is not None else ""

        size_el = prop.find("d:getcontentlength", namespaces=ns)
        try:
            size = int(size_el.text) if size_el is not None and size_el.text else 0
        except Exception:
            size = 0

        entries.append(RemoteEntry(rel_path=rel, is_dir=is_dir, etag=etag, size=size))

    return entries

def webdav_recursive_list_files(session: requests.Session, base: str, root_folder: str, max_files: int = 3000) -> List[RemoteEntry]:
    root_folder = root_folder.strip("/")
    queue = [root_folder]
    seen = set()
    out: List[RemoteEntry] = []

    while queue:
        folder = queue.pop(0)
        if folder in seen:
            continue
        seen.add(folder)

        children = webdav_list_children(session, base, folder)
        for e in children:
            if should_ignore_remote(e.rel_path):
                continue
            if e.is_dir:
                queue.append(e.rel_path.rstrip("/"))
            else:
                ext = Path(e.rel_path).suffix.lower()
                if ext in ALLOWED_EXT:
                    out.append(e)
                    if len(out) >= max_files:
                        return out
    return out

def webdav_download(session: requests.Session, base: str, rel_path: str, dest: Path, timeout_s: int = 180):
    url = _dav_url(base, rel_path)
    r = session.get(url, stream=True, timeout=(30, timeout_s))
    r.raise_for_status()
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("wb") as f:
        for chunk in r.iter_content(chunk_size=1024 * 256):
            if chunk:
                f.write(chunk)


# =========================
# SYNC + INDEX
# =========================
def load_manifest() -> Dict[str, Dict]:
    if MANIFEST_PATH.exists():
        try:
            return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}

def save_manifest(manifest: Dict[str, Dict]):
    ensure_dirs()
    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

def build_manifest_hash(manifest: Dict[str, Dict]) -> str:
    items = []
    for k in sorted(manifest.keys()):
        v = manifest[k]
        items.append(f"{k}|{v.get('etag','')}|{v.get('size',0)}")
    return sha1_text("\n".join(items))

def sync_from_webdav(base: str, username: str, password: str, remote_folder: str, progress_cb=None) -> Dict[str, Dict]:
    ensure_dirs()
    manifest = load_manifest()

    sess = make_session(username, password)

    files = webdav_recursive_list_files(sess, base, remote_folder)
    total = max(len(files), 1)

    remote_set = set()
    for i, f in enumerate(files, start=1):
        rel = f.rel_path
        remote_set.add(rel)
        local_path = FILES_DIR / rel

        needs = True
        if rel in manifest:
            old = manifest[rel]
            old_local = Path(old.get("local_path", ""))
            if old.get("etag") == f.etag and old.get("size") == f.size and old_local.exists():
                needs = False

        if needs:
            webdav_download(sess, base, rel, local_path)

        manifest[rel] = {"etag": f.etag, "size": f.size, "local_path": str(local_path)}

        if progress_cb:
            progress_cb(i / total, f"Sync {i}/{total} — {Path(rel).name}")

    # purge deleted
    for k in list(manifest.keys()):
        if k not in remote_set:
            manifest.pop(k, None)

    save_manifest(manifest)
    return manifest

@dataclass
class Chunk:
    doc_path: str
    chunk_id: int
    text: str

@dataclass
class KBIndex:
    vectorizer: TfidfVectorizer
    matrix: Any
    chunks: List[Chunk]
    created_at: float
    manifest_hash: str

def index_documents(manifest: Dict[str, Dict], progress_cb=None) -> KBIndex:
    chunks: List[Chunk] = []
    items = list(manifest.items())
    total = max(len(items), 1)

    for i, (rel, meta) in enumerate(items, start=1):
        lp = Path(meta.get("local_path", ""))
        if not lp.exists():
            continue
        text = extract_text(lp)
        if looks_like_binary_or_empty(text):
            continue
        text = re.sub(r"[ \t]+", " ", text).strip()

        ck_list = chunk_text(text)
        for j, ck in enumerate(ck_list):
            chunks.append(Chunk(doc_path=rel, chunk_id=j, text=ck))

        if progress_cb:
            progress_cb(i / total, f"Index {i}/{total} — {lp.name}")

    manifest_hash = build_manifest_hash(manifest)

    # SAFE fallback: never fit on empty string
    if not chunks:
        vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
        matrix = vectorizer.fit_transform(["placeholder"])
        return KBIndex(vectorizer, matrix, [], time.time(), manifest_hash)

    corpus = [c.text for c in chunks if norm(c.text)]
    if not corpus:
        vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
        matrix = vectorizer.fit_transform(["placeholder"])
        return KBIndex(vectorizer, matrix, [], time.time(), manifest_hash)

    vectorizer = TfidfVectorizer(max_features=70000, ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(corpus)
    return KBIndex(vectorizer, matrix, chunks, time.time(), manifest_hash)

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
    if not idx or not idx.chunks:
        return []
    q = norm(query)
    if not q:
        return []
    qv = idx.vectorizer.transform([q])
    sims = cosine_similarity(qv, idx.matrix).ravel()
    top_idx = sims.argsort()[::-1][:top_k]
    out = []
    for i in top_idx:
        score = float(sims[i])
        if score >= MIN_SCORE:
            out.append((idx.chunks[i], score))
    return out

def build_answer(query: str, hits: List[Tuple[Chunk, float]]) -> str:
    if not hits:
        return "Je n’ai pas trouvé de source suffisamment pertinente dans la base. Reformule, ou ajoute le document manquant dans la KB."

    lines = ["Voici les extraits les plus pertinents :"]
    sources = []
    for ch, sc in hits:
        snippet = re.sub(r"\s+", " ", ch.text).strip()
        snippet = snippet[:480] + ("…" if len(snippet) > 480 else "")
        lines.append(f"- {snippet}")
        sources.append(f"{ch.doc_path}#chunk{ch.chunk_id}")
    lines.append("")
    lines.append("Sources :")
    lines.extend([f"- {s}" for s in sources])
    return "\n".join(lines)


# =========================
# STREAMLIT UI
# =========================
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)
st.caption("WebDAV Nextcloud → Sync → Index → Chat sourcé.")

ensure_dirs()
manifest = load_manifest()
idx = load_index()

with st.expander("⚙️ Connexion Cloudbox (WebDAV)", expanded=True):
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        webdav_base = st.text_input("URL WebDAV (base)", value=DEFAULT_WEBDAV_BASE)
        remote_folder = st.text_input("Dossier à indexer", value="KB_CentreSeminaire_OFFICIEL")
        st.caption("Crée ce dossier dans Cloudbox. Mets les archives dans 99_ARCHIVES ou ARCHIVES.")
    with c2:
        username = st.text_input("Identifiant", value="ambroise.leleve")
    with c3:
        password = st.text_input("Mot de passe (idéal: mot de passe d’application)", type="password")

    c4, c5, c6 = st.columns([1, 1, 2])
    btn_sync_reindex = c4.button("📥 Sync + 🧠 Réindex", use_container_width=True)
    btn_reindex = c5.button("🧠 Réindex (sans sync)", use_container_width=True)
    debug = c6.checkbox("Debug", value=False)

if debug:
    st.write("Cache:", str(CACHE_DIR))
    st.write("Manifest:", len(manifest))
    if idx:
        st.write("Chunks:", len(idx.chunks))
        st.write("Index date:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(idx.created_at)))
        st.write("Manifest hash:", idx.manifest_hash)

def prog_cb_factory():
    prog = st.progress(0, text="…")
    def cb(frac, label):
        prog.progress(min(max(frac, 0.0), 1.0), text=label)
    return cb

if btn_sync_reindex:
    if not webdav_base or not username or not password or not remote_folder:
        st.error("Renseigne URL, identifiant, mot de passe et dossier.")
    else:
        sess = make_session(username, password)
        ok, msg = check_connectivity(sess, webdav_base, timeout_s=20)
        if not ok:
            st.error(
                "Impossible de joindre Cloudbox depuis l’endroit où tourne l’app.\n\n"
                f"Détail: **{msg}**\n\n"
                "👉 Si tu es sur Streamlit Cloud / un serveur externe: c’est très probablement un blocage réseau.\n"
                "Solution: exécuter l’app en interne (réseau Imagine/VPN) ou faire ouvrir l’accès WebDAV côté IT."
            )
        else:
            try:
                cb = prog_cb_factory()
                manifest = sync_from_webdav(webdav_base, username, password, remote_folder, progress_cb=cb)
                st.success(f"Sync OK — {len(manifest)} fichiers suivis.")
            except Exception as e:
                st.error(f"Erreur sync WebDAV: {e}")
                manifest = load_manifest()  # keep previous if any

            # Index only if we actually have something
            if manifest:
                try:
                    cb2 = prog_cb_factory()
                    idx = index_documents(manifest, progress_cb=cb2)
                    save_index(idx)
                    st.success("Index mis à jour.")
                except Exception as e:
                    st.error(f"Erreur indexation: {e}")
            else:
                st.warning("Aucun fichier synchronisé. Indexation ignorée.")

if btn_reindex:
    if not manifest:
        st.warning("Pas de manifest. Fais d’abord Sync.")
    else:
        try:
            cb2 = prog_cb_factory()
            idx = index_documents(manifest, progress_cb=cb2)
            save_index(idx)
            st.success("Index mis à jour.")
        except Exception as e:
            st.error(f"Erreur indexation: {e}")

st.divider()
st.subheader("💬 Chat (réponses sourcées)")

if not idx or not idx.chunks:
    st.info("Aucun index (ou aucun contenu exploitable). Clique sur “Sync + Réindex”.")
    st.stop()

if "chat" not in st.session_state:
    st.session_state.chat = []

for role, content in st.session_state.chat:
    with st.chat_message(role):
        st.markdown(content)

q = st.chat_input("Pose une question (ex: 'livraison traiteur', 'capacité salle du Conseil', 'règles technique')")

if q:
    st.session_state.chat.append(("user", q))
    with st.chat_message("user"):
        st.markdown(q)

    hits = retrieve(idx, q, top_k=TOP_K)

    if REQUIRE_SOURCES and not hits:
        a = "Je ne peux pas répondre de façon fiable: aucune source pertinente trouvée. Ajoute/actualise le doc dans la KB ou reformule."
    else:
        a = build_answer(q, hits)

    st.session_state.chat.append(("assistant", a))
    with st.chat_message("assistant"):
        st.markdown(a)

st.caption("Conseil: commence avec 10–30 docs OFFICIELS. Mets le reste en ARCHIVES.")
