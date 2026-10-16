"""Small Supabase Storage adapter used to persist ForestWatch imagery.

The dashboard continues to analyse ordinary local files.  When cloud storage
is configured, this module mirrors those files to a local runtime cache before
analysis and mirrors newly uploaded images back to Supabase.
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

from .config import settings

_synced = False


class CloudStorageError(RuntimeError):
    """A concise, safe error for the API client."""


def enabled() -> bool:
    return bool(settings.supabase_url and settings.supabase_service_role_key and settings.supabase_bucket)


def _request(path: str, *, method: str = "GET", data: bytes | None = None, content_type: str | None = None):
    if not enabled():
        return None
    headers = {
        "Authorization": f"Bearer {settings.supabase_service_role_key}",
        "apikey": settings.supabase_service_role_key,
    }
    if content_type:
        headers["Content-Type"] = content_type
    request = Request(f"{settings.supabase_url.rstrip('/')}{path}", data=data, headers=headers, method=method)
    try:
        with urlopen(request, timeout=60) as response:
            return response.read()
    except (HTTPError, URLError) as exc:
        raise CloudStorageError("ForestWatch could not reach its cloud image storage.") from exc


def _list(prefix: str = "") -> list[dict]:
    payload = json.dumps({"prefix": prefix, "limit": 1000, "offset": 0}).encode()
    data = _request(
        f"/storage/v1/object/list/{quote(settings.supabase_bucket, safe='')}",
        method="POST", data=payload, content_type="application/json",
    )
    return json.loads(data.decode() if data else "[]")


def _objects(prefix: str = "") -> list[str]:
    results: list[str] = []
    for entry in _list(prefix):
        name = str(entry.get("name") or "")
        if not name:
            continue
        object_name = f"{prefix}/{name}" if prefix else name
        if entry.get("id") or entry.get("metadata"):
            results.append(object_name)
        else:
            results.extend(_objects(object_name))
    return results


def _local_path(object_name: str) -> Path:
    target = (settings.dataset_path / object_name).resolve()
    root = settings.dataset_path.resolve()
    if target != root and root not in target.parents:
        raise CloudStorageError("Invalid cloud image path.")
    return target


def sync_to_local() -> int:
    """Download cloud files that are absent from this server's temporary cache."""
    global _synced
    if not enabled():
        return 0
    if _synced:
        return 0
    missing = []
    for object_name in _objects():
        target = _local_path(object_name)
        if target.exists():
            continue
        missing.append((object_name, target))

    def download(entry: tuple[str, Path]) -> None:
        object_name, target = entry
        body = _request(
            f"/storage/v1/object/{quote(settings.supabase_bucket, safe='')}/{quote(object_name, safe='/')}"
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(body or b"")

    # A free Render instance has limited CPU, but parallel network transfers
    # prevent the dashboard's first cloud sync from waiting once per image.
    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(download, missing))
    _synced = True
    return len(missing)


def upload(local_path: Path, object_name: str, content_type: str | None = None) -> None:
    """Persist a successfully received image in the configured private bucket."""
    if not enabled():
        return
    body = local_path.read_bytes()
    content_type = content_type or "application/octet-stream"
    _request(
        f"/storage/v1/object/{quote(settings.supabase_bucket, safe='')}/{quote(object_name, safe='/')}",
        method="POST", data=body, content_type=content_type,
    )
