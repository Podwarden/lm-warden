from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from app.chat2.limits import ALLOWED_IMAGE_MIMES, MAX_IMAGE_BYTES, MAX_IMAGE_PIXELS

_FORMAT_BY_MIME = {"image/png": "PNG", "image/jpeg": "JPEG", "image/webp": "WEBP"}
_MIME_BY_FORMAT = {v: k for k, v in _FORMAT_BY_MIME.items()}

# Pillow raises DecompressionBombError from within Image.open()/load() once
# the reported pixel count crosses its own (much higher) internal threshold;
# caught alongside our own MAX_IMAGE_PIXELS pre-check so an oversized/hostile
# image always maps to a clean 415, never an uncaught 500. ValueError covers
# other bare Pillow failures during convert()/paste()/save() -- e.g. a mode
# or parameter combination Pillow itself rejects -- so those also become a
# clean UnsupportedImage/415 rather than an uncaught 500. This tuple is only
# ever active around real Pillow calls, never around the ImageTooLarge
# raises below (those sit outside every try block), so it can't accidentally
# swallow our own ImageTooLarge/UnsupportedImage (both ValueError subclasses).
_PIL_DECODE_ERRORS = (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError)


class ImageTooLarge(ValueError):
    """Raised when the upload (or its re-encoded form) exceeds MAX_IMAGE_BYTES."""


class UnsupportedImage(ValueError):
    """Raised when the upload isn't a supported, decodable image."""


@dataclass(frozen=True)
class StoredImage:
    sha256: str
    mime: str
    ext: str
    size_bytes: int
    width: int
    height: int


def reencode_image(raw: bytes) -> tuple[StoredImage, bytes]:
    """Sniff, validate and re-encode an upload; drops EXIF/ancillary chunks and polyglots."""
    if len(raw) > MAX_IMAGE_BYTES:
        raise ImageTooLarge(f"raw image larger than {MAX_IMAGE_BYTES} bytes")
    try:
        img = Image.open(io.BytesIO(raw))
    except _PIL_DECODE_ERRORS as exc:
        raise UnsupportedImage("not a supported image") from exc
    # Decompression-bomb guard: reject on declared dimensions BEFORE decoding
    # pixel data (img.load()), so a hostile small file claiming a huge
    # resolution never reaches the expensive/memory-heavy decode step.
    if img.size[0] * img.size[1] > MAX_IMAGE_PIXELS:
        raise UnsupportedImage("image exceeds max pixel count")
    try:
        img.load()
    except _PIL_DECODE_ERRORS as exc:
        raise UnsupportedImage("not a supported image") from exc
    fmt = (img.format or "").upper()
    mime = _MIME_BY_FORMAT.get(fmt)
    if mime is None or mime not in ALLOWED_IMAGE_MIMES:
        raise UnsupportedImage(f"unsupported image format {fmt or '?'}")
    # JPEG has no alpha channel (and can't encode CMYK/P/etc. source modes
    # directly) so JPEG output is always flattened to RGB; PNG/WEBP keep RGBA
    # so a transparent source round-trips without losing its alpha channel.
    target_mode = "RGB" if fmt == "JPEG" else "RGBA"
    try:
        clean = Image.new(target_mode, img.size)
        clean.paste(img.convert(target_mode))
        out = io.BytesIO()
        clean.save(out, format=fmt, **({"quality": 90} if fmt == "JPEG" else {}))
    except _PIL_DECODE_ERRORS as exc:
        raise UnsupportedImage(f"failed to re-encode image: {exc}") from exc
    data = out.getvalue()
    if len(data) > MAX_IMAGE_BYTES:
        raise ImageTooLarge(f"re-encoded image larger than {MAX_IMAGE_BYTES} bytes")
    stored = StoredImage(
        sha256=hashlib.sha256(data).hexdigest(),
        mime=mime,
        ext=ALLOWED_IMAGE_MIMES[mime],
        size_bytes=len(data),
        width=clean.width,
        height=clean.height,
    )
    return stored, data


def file_path(data_dir: Path, user_id: int, sha256: str, ext: str) -> Path:
    if not (len(sha256) == 64 and all(c in "0123456789abcdef" for c in sha256)):
        raise ValueError("bad sha256")
    return data_dir / "chat2" / str(int(user_id)) / f"{sha256}.{ext}"


def write_file(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)


def unlink_quiet(path: Path) -> bool:
    try:
        path.unlink()
        return True
    except FileNotFoundError:
        return False
