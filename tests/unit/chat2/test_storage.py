from __future__ import annotations

import io
import struct
import time
import zlib
from typing import Any

import pytest
from PIL import Image

from app.chat2.limits import MAX_IMAGE_BYTES
from app.chat2.signing import attachment_url, sign_attachment, verify_attachment
from app.chat2.storage import ImageTooLarge, UnsupportedImage, reencode_image


def _png(w: int = 4, h: int = 3) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (w, h), (255, 0, 0)).save(buf, format="PNG")
    return buf.getvalue()


def _jpeg_with_exif(w: int = 4, h: int = 3) -> bytes:
    """A JPEG whose bytes genuinely carry an EXIF (APP1) segment.

    JPEG's APP1/EXIF payload is spec-mandated to start with the literal
    ASCII bytes ``Exif\\x00\\x00`` -- unlike PNG's eXIf chunk (raw TIFF, no
    such marker), so this is a reliable "does the fixture actually contain
    EXIF" probe, not just an ``img.info`` dict entry that Pillow may never
    write to the file at all.
    """
    buf = io.BytesIO()
    img = Image.new("RGB", (w, h), (255, 0, 0))
    exif = Image.Exif()
    exif[0x010F] = "FakeMake"  # Make tag
    img.save(buf, format="JPEG", exif=exif.tobytes())
    return buf.getvalue()


def _cmyk_jpeg(w: int = 4, h: int = 3) -> bytes:
    buf = io.BytesIO()
    Image.new("CMYK", (w, h), (0, 0, 0, 0)).save(buf, format="JPEG")
    return buf.getvalue()


def test_reencode_reports_dims_and_dedupes_by_content() -> None:
    stored, data = reencode_image(_png(4, 3))
    assert (stored.mime, stored.ext, stored.width, stored.height) == ("image/png", "png", 4, 3)
    assert stored.size_bytes == len(data)
    assert len(stored.sha256) == 64


def test_reencode_strips_real_exif_from_jpeg() -> None:
    raw = _jpeg_with_exif(4, 3)
    assert b"Exif" in raw, "fixture sanity: input must genuinely carry EXIF"
    stored, data = reencode_image(raw)
    assert stored.mime == "image/jpeg"
    assert b"Exif" not in data


def test_reencode_handles_cmyk_jpeg_without_crashing() -> None:
    stored, data = reencode_image(_cmyk_jpeg(4, 3))
    assert stored.mime == "image/jpeg" and stored.width == 4 and stored.height == 3
    assert len(data) > 0


def test_reencode_rejects_non_image() -> None:
    with pytest.raises(UnsupportedImage):
        reencode_image(b"<svg xmlns='http://www.w3.org/2000/svg'></svg>")


def _png_with_declared_size(w: int, h: int) -> bytes:
    """A syntactically-valid 1x1 PNG whose IHDR chunk is patched to declare
    a different (w, h) -- with its CRC32 recomputed so Pillow's chunk-CRC
    check doesn't reject it outright. This is exactly the shape of a real
    decompression-bomb PNG: a tiny file that *claims* to be huge; the actual
    IDAT payload never has to hold that many pixels because we must reject
    it (based on the declared size alone) before ever decoding IDAT.
    """
    buf = io.BytesIO()
    Image.new("RGB", (1, 1)).save(buf, format="PNG")
    data = bytearray(buf.getvalue())
    ihdr_type_at = data.index(b"IHDR")
    data_start = ihdr_type_at + 4  # start of the 13-byte IHDR payload
    data[data_start : data_start + 4] = w.to_bytes(4, "big")
    data[data_start + 4 : data_start + 8] = h.to_bytes(4, "big")
    chunk_type_and_data = bytes(data[ihdr_type_at : data_start + 13])
    crc_at = data_start + 13
    data[crc_at : crc_at + 4] = struct.pack(">I", zlib.crc32(chunk_type_and_data))
    return bytes(data)


def test_reencode_rejects_declared_decompression_bomb() -> None:
    """Pixel count is checked BEFORE Image.load(): a PNG declaring 9000x9000
    (81M px, over MAX_IMAGE_PIXELS=40M but under Pillow's own much higher
    built-in DecompressionBombError threshold, so Image.open() itself lets
    it through) must still be rejected by our own explicit guard."""
    with pytest.raises(UnsupportedImage):
        reencode_image(_png_with_declared_size(9000, 9000))


def test_reencode_rejects_pillows_own_decompression_bomb_as_unsupported() -> None:
    """Dimensions large enough to trip Pillow's OWN built-in bomb guard
    (fired from inside Image.open(), before our explicit pixel check even
    runs) must still map to UnsupportedImage, never propagate as a raw
    Image.DecompressionBombError / crash the request as a 500."""
    with pytest.raises(UnsupportedImage):
        reencode_image(_png_with_declared_size(20000, 20000))


def test_reencode_rejects_raw_input_over_byte_limit(monkeypatch) -> None:
    """The cheap up-front check: raw upload bytes alone exceed the limit."""
    monkeypatch.setattr("app.chat2.storage.MAX_IMAGE_BYTES", 1)
    with pytest.raises(ImageTooLarge):
        reencode_image(_png(4, 3))


def test_reencode_rejects_when_reencoded_output_exceeds_byte_limit(monkeypatch) -> None:
    """Distinct from the raw-input check above: a small, valid input that
    passes the up-front size check must still be rejected if its RE-ENCODED
    output exceeds MAX_IMAGE_BYTES. Forces the second (post-save) size check
    in reencode_image by making Pillow's own save() emit an oversized
    payload, independent of whatever real compression happens to produce for
    a tiny fixture image."""
    raw = _png(4, 3)
    assert len(raw) <= MAX_IMAGE_BYTES  # sanity: raw input clears the up-front check

    real_save = Image.Image.save

    def _bloated_save(self: Image.Image, fp: Any, format: str | None = None, **kw: Any) -> None:
        real_save(self, fp, format=format, **kw)
        fp.write(b"0" * (MAX_IMAGE_BYTES + 1))

    monkeypatch.setattr(Image.Image, "save", _bloated_save)
    with pytest.raises(ImageTooLarge):
        reencode_image(raw)


def test_reencode_wraps_bare_pillow_value_error_as_unsupported(monkeypatch) -> None:
    """Pillow's convert()/paste()/save() can raise a bare ValueError for some
    malformed/edge-case mode-or-parameter combinations it rejects outright;
    this must map to UnsupportedImage (415), never propagate as an uncaught
    ValueError / crash the request as a 500."""
    raw = _png(4, 3)  # build the fixture bytes with the REAL save, before patching

    def _raise_value_error(
        self: Image.Image, fp: Any, format: str | None = None, **kw: Any
    ) -> None:
        raise ValueError("simulated Pillow save failure")

    monkeypatch.setattr(Image.Image, "save", _raise_value_error)
    with pytest.raises(UnsupportedImage):
        reencode_image(raw)


def test_image_too_large_and_unsupported_are_distinct_value_errors() -> None:
    assert issubclass(ImageTooLarge, ValueError)
    assert issubclass(UnsupportedImage, ValueError)
    assert not issubclass(ImageTooLarge, UnsupportedImage)
    assert not issubclass(UnsupportedImage, ImageTooLarge)


def test_signing_roundtrip_and_tamper() -> None:
    now = int(time.time())
    tok = sign_attachment("s3cret", 7, "att1", now + 60)
    assert verify_attachment("s3cret", 7, "att1", tok, now)
    assert not verify_attachment("s3cret", 8, "att1", tok, now)  # other user
    assert not verify_attachment("s3cret", 7, "att1", tok, now + 61)  # expired
    assert not verify_attachment("s3cret", 7, "att1", tok[:-2] + "zz", now)  # tampered
    assert attachment_url("s3cret", 7, "att1", 600).startswith("/api/chat2/attachments/att1?t=")
