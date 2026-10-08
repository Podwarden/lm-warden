MAX_IMAGE_BYTES = 10 * 1024**2
MAX_IMAGE_PIXELS = 40_000_000  # decompression-bomb guard, checked before Image.load()
# Multipart form encoding adds boundary markers, field headers etc. on top of
# the raw file bytes; this is slack for that overhead, not slack for a bigger
# image. A declared Content-Length past MAX_IMAGE_BYTES + this is rejected
# outright, both by Chat2BodyLimitMiddleware (app/chat2/body_limit.py) and,
# as defence-in-depth, by routes_attachments._reject_oversized_content_length.
MULTIPART_OVERHEAD_BYTES = 4096
MAX_IMAGES_PER_MESSAGE = 8
ALLOWED_IMAGE_MIMES: dict[str, str] = {
    "image/png": "png",
    "image/jpeg": "jpg",
    "image/webp": "webp",
}
RESERVE_CAP_TOKENS = 4096
DRAFT_TTL_DAYS = 7
ATTACHMENT_TTL_DAYS = 90
GC_INTERVAL_S = 6 * 3600
SIGNED_URL_TTL_S = 600
TITLE_MAX_CHARS = 60
