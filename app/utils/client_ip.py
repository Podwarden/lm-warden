from starlette.requests import HTTPConnection


def client_ip(conn: HTTPConnection) -> str | None:
    """First X-Forwarded-For hop, else X-Real-Ip, else the socket peer.

    Shared by the proxy's live request registry and request history
    (app/proxy/routes.py) and the admin-token audit trail
    (app/auth/admin_audit.py), so both record the same address. The headers
    are whatever the reverse proxy (or the client) sent, so this can be
    forged; the audit trail keeps the socket peer next to it.
    """
    xff = conn.headers.get("x-forwarded-for")
    if xff:
        return xff.split(",")[0].strip() or None
    xri = conn.headers.get("x-real-ip")
    if xri:
        return xri.strip() or None
    return conn.client.host if conn.client else None
