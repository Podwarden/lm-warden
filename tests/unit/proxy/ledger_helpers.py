def flush_ledger(client) -> None:
    """Write the proxy's in-memory ledger now (it otherwise flushes once a second)."""
    client.portal.call(client.app.state.ledger.flush)
