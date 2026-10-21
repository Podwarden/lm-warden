def flush_ledger(client) -> None:
    """Write the proxy's in-memory ledger now (it otherwise flushes once a second)."""
    client.portal.call(client.app.state.ledger.flush)


async def pending_settled(ledger) -> bool:
    """ledger.pending() read under the ledger's own lock, so any in-flight flush
    (which swapped the batch for an empty one and is about to restore it) has
    settled first; a bare read can race that window."""
    async with ledger._lock:
        return ledger.pending()


def ledger_pending_settled(client) -> bool:
    """The proxy's ledger batch settled into memory (see pending_settled)."""
    return client.portal.call(pending_settled, client.app.state.ledger)
