"""Conformance harness for the @podwarden/chat-ui contract.

`serve.py` boots the real app against a faked model upstream so the
package's own conformance kit (vitest) can drive it over HTTP;
`test_contract_routes.py` is the cheap static half — every route the
shipped OpenAPI slice names must exist on this backend.
"""
