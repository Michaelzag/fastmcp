"""Absolute upstream deadlines are honoured at every use site, never re-based.

A token response carries absolute deadlines (`expires_at`, set by authlib on
receipt or sent by the IdP, and an explicit `refresh_token_expires_at`) and
relative lifetimes (`expires_in`, `refresh_expires_in`). Each deadline is the
earlier of the absolute value and receipt + the relative one, snapshotted when
the response arrives. A redemption or refresh that happens later ends the
FastMCP JWT, its JTI mapping and the stored upstream token there, not at
`now + expires_in`. Responses without an absolute access deadline keep the
relative behaviour; a present but invalid value is refused.
"""

import math
import time
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, Mock, patch
from urllib.parse import parse_qs, urlparse

import pytest
from key_value.aio._utils import managed_entry, time_to_live
from key_value.aio.stores.memory import MemoryStore
from mcp.server.auth.provider import (
    AccessToken,
    AuthorizationCode,
    RefreshToken,
    TokenError,
)
from mcp.shared.auth import OAuthClientInformationFull
from pydantic import AnyUrl

from fastmcp.server.auth.auth import TokenVerifier
from fastmcp.server.auth.oauth_proxy import OAuthProxy
from fastmcp.server.auth.oauth_proxy.models import (
    ClientCode,
    JTIMapping,
    OAuthTransaction,
    UpstreamTokenSet,
    _hash_token,
)

REDIRECT = "http://localhost:12345/callback"
DELAY = 120


@pytest.fixture
def verifier():
    """Accepts upstream access tokens named refreshed-* or valid-*."""
    mock = Mock(spec=TokenVerifier)
    mock.required_scopes = ["read"]

    async def verify(token: str) -> AccessToken | None:
        if token.startswith(("refreshed-", "valid-")):
            return AccessToken(token=token, client_id="test-client", scopes=["read"])
        return None

    mock.verify_token = AsyncMock(side_effect=verify)
    return mock


@pytest.fixture
def store() -> MemoryStore:
    return MemoryStore()


def make_proxy(verifier, store, **config) -> OAuthProxy:
    proxy = OAuthProxy(
        upstream_authorization_endpoint="https://idp.example.com/authorize",
        upstream_token_endpoint="https://idp.example.com/token",
        upstream_client_id="test-client",
        upstream_client_secret="test-secret",
        token_verifier=verifier,
        base_url="https://proxy.example.com",
        jwt_signing_key="test-secret-key",
        require_authorization_consent=False,
        client_storage=store,
        **config,
    )
    proxy.set_mcp_path("/mcp")
    return proxy


@pytest.fixture
def proxy(verifier, store) -> OAuthProxy:
    return make_proxy(verifier, store)


CONFIGURED_ACCESS = 3_600
CONFIGURED_REFRESH_FALLBACK = 86_400


@pytest.fixture
def configured(verifier, store) -> OAuthProxy:
    """A proxy with a configured FastMCP access lifetime and refresh fallback."""
    return make_proxy(
        verifier,
        store,
        fastmcp_access_token_expiry_seconds=CONFIGURED_ACCESS,
        fallback_refresh_token_expiry_seconds=CONFIGURED_REFRESH_FALLBACK,
    )


async def register(proxy: OAuthProxy) -> OAuthClientInformationFull:
    client = OAuthClientInformationFull(
        client_id="test-client",
        client_secret="test-secret",
        redirect_uris=[AnyUrl(REDIRECT)],
    )
    await proxy.register_client(client)
    return client


async def exchange(proxy: OAuthProxy, client: OAuthClientInformationFull, code: str):
    return await proxy.exchange_authorization_code(
        client=client,
        authorization_code=AuthorizationCode(
            code=code,
            scopes=["read"],
            expires_at=time.time() + 300,
            client_id="test-client",
            code_challenge="challenge",
            redirect_uri=AnyUrl(REDIRECT),
            redirect_uri_provided_explicitly=True,
        ),
    )


async def redeem(
    proxy: OAuthProxy,
    client: OAuthClientInformationFull,
    idp_tokens: dict,
    *,
    access_token_expires_at: float | None = None,
    refresh_token_expires_at: float | None = None,
    created_at: float | None = None,
):
    """Redeem a code whose callback recorded the given deadlines."""
    await proxy._code_store.put(
        key="code",
        value=ClientCode(
            code="code",
            client_id="test-client",
            redirect_uri=REDIRECT,
            code_challenge="challenge",
            code_challenge_method="S256",
            scopes=["read"],
            idp_tokens=idp_tokens,
            expires_at=time.time() + 300,
            created_at=time.time() if created_at is None else created_at,
            access_token_expires_at=access_token_expires_at,
            refresh_token_expires_at=refresh_token_expires_at,
        ),
    )
    return await exchange(proxy, client, "code")


async def callback(proxy: OAuthProxy, token_response: dict):
    """Run the IdP callback with an upstream that answers `token_response`."""
    await proxy._transaction_store.put(
        key="txn-id",
        value=OAuthTransaction(
            txn_id="txn-id",
            client_id="test-client",
            client_redirect_uri=REDIRECT,
            client_state="client-state",
            code_challenge="",
            code_challenge_method="S256",
            scopes=["read"],
            created_at=time.time(),
        ),
    )
    request = Mock()
    request.query_params = {"code": "idp-code", "state": "txn-id"}
    request.cookies = {}
    upstream = AsyncMock()
    upstream.fetch_token = AsyncMock(return_value=token_response)
    with patch.object(proxy, "_create_upstream_oauth_client", return_value=upstream):
        return await proxy._handle_idp_callback(request)


async def recorded_code(proxy: OAuthProxy, token_response: dict):
    """The callback's issued code and the ClientCode it stored."""
    response = await callback(proxy, token_response)
    assert response.status_code == 302
    code = parse_qs(urlparse(response.headers["location"]).query)["code"][0]
    stored = await proxy._code_store.get(key=code)
    assert stored is not None
    return code, stored


async def entry_ends_at(store: MemoryStore, key: str, collection: str) -> float:
    value, ttl = await store.ttl(key, collection=collection)
    assert value is not None and ttl is not None
    return time.time() + ttl


async def stored_set(
    proxy: OAuthProxy, access_token: str
) -> tuple[str, UpstreamTokenSet]:
    jti = proxy.jwt_issuer.verify_token(access_token)["jti"]
    mapping = await proxy._jti_mapping_store.get(key=jti)
    assert mapping is not None
    token_set = await proxy._upstream_token_store.get(key=mapping.upstream_token_id)
    assert token_set is not None
    return jti, token_set


async def everything(store: MemoryStore) -> dict:
    return {
        collection: {
            key: await store.get(key, collection=collection)
            for key in await store.keys(collection)
        }
        for collection in await store.collections()
    }


NOTHING_PUBLISHED = {
    "mcp-upstream-tokens": [],
    "mcp-jti-mappings": [],
    "mcp-refresh-tokens": [],
}


async def published(store: MemoryStore) -> dict:
    """Every token record a redemption could publish."""
    return {name: await store.keys(name) for name in NOTHING_PUBLISHED}


# --------------------------------------------------------------------------
# Helpers: normalization, validity and floor
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("token_data", "expected"),
    [
        ({"expires_at": 1_000}, 1_000),
        ({"expires_at": 1_000.9}, 1_000.9),
        ({"expires_at": None, "expires_in": 300}, None),
        ({"expires_in": 300}, None),
        # Both present: the earlier wins, in either direction.
        ({"expires_at": 1_000, "expires_in": 300}, 1_000),
        ({"expires_at": 1_300, "expires_in": 300}, 1_200),
        ({"expires_at": 1_300, "expires_in": "300"}, 1_200),
        # A millisecond or far-future expires_at is bounded by the relative one.
        ({"expires_at": 1_300_000, "expires_in": 300}, 1_200),
        ({"expires_at": 10**12, "expires_in": 300}, 1_200),
    ],
)
def test_the_access_deadline_is_the_earlier_representation(token_data, expected):
    # Imported here so the behavioural tests below still collect without the fix.
    from fastmcp.server.auth.oauth_proxy.proxy import _access_deadline

    assert _access_deadline(token_data, received_at=900) == expected


@pytest.mark.parametrize(
    ("token_data", "expected"),
    [
        ({"refresh_token_expires_at": 2_000, "refresh_expires_in": 60}, None),
        ({"refresh_token": "r"}, None),
        ({"refresh_token": "r", "refresh_expires_in": 0}, None),
        ({"refresh_token": "r", "refresh_expires_in": 3_600}, 3_700),
        ({"refresh_token": "r", "refresh_token_expires_at": 2_000}, 2_000),
        (
            {
                "refresh_token": "r",
                "refresh_token_expires_at": 2_000,
                "refresh_expires_in": 3_600,
            },
            2_000,
        ),
        (
            {
                "refresh_token": "r",
                "refresh_token_expires_at": 9_000,
                "refresh_expires_in": 3_600,
            },
            3_700,
        ),
    ],
)
def test_the_refresh_deadline_is_the_earlier_representation(token_data, expected):
    from fastmcp.server.auth.oauth_proxy.proxy import _refresh_deadline

    assert _refresh_deadline(token_data, received_at=100) == expected


@pytest.mark.parametrize(
    "value", [True, False, "1000", "soon", math.nan, math.inf, -math.inf, [1_000]]
)
@pytest.mark.parametrize("field", ["expires_at", "refresh_token_expires_at"])
def test_a_present_but_invalid_absolute_deadline_fails_closed(field, value):
    from fastmcp.server.auth.oauth_proxy.proxy import (
        _access_deadline,
        _refresh_deadline,
    )

    token_data = {
        "refresh_token": "r",
        "expires_at": 2_000,
        "expires_in": 300,
        "refresh_expires_in": 3_600,
        field: value,
    }
    deadline_of = _refresh_deadline if field.startswith("refresh") else _access_deadline
    with pytest.raises(TokenError) as refused:
        deadline_of(token_data, received_at=100)
    assert refused.value.error == "invalid_grant"


@pytest.mark.parametrize("value", [True, "soon", math.nan, math.inf, [300]])
@pytest.mark.parametrize("field", ["expires_in", "refresh_expires_in"])
def test_a_present_but_invalid_relative_lifetime_fails_closed(field, value):
    from fastmcp.server.auth.oauth_proxy.proxy import (
        _access_deadline,
        _refresh_deadline,
    )

    token_data = {
        "refresh_token": "r",
        "expires_at": 2_000,
        "expires_in": 300,
        "refresh_expires_in": 3_600,
        field: value,
    }
    deadline_of = _refresh_deadline if field.startswith("refresh") else _access_deadline
    with pytest.raises(TokenError) as refused:
        deadline_of(token_data, received_at=100)
    assert refused.value.error == "invalid_grant"


@pytest.mark.parametrize(
    ("deadline", "now", "remaining"),
    [
        (1_001, 999.9, 1),
        (1_001, 1_000.0, 1),
        (1_001, 1_000.0001, 0),
        (1_001, 1_000.4, 0),
        (1_001, 1_001.0, 0),
        (1_001, 1_002.5, -2),
    ],
)
def test_remaining_lifetime_is_floored_not_rounded_up(deadline, now, remaining):
    from fastmcp.server.auth.oauth_proxy.proxy import _seconds_until

    assert _seconds_until(deadline, now) == remaining
    # int(deadline) - int(now) would have granted a second that does not exist.
    if now == 1_000.4:
        assert int(deadline) - int(now) == 1


def test_a_storage_ttl_is_never_fabricated_for_a_passed_deadline(monkeypatch):
    from fastmcp.server.auth.oauth_proxy.proxy import _ttl_until

    monkeypatch.setattr(time, "time", lambda: 1_000.25)
    assert _ttl_until(1_001) == 0.75
    for deadline in (1_000.25, 999):
        with pytest.raises(TokenError) as refused:
            _ttl_until(deadline)
        assert refused.value.error == "invalid_grant"


def test_a_client_code_stored_before_the_fix_still_loads():
    stored = {
        "code": "code",
        "client_id": "test-client",
        "redirect_uri": REDIRECT,
        "code_challenge": "challenge",
        "code_challenge_method": "S256",
        "scopes": ["read"],
        "idp_tokens": {"access_token": "valid-upstream", "expires_in": 300},
        "expires_at": 1_000.0,
        "created_at": 700.0,
    }
    code = ClientCode.model_validate(stored)
    assert code.access_token_expires_at is None
    assert code.refresh_token_expires_at is None


def test_the_issuer_signs_an_absolute_exp(proxy):
    token = proxy.jwt_issuer.issue_access_token(
        client_id="test-client",
        scopes=["read"],
        jti="jti",
        expires_in=3_600,
        expires_at=2_000_000_000,
    )
    assert proxy.jwt_issuer.verify_token(token)["exp"] == 2_000_000_000


# --------------------------------------------------------------------------
# The callback snapshots the deadlines on receipt
# --------------------------------------------------------------------------


async def test_the_callback_records_the_deadlines_on_receipt(proxy):
    access_deadline = int(time.time()) + 100
    before = time.time()
    _, stored = await recorded_code(
        proxy,
        {
            "access_token": "valid-upstream",
            "refresh_token": "upstream-refresh",
            "expires_in": 300,
            "expires_at": access_deadline,
            "refresh_expires_in": 3_600,
            "token_type": "Bearer",
        },
    )
    after = time.time()

    assert stored.access_token_expires_at == access_deadline
    assert stored.refresh_token_expires_at is not None
    assert before + 3_600 <= stored.refresh_token_expires_at <= after + 3_600


@pytest.mark.parametrize(
    "expires_at",
    [
        pytest.param(lambda now: now + 600, id="later"),
        pytest.param(lambda now: int(now * 1_000), id="milliseconds"),
        pytest.param(lambda now: 10**12, id="far-future"),
    ],
)
async def test_the_callback_bounds_the_access_deadline_by_receipt(proxy, expires_at):
    before = time.time()
    _, stored = await recorded_code(
        proxy,
        {
            "access_token": "valid-upstream",
            "expires_in": 300,
            "expires_at": expires_at(before),
            "token_type": "Bearer",
        },
    )
    after = time.time()

    assert stored.access_token_expires_at is not None
    assert before + 300 <= stored.access_token_expires_at <= after + 300


@pytest.mark.parametrize("absolute_first", [True, False])
async def test_the_callback_records_the_earlier_refresh_deadline(proxy, absolute_first):
    before = time.time()
    absolute = before + (1_800 if absolute_first else 9_000)
    _, stored = await recorded_code(
        proxy,
        {
            "access_token": "valid-upstream",
            "refresh_token": "upstream-refresh",
            "expires_in": 300,
            "expires_at": int(before) + 300,
            "refresh_token_expires_at": absolute,
            "refresh_expires_in": 3_600,
            "token_type": "Bearer",
        },
    )
    after = time.time()

    assert stored.refresh_token_expires_at is not None
    if absolute_first:
        assert stored.refresh_token_expires_at == absolute
    else:
        assert before + 3_600 <= stored.refresh_token_expires_at <= after + 3_600


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("expires_at", math.nan),
        ("expires_at", math.inf),
        ("expires_at", "soon"),
        ("expires_at", True),
        ("expires_in", "soon"),
        ("refresh_token_expires_at", math.nan),
        ("refresh_token_expires_at", "soon"),
        ("refresh_token_expires_at", True),
        ("refresh_expires_in", "soon"),
    ],
)
async def test_the_callback_refuses_an_invalid_deadline(proxy, store, field, value):
    response = await callback(
        proxy,
        {
            "access_token": "valid-upstream",
            "refresh_token": "upstream-refresh",
            "expires_in": 300,
            "expires_at": int(time.time()) + 300,
            "refresh_expires_in": 3_600,
            "token_type": "Bearer",
            field: value,
        },
    )
    assert response.status_code == 500
    assert await store.keys("mcp-authorization-codes") == []


# --------------------------------------------------------------------------
# Downstream code redemption
# --------------------------------------------------------------------------


async def test_a_delayed_redemption_ends_at_the_callback_deadlines(proxy, store):
    client = await register(proxy)
    received_at = time.time() - DELAY  # the callback ran DELAY seconds ago
    access_deadline = math.floor(received_at) + 300
    refresh_deadline = received_at + 3_600
    result = await redeem(
        proxy,
        client,
        {
            "access_token": "valid-upstream",
            "refresh_token": "upstream-refresh",
            "expires_in": 300,
            "expires_at": access_deadline,
            "refresh_expires_in": 3_600,
            "token_type": "Bearer",
        },
        access_token_expires_at=access_deadline,
        refresh_token_expires_at=refresh_deadline,
    )

    access = proxy.jwt_issuer.verify_token(result.access_token)
    assert access["exp"] == access_deadline
    assert result.expires_in is not None
    assert result.expires_in <= 300 - DELAY
    access_jti, token_set = await stored_set(proxy, result.access_token)
    assert token_set.expires_at == access_deadline
    assert token_set.refresh_token_expires_at == refresh_deadline
    assert await entry_ends_at(store, access_jti, "mcp-jti-mappings") == pytest.approx(
        access_deadline, abs=0.5
    )
    # Upstream retention follows the refresh deadline, not the access lifetime.
    upstream_ends = await entry_ends_at(
        store, token_set.upstream_token_id, "mcp-upstream-tokens"
    )
    assert upstream_ends == pytest.approx(refresh_deadline, abs=0.5)

    assert result.refresh_token is not None
    refresh = proxy.jwt_issuer.verify_token(
        result.refresh_token, expected_token_use="refresh"
    )
    assert refresh["exp"] == math.floor(refresh_deadline)
    assert await entry_ends_at(
        store, refresh["jti"], "mcp-jti-mappings"
    ) == pytest.approx(refresh_deadline, abs=0.5)
    metadata = await proxy._refresh_token_store.get(
        key=_hash_token(result.refresh_token)
    )
    assert metadata is not None
    assert metadata.expires_at == math.floor(refresh_deadline)


async def test_a_malformed_expires_at_redeems_within_the_relative_lifetime(proxy):
    client = await register(proxy)
    before = time.time()
    code, _ = await recorded_code(
        proxy,
        {
            "access_token": "valid-upstream",
            "expires_in": 300,
            "expires_at": int(before * 1_000),
            "token_type": "Bearer",
        },
    )
    result = await exchange(proxy, client, code)

    assert result.expires_in is not None
    assert result.expires_in <= 300
    assert proxy.jwt_issuer.verify_token(result.access_token)["exp"] <= before + 301


async def test_a_refresh_only_absolute_deadline_is_honored_alone(proxy, store):
    client = await register(proxy)
    refresh_deadline = time.time() + 1_800.5
    code, stored = await recorded_code(
        proxy,
        {
            "access_token": "valid-upstream",
            "refresh_token": "upstream-refresh",
            "expires_in": 300,
            "refresh_token_expires_at": refresh_deadline,
            "token_type": "Bearer",
        },
    )
    assert stored.access_token_expires_at is None
    assert stored.refresh_token_expires_at == refresh_deadline

    result = await exchange(proxy, client, code)
    # No absolute access deadline: the relative access path is unchanged.
    access = proxy.jwt_issuer.verify_token(result.access_token)
    assert result.expires_in == 300
    assert access["exp"] - access["iat"] == 300
    assert result.refresh_token is not None
    refresh = proxy.jwt_issuer.verify_token(
        result.refresh_token, expected_token_use="refresh"
    )
    assert refresh["exp"] == math.floor(refresh_deadline)
    assert await entry_ends_at(
        store, refresh["jti"], "mcp-jti-mappings"
    ) == pytest.approx(refresh_deadline, abs=0.5)
    metadata = await proxy._refresh_token_store.get(
        key=_hash_token(result.refresh_token)
    )
    assert metadata is not None
    assert metadata.expires_at == math.floor(refresh_deadline)


async def test_redeeming_after_the_access_deadline_is_refused(proxy, store):
    client = await register(proxy)
    with pytest.raises(TokenError) as refused:
        await redeem(
            proxy,
            client,
            {"access_token": "valid-upstream", "expires_in": 300},
            access_token_expires_at=int(time.time()) - 1,
        )
    assert refused.value.error == "invalid_grant"
    assert await published(store) == NOTHING_PUBLISHED


async def test_redeeming_after_the_refresh_deadline_is_refused(proxy, store):
    client = await register(proxy)
    with pytest.raises(TokenError) as refused:
        await redeem(
            proxy,
            client,
            {
                "access_token": "valid-upstream",
                "refresh_token": "upstream-refresh",
                "expires_in": 300,
                "refresh_expires_in": 3_600,
            },
            refresh_token_expires_at=time.time() - 1,
        )
    assert refused.value.error == "invalid_grant"
    assert await published(store) == NOTHING_PUBLISHED


@pytest.mark.parametrize(
    ("seconds_before_deadline", "accepted"),
    [(1.0, True), (1.3, True), (0.6, False), (0.0001, False)],
)
async def test_redemption_at_a_fractional_boundary(
    proxy, monkeypatch, seconds_before_deadline, accepted
):
    client = await register(proxy)
    deadline = 2_000_000_000
    monkeypatch.setattr(time, "time", lambda: deadline - seconds_before_deadline)
    tokens = {"access_token": "valid-upstream", "expires_in": 300}
    if not accepted:
        with pytest.raises(TokenError):
            await redeem(proxy, client, tokens, access_token_expires_at=deadline)
        return
    result = await redeem(proxy, client, tokens, access_token_expires_at=deadline)
    assert result.expires_in == 1
    assert proxy.jwt_issuer.verify_token(result.access_token)["exp"] == deadline


async def test_a_legacy_code_is_bounded_by_its_response_expires_at(proxy, store):
    client = await register(proxy)
    deadline = int(time.time()) + 100
    # A code stored before deadlines were recorded: no ClientCode fields.
    result = await redeem(
        proxy,
        client,
        {
            "access_token": "valid-upstream",
            "refresh_token": "upstream-refresh",
            "expires_in": 300,
            "expires_at": deadline,
            "refresh_expires_in": 3_600,
            "token_type": "Bearer",
        },
    )
    access = proxy.jwt_issuer.verify_token(result.access_token)
    assert result.expires_in is not None
    assert result.expires_in <= 100
    assert access["exp"] == deadline
    access_jti, token_set = await stored_set(proxy, result.access_token)
    assert token_set.expires_at == deadline
    assert await entry_ends_at(store, access_jti, "mcp-jti-mappings") == pytest.approx(
        deadline, abs=0.5
    )
    # No absolute refresh deadline anywhere: the refresh lifetime stays relative.
    assert result.refresh_token is not None
    refresh = proxy.jwt_issuer.verify_token(
        result.refresh_token, expected_token_use="refresh"
    )
    assert refresh["exp"] - refresh["iat"] == 3_600


async def test_a_legacy_code_is_bounded_by_the_earlier_relative_lifetime(proxy):
    client = await register(proxy)
    before = time.time()
    result = await redeem(
        proxy,
        client,
        {
            "access_token": "valid-upstream",
            "expires_in": 300,
            "expires_at": int(before * 1_000),
            "token_type": "Bearer",
        },
    )
    assert result.expires_in is not None
    assert result.expires_in <= 300
    assert proxy.jwt_issuer.verify_token(result.access_token)["exp"] <= before + 301


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("expires_at", "soon"),
        ("expires_at", True),
        ("refresh_token_expires_at", "soon"),
    ],
)
async def test_a_legacy_code_with_an_invalid_deadline_is_refused(
    proxy, store, field, value
):
    client = await register(proxy)
    tokens = {
        "access_token": "valid-upstream",
        "refresh_token": "upstream-refresh",
        "expires_in": 300,
        "refresh_expires_in": 3_600,
        field: value,
    }
    with pytest.raises(TokenError) as refused:
        await redeem(proxy, client, tokens)
    assert refused.value.error == "invalid_grant"
    assert await published(store) == NOTHING_PUBLISHED


async def test_a_legacy_code_caps_the_refresh_fallback_at_its_refresh_deadline(
    configured, store
):
    client = await register(configured)
    refresh_deadline = time.time() + 1_800.5
    result = await redeem(
        configured,
        client,
        {
            "access_token": "valid-upstream",
            "refresh_token": "upstream-refresh",
            "expires_in": 300,
            "refresh_token_expires_at": refresh_deadline,
            "token_type": "Bearer",
        },
    )
    assert result.refresh_token is not None
    refresh = configured.jwt_issuer.verify_token(
        result.refresh_token, expected_token_use="refresh"
    )
    assert refresh["exp"] == math.floor(refresh_deadline)
    assert await entry_ends_at(
        store, refresh["jti"], "mcp-jti-mappings"
    ) == pytest.approx(refresh_deadline, abs=0.5)
    metadata = await configured._refresh_token_store.get(
        key=_hash_token(result.refresh_token)
    )
    assert metadata is not None
    assert metadata.expires_at == math.floor(refresh_deadline)


async def test_a_delayed_legacy_code_counts_its_relative_bound_from_creation(proxy):
    client = await register(proxy)
    created_at = time.time() - DELAY  # the code was created DELAY seconds ago
    result = await redeem(
        proxy,
        client,
        {
            "access_token": "valid-upstream",
            "refresh_token": "upstream-refresh",
            "expires_in": 300,
            "expires_at": int(time.time()) + 1_000,
            "refresh_expires_in": 3_600,
            "refresh_token_expires_at": int(time.time()) + 100_000,
        },
        created_at=created_at,
    )
    # Counted from redemption these would be 300 s and 3600 s.
    access = proxy.jwt_issuer.verify_token(result.access_token)
    assert access["exp"] == math.floor(created_at + 300)
    assert result.expires_in is not None
    assert result.expires_in <= 300 - DELAY
    _, token_set = await stored_set(proxy, result.access_token)
    assert token_set.expires_at == created_at + 300
    assert token_set.refresh_token_expires_at == created_at + 3_600
    assert result.refresh_token is not None
    refresh = proxy.jwt_issuer.verify_token(
        result.refresh_token, expected_token_use="refresh"
    )
    assert refresh["exp"] == math.floor(created_at + 3_600)


async def test_a_delayed_relative_only_legacy_code_is_unchanged(proxy):
    client = await register(proxy)
    result = await redeem(
        proxy,
        client,
        {"access_token": "valid-upstream", "expires_in": 300},
        created_at=time.time() - DELAY,
    )
    access = proxy.jwt_issuer.verify_token(result.access_token)
    assert result.expires_in == 300
    assert access["exp"] - access["iat"] == 300


@pytest.mark.parametrize("absolute", [True, False])
async def test_a_null_expires_in_is_absent_from_callback_to_redemption(proxy, absolute):
    client = await register(proxy)
    deadline = int(time.time()) + 200
    response = {"access_token": "valid-upstream", "expires_in": None}
    if absolute:
        response["expires_at"] = deadline
    code, stored = await recorded_code(proxy, response)
    absent = dict(response)
    del absent["expires_in"]
    _, stored_absent = await recorded_code(proxy, absent)
    assert stored.access_token_expires_at == stored_absent.access_token_expires_at

    result = await exchange(proxy, client, code)

    access = proxy.jwt_issuer.verify_token(result.access_token)
    if absolute:
        assert access["exp"] == deadline
        assert result.expires_in is not None
        assert result.expires_in <= 200
    else:
        # The relative fallback for a response without expires_in.
        assert result.expires_in is not None
        assert access["exp"] - access["iat"] == result.expires_in
        assert result.expires_in > 300


@pytest.mark.parametrize("absolute", [True, False])
async def test_a_null_refresh_expires_in_is_absent_from_callback_to_redemption(
    configured, store, absolute
):
    client = await register(configured)
    refresh_deadline = time.time() + 1_800.5
    response = {
        "access_token": "valid-upstream",
        "refresh_token": "upstream-refresh",
        "expires_in": 300,
        "refresh_expires_in": None,
    }
    if absolute:
        response["refresh_token_expires_at"] = refresh_deadline
    code, stored = await recorded_code(configured, response)
    assert stored.refresh_token_expires_at == (refresh_deadline if absolute else None)

    result = await exchange(configured, client, code)

    assert result.refresh_token is not None
    refresh = configured.jwt_issuer.verify_token(
        result.refresh_token, expected_token_use="refresh"
    )
    metadata = await configured._refresh_token_store.get(
        key=_hash_token(result.refresh_token)
    )
    assert metadata is not None
    if absolute:
        assert refresh["exp"] == math.floor(refresh_deadline)
        assert metadata.expires_at == math.floor(refresh_deadline)
        assert await entry_ends_at(
            store, refresh["jti"], "mcp-jti-mappings"
        ) == pytest.approx(refresh_deadline, abs=0.5)
    else:
        assert refresh["exp"] - refresh["iat"] == CONFIGURED_REFRESH_FALLBACK


async def test_without_any_absolute_deadline_redemption_is_unchanged(proxy):
    client = await register(proxy)
    result = await redeem(
        proxy,
        client,
        {
            "access_token": "valid-upstream",
            "refresh_token": "upstream-refresh",
            "expires_in": 300,
            "refresh_expires_in": 3_600,
            "token_type": "Bearer",
        },
    )
    access = proxy.jwt_issuer.verify_token(result.access_token)
    assert result.expires_in == 300
    assert access["exp"] - access["iat"] == 300
    assert result.refresh_token is not None
    refresh = proxy.jwt_issuer.verify_token(
        result.refresh_token, expected_token_use="refresh"
    )
    assert refresh["exp"] - refresh["iat"] == 3_600


# --------------------------------------------------------------------------
# Configured lifetimes are capped by absolute deadlines
# --------------------------------------------------------------------------


@pytest.mark.parametrize("recorded", [True, False])
async def test_a_configured_access_lifetime_is_capped_on_redemption(
    configured, store, recorded
):
    client = await register(configured)
    deadline = int(time.time()) + 100
    tokens = {
        "access_token": "valid-upstream",
        "refresh_token": "upstream-refresh",
        "expires_in": 300,
        "expires_at": deadline,
        "refresh_expires_in": 3_600,
    }
    result = await redeem(
        configured,
        client,
        tokens,
        access_token_expires_at=deadline if recorded else None,
        refresh_token_expires_at=time.time() + 3_600 if recorded else None,
    )
    assert result.expires_in is not None
    assert result.expires_in <= 100
    assert configured.jwt_issuer.verify_token(result.access_token)["exp"] == deadline
    access_jti, token_set = await stored_set(configured, result.access_token)
    assert token_set.expires_at == deadline
    assert await entry_ends_at(store, access_jti, "mcp-jti-mappings") == pytest.approx(
        deadline, abs=0.5
    )
    # The refresh lifetime is not clamped to the access deadline.
    assert token_set.refresh_token_expires_at is not None
    assert token_set.refresh_token_expires_at >= time.time() + 3_500


async def test_a_delayed_code_with_configured_lifetimes_ends_at_its_deadlines(
    configured, store, monkeypatch
):
    """Callback, then a redemption DELAY seconds later, through the real flow."""
    client = await register(configured)
    received = time.time()
    access_deadline = int(received) + 300
    code, stored = await recorded_code(
        configured,
        {
            "access_token": "valid-upstream",
            "refresh_token": "upstream-refresh",
            "expires_in": 300,
            "expires_at": access_deadline,
            "refresh_expires_in": 3_600,
            "token_type": "Bearer",
        },
    )
    refresh_deadline = stored.refresh_token_expires_at
    assert refresh_deadline is not None
    real_time = time.time
    monkeypatch.setattr(time, "time", lambda: real_time() + DELAY)
    result = await exchange(configured, client, code)

    # Access: the configured 3600 s is a maximum, capped at the upstream deadline.
    access = configured.jwt_issuer.verify_token(result.access_token)
    assert access["exp"] == access_deadline
    assert result.expires_in is not None
    assert result.expires_in <= 300 - DELAY
    access_jti, token_set = await stored_set(configured, result.access_token)
    assert token_set.expires_at == access_deadline
    assert await entry_ends_at(store, access_jti, "mcp-jti-mappings") == pytest.approx(
        access_deadline, abs=0.5
    )
    # Refresh: the recorded deadline, neither re-based nor clamped to 300 s.
    assert token_set.refresh_token_expires_at == refresh_deadline
    assert result.refresh_token is not None
    refresh = configured.jwt_issuer.verify_token(
        result.refresh_token, expected_token_use="refresh"
    )
    assert refresh["exp"] == math.floor(refresh_deadline)
    assert await entry_ends_at(
        store, refresh["jti"], "mcp-jti-mappings"
    ) == pytest.approx(refresh_deadline, abs=0.5)
    metadata = await configured._refresh_token_store.get(
        key=_hash_token(result.refresh_token)
    )
    assert metadata is not None
    assert metadata.expires_at == math.floor(refresh_deadline)
    assert await entry_ends_at(
        store, token_set.upstream_token_id, "mcp-upstream-tokens"
    ) == pytest.approx(refresh_deadline, abs=0.5)


async def test_a_configured_recorded_refresh_deadline_beats_the_fallback(
    configured, store
):
    client = await register(configured)
    refresh_deadline = time.time() + 600.5
    result = await redeem(
        configured,
        client,
        {"access_token": "valid-upstream", "refresh_token": "upstream-refresh"},
        refresh_token_expires_at=refresh_deadline,
    )
    assert result.refresh_token is not None
    refresh = configured.jwt_issuer.verify_token(
        result.refresh_token, expected_token_use="refresh"
    )
    assert refresh["exp"] == math.floor(refresh_deadline)
    assert await entry_ends_at(
        store, refresh["jti"], "mcp-jti-mappings"
    ) == pytest.approx(refresh_deadline, abs=0.5)


async def test_without_any_absolute_deadline_configured_lifetimes_are_unchanged(
    configured,
):
    client = await register(configured)
    result = await redeem(
        configured,
        client,
        {
            "access_token": "valid-upstream",
            "refresh_token": "upstream-refresh",
            "expires_in": 300,
        },
    )
    access = configured.jwt_issuer.verify_token(result.access_token)
    assert result.expires_in == CONFIGURED_ACCESS
    assert access["exp"] - access["iat"] == CONFIGURED_ACCESS
    assert result.refresh_token is not None
    refresh = configured.jwt_issuer.verify_token(
        result.refresh_token, expected_token_use="refresh"
    )
    assert refresh["exp"] - refresh["iat"] == CONFIGURED_REFRESH_FALLBACK


# --------------------------------------------------------------------------
# Downstream refresh
# --------------------------------------------------------------------------


async def signed_in(proxy: OAuthProxy, *, refresh_deadline: float | None = None):
    client = await register(proxy)
    result = await redeem(
        proxy,
        client,
        {
            "access_token": "valid-upstream",
            "refresh_token": "upstream-refresh",
            "expires_in": 300,
            "refresh_expires_in": 3_600,
            "token_type": "Bearer",
        },
        access_token_expires_at=int(time.time()) + 300,
        refresh_token_expires_at=refresh_deadline or time.time() + 3_600,
    )
    return client, result


def refresh_response(
    expires_at: object = None,
    *,
    rotate: bool = True,
    refresh_token_expires_at: object = None,
) -> dict:
    response: dict = {
        "access_token": "refreshed-upstream",
        "token_type": "Bearer",
        "expires_in": 300,
        "scope": "read",
    }
    if expires_at is not None:
        response["expires_at"] = expires_at
    if rotate:
        response["refresh_token"] = "upstream-refresh-2"
        response["refresh_expires_in"] = 3_600
    if refresh_token_expires_at is not None:
        response["refresh_token_expires_at"] = refresh_token_expires_at
    return response


async def refresh(proxy: OAuthProxy, client, result, response: dict):
    upstream = AsyncMock()
    upstream.refresh_token = AsyncMock(return_value=response)
    with patch.object(proxy, "_create_upstream_oauth_client", return_value=upstream):
        return await proxy.exchange_refresh_token(
            client=client,
            refresh_token=RefreshToken(
                token=result.refresh_token, client_id="test-client", scopes=["read"]
            ),
            scopes=["read"],
        )


async def test_a_downstream_refresh_ends_at_the_response_deadline(proxy, store):
    client, result = await signed_in(proxy)
    deadline = int(time.time()) + 300 - DELAY  # the response arrived DELAY s ago
    refreshed = await refresh(proxy, client, result, refresh_response(deadline))

    access = proxy.jwt_issuer.verify_token(refreshed.access_token)
    assert access["exp"] == deadline
    assert refreshed.expires_in is not None
    assert refreshed.expires_in <= 300 - DELAY
    access_jti, token_set = await stored_set(proxy, refreshed.access_token)
    assert token_set.expires_at == deadline
    assert await entry_ends_at(store, access_jti, "mcp-jti-mappings") == pytest.approx(
        deadline, abs=0.5
    )
    assert token_set.refresh_token_expires_at is not None
    assert refreshed.refresh_token is not None
    refresh_deadline = token_set.refresh_token_expires_at
    rotated = proxy.jwt_issuer.verify_token(
        refreshed.refresh_token, expected_token_use="refresh"
    )
    assert rotated["exp"] == math.floor(refresh_deadline)
    assert await entry_ends_at(
        store, rotated["jti"], "mcp-jti-mappings"
    ) == pytest.approx(refresh_deadline, abs=0.5)
    metadata = await proxy._refresh_token_store.get(
        key=_hash_token(refreshed.refresh_token)
    )
    assert metadata is not None
    assert metadata.expires_at == math.floor(refresh_deadline)


async def test_a_downstream_refresh_bounds_a_malformed_expires_at(proxy):
    client, result = await signed_in(proxy)
    before = time.time()
    response = refresh_response(int(before * 1_000))
    refreshed = await refresh(proxy, client, result, response)

    _, token_set = await stored_set(proxy, refreshed.access_token)
    assert before + 300 <= token_set.expires_at <= time.time() + 300
    exp = proxy.jwt_issuer.verify_token(refreshed.access_token)["exp"]
    assert exp == math.floor(token_set.expires_at)


@pytest.mark.parametrize("absolute_first", [True, False])
async def test_a_downstream_refresh_takes_the_earlier_refresh_deadline(
    proxy, absolute_first
):
    client, result = await signed_in(proxy)
    before = time.time()
    absolute = int(before) + (1_800 if absolute_first else 9_000)
    response = refresh_response(int(before) + 300, refresh_token_expires_at=absolute)
    refreshed = await refresh(proxy, client, result, response)
    after = time.time()

    _, token_set = await stored_set(proxy, refreshed.access_token)
    deadline = token_set.refresh_token_expires_at
    assert deadline is not None
    if absolute_first:
        assert deadline == absolute
    else:
        assert before + 3_600 <= deadline <= after + 3_600
    assert refreshed.refresh_token is not None
    rotated = proxy.jwt_issuer.verify_token(
        refreshed.refresh_token, expected_token_use="refresh"
    )
    assert rotated["exp"] == math.floor(deadline)


async def test_a_downstream_refresh_caps_the_configured_access_lifetime(
    configured, store
):
    client, result = await signed_in(configured)
    deadline = int(time.time()) + 100
    new_refresh_deadline = int(time.time()) + 7_200
    response = refresh_response(deadline, refresh_token_expires_at=new_refresh_deadline)
    response["refresh_expires_in"] = 9_000
    refreshed = await refresh(configured, client, result, response)

    access = configured.jwt_issuer.verify_token(refreshed.access_token)
    assert access["exp"] == deadline
    assert refreshed.expires_in is not None
    assert refreshed.expires_in <= 100
    access_jti, token_set = await stored_set(configured, refreshed.access_token)
    assert token_set.expires_at == deadline
    assert await entry_ends_at(store, access_jti, "mcp-jti-mappings") == pytest.approx(
        deadline, abs=0.5
    )
    # The fresh response's refresh deadline advances the recorded one.
    assert token_set.refresh_token_expires_at == new_refresh_deadline
    assert refreshed.refresh_token is not None
    rotated = configured.jwt_issuer.verify_token(
        refreshed.refresh_token, expected_token_use="refresh"
    )
    assert rotated["exp"] == new_refresh_deadline
    assert await entry_ends_at(
        store, rotated["jti"], "mcp-jti-mappings"
    ) == pytest.approx(new_refresh_deadline, abs=0.5)
    metadata = await configured._refresh_token_store.get(
        key=_hash_token(refreshed.refresh_token)
    )
    assert metadata is not None
    assert metadata.expires_at == new_refresh_deadline


async def test_a_downstream_refresh_without_deadlines_keeps_configured_lifetimes(
    configured,
):
    client = await register(configured)
    result = await redeem(
        configured,
        client,
        {"access_token": "valid-upstream", "refresh_token": "upstream-refresh"},
    )
    response = refresh_response(rotate=False)
    refreshed = await refresh(configured, client, result, response)

    access = configured.jwt_issuer.verify_token(refreshed.access_token)
    assert refreshed.expires_in == CONFIGURED_ACCESS
    assert access["exp"] - access["iat"] == CONFIGURED_ACCESS


# --------------------------------------------------------------------------
# A slow upstream-claims hook never restarts the clock
# --------------------------------------------------------------------------

SHORT_CONFIGURED_ACCESS = 60


@pytest.fixture
def short_configured(verifier, store) -> OAuthProxy:
    return make_proxy(
        verifier, store, fastmcp_access_token_expiry_seconds=SHORT_CONFIGURED_ACCESS
    )


def slow_claims_hook(proxy: OAuthProxy, monkeypatch, seconds: float) -> None:
    """Make the upstream-claims hook take `seconds` of (patched) wall time."""
    real_time = time.time
    offset = [0.0]
    monkeypatch.setattr(time, "time", lambda: real_time() + offset[0])

    async def extract(_token_data):
        offset[0] += seconds

    monkeypatch.setattr(proxy, "_extract_upstream_claims", extract)


async def test_a_slow_hook_keeps_the_configured_redemption_expiry_fixed(
    short_configured, store, monkeypatch
):
    client = await register(short_configured)
    start = time.time()
    tokens = {"access_token": "valid-upstream", "refresh_token": "upstream-refresh"}
    slow_claims_hook(short_configured, monkeypatch, 50)
    result = await redeem(
        short_configured,
        client,
        tokens,
        access_token_expires_at=start + 100,
        refresh_token_expires_at=start + 3_600,
    )

    # Fixed when calculated: start + 60, not hook end (start + 50) + 60.
    exp = short_configured.jwt_issuer.verify_token(result.access_token)["exp"]
    assert start + 59 <= exp <= start + SHORT_CONFIGURED_ACCESS
    assert result.expires_in is not None
    assert result.expires_in <= SHORT_CONFIGURED_ACCESS - 50
    access_jti, token_set = await stored_set(short_configured, result.access_token)
    assert token_set.expires_at == start + 100
    assert await entry_ends_at(store, access_jti, "mcp-jti-mappings") == pytest.approx(
        exp, abs=0.5
    )


@pytest.mark.parametrize(
    ("configured_access", "upstream_lifetime"),
    [(True, 100), (False, 30)],
    ids=["configured-maximum", "upstream-deadline"],
)
async def test_a_slow_hook_crossing_the_redemption_expiry_is_refused(
    verifier, store, monkeypatch, configured_access, upstream_lifetime
):
    config = (
        {"fastmcp_access_token_expiry_seconds": SHORT_CONFIGURED_ACCESS}
        if configured_access
        else {}
    )
    proxy = make_proxy(verifier, store, **config)
    client = await register(proxy)
    start = time.time()
    slow_claims_hook(proxy, monkeypatch, 70)
    with pytest.raises(TokenError) as refused:
        await redeem(
            proxy,
            client,
            {"access_token": "valid-upstream", "refresh_token": "upstream-refresh"},
            access_token_expires_at=start + upstream_lifetime,
            refresh_token_expires_at=start + 3_600,
        )
    assert refused.value.error == "invalid_grant"
    assert await store.keys("mcp-jti-mappings") == []
    assert await store.keys("mcp-refresh-tokens") == []


async def test_a_slow_hook_keeps_the_configured_refresh_expiry_fixed(
    short_configured, store, monkeypatch
):
    client, result = await signed_in(short_configured)
    start = time.time()
    slow_claims_hook(short_configured, monkeypatch, 50)
    refreshed = await refresh(
        short_configured, client, result, refresh_response(int(start) + 100)
    )

    exp = short_configured.jwt_issuer.verify_token(refreshed.access_token)["exp"]
    assert start + 59 <= exp <= start + SHORT_CONFIGURED_ACCESS
    assert refreshed.expires_in is not None
    assert refreshed.expires_in <= SHORT_CONFIGURED_ACCESS - 50
    access_jti, token_set = await stored_set(short_configured, refreshed.access_token)
    assert token_set.expires_at == int(start) + 100
    assert await entry_ends_at(store, access_jti, "mcp-jti-mappings") == pytest.approx(
        exp, abs=0.5
    )


async def test_a_slow_hook_crossing_the_refresh_expiry_is_refused(
    short_configured, store, monkeypatch
):
    client, result = await signed_in(short_configured)
    assert result.refresh_token is not None
    old_metadata_key = _hash_token(result.refresh_token)
    start = time.time()
    slow_claims_hook(short_configured, monkeypatch, 70)
    with pytest.raises(TokenError) as refused:
        await refresh(
            short_configured, client, result, refresh_response(int(start) + 100)
        )
    assert refused.value.error == "invalid_grant"
    # The refused rotation published no new refresh token; the old one stays.
    assert await store.keys("mcp-refresh-tokens") == [old_metadata_key]


async def test_a_downstream_refresh_keeps_the_recorded_refresh_deadline(proxy):
    refresh_deadline = time.time() + 1_800.5
    client, result = await signed_in(proxy, refresh_deadline=refresh_deadline)
    deadline = int(time.time()) + 300
    refreshed = await refresh(
        proxy, client, result, refresh_response(deadline, rotate=False)
    )

    assert refreshed.refresh_token is not None
    rotated = proxy.jwt_issuer.verify_token(
        refreshed.refresh_token, expected_token_use="refresh"
    )
    assert rotated["exp"] == math.floor(refresh_deadline)
    _, token_set = await stored_set(proxy, refreshed.access_token)
    assert token_set.refresh_token_expires_at == refresh_deadline


@pytest.mark.parametrize(
    "response",
    [
        pytest.param(lambda: refresh_response(int(time.time()) - 1), id="past"),
        pytest.param(lambda: refresh_response(True), id="bool"),
        pytest.param(lambda: refresh_response(math.nan), id="nan"),
        pytest.param(lambda: refresh_response("1000"), id="string"),
        pytest.param(
            lambda: refresh_response(
                int(time.time()) + 300, refresh_token_expires_at="soon"
            ),
            id="refresh-string",
        ),
        pytest.param(
            lambda: refresh_response(
                int(time.time()) + 300, refresh_token_expires_at=time.time() - 1
            ),
            id="refresh-past",
        ),
    ],
)
async def test_a_downstream_refresh_with_a_bad_deadline_publishes_nothing(
    proxy, store, response
):
    client, result = await signed_in(proxy)
    before = await everything(store)
    with pytest.raises(TokenError) as refused:
        await refresh(proxy, client, result, response())
    assert refused.value.error == "invalid_grant"
    assert await everything(store) == before


@pytest.mark.parametrize("access_absolute", [True, False])
async def test_a_downstream_refresh_past_the_recorded_refresh_deadline_is_refused(
    proxy, store, access_absolute
):
    client, result = await signed_in(proxy)
    # The upstream still answers, but the refresh deadline recorded for this
    # session has passed: no refresh token with a fabricated lifetime is issued,
    # whether or not the response carries an absolute access deadline.
    _, token_set = await stored_set(proxy, result.access_token)
    token_set.refresh_token_expires_at = time.time() - 1
    await proxy._upstream_token_store.put(
        key=token_set.upstream_token_id, value=token_set, ttl=3_600
    )
    before = await everything(store)
    expires_at = int(time.time()) + 300 if access_absolute else None
    with pytest.raises(TokenError) as refused:
        await refresh(proxy, client, result, refresh_response(expires_at, rotate=False))
    assert refused.value.error == "invalid_grant"
    assert await everything(store) == before


# --------------------------------------------------------------------------
# Transparent refresh
# --------------------------------------------------------------------------


async def expired_session(proxy: OAuthProxy) -> str:
    """A session whose access token expired; its refresh deadline is authoritative."""
    await proxy._upstream_token_store.put(
        key="upstream-id",
        value=UpstreamTokenSet(
            upstream_token_id="upstream-id",
            access_token="expired-upstream",
            refresh_token="upstream-refresh",
            refresh_token_expires_at=time.time() + 3_600,
            refresh_token_expires_at_authoritative=True,
            expires_at=time.time() - 60,
            token_type="Bearer",
            scope="read",
            client_id="test-client",
            created_at=time.time() - 600,
        ),
        ttl=3_600,
    )
    await proxy._jti_mapping_store.put(
        key="access-jti",
        value=JTIMapping(
            jti="access-jti", upstream_token_id="upstream-id", created_at=0
        ),
        ttl=3_600,
    )
    return proxy.jwt_issuer.issue_access_token(
        client_id="test-client", scopes=["read"], jti="access-jti", expires_in=3_600
    )


async def transparently_refresh(proxy: OAuthProxy, token: str, response: dict):
    upstream = AsyncMock()
    upstream.refresh_token = AsyncMock(return_value=response)
    with patch.object(proxy, "_create_upstream_oauth_client", return_value=upstream):
        return await proxy.load_access_token(token)


async def test_a_transparent_refresh_ends_at_the_response_deadline(proxy, store):
    token = await expired_session(proxy)
    before = await proxy._upstream_token_store.get(key="upstream-id")
    assert before is not None
    deadline = int(time.time()) + 300 - DELAY
    validated = await transparently_refresh(
        proxy, token, refresh_response(deadline, rotate=False)
    )

    # A plain OAuthProxy returns the verifier's AccessToken unchanged; only an
    # alternate-verification provider (OIDCProxy with verify_id_token) copies
    # the stored expires_at onto it, so the stored set is what is asserted.
    assert validated is not None
    token_set = await proxy._upstream_token_store.get(key="upstream-id")
    assert token_set is not None
    assert token_set.expires_at == deadline
    assert token_set.refresh_token_expires_at == before.refresh_token_expires_at
    assert before.refresh_token_expires_at is not None
    assert await entry_ends_at(
        store, "upstream-id", "mcp-upstream-tokens"
    ) == pytest.approx(before.refresh_token_expires_at, abs=1)


async def test_a_transparent_refresh_bounds_a_malformed_expires_at(proxy):
    token = await expired_session(proxy)
    before = time.time()
    validated = await transparently_refresh(
        proxy, token, refresh_response(10**12, refresh_token_expires_at=before + 900)
    )

    assert validated is not None
    token_set = await proxy._upstream_token_store.get(key="upstream-id")
    assert token_set is not None
    assert before + 300 <= token_set.expires_at <= time.time() + 300
    assert token_set.refresh_token_expires_at == before + 900


LONG_CONFIGURED_ACCESS = 30 * 86_400


async def test_a_transparent_refresh_does_not_extend_retention_by_configuration(
    verifier, store
):
    proxy = make_proxy(
        verifier, store, fastmcp_access_token_expiry_seconds=LONG_CONFIGURED_ACCESS
    )
    token = await expired_session(proxy)
    before = await proxy._upstream_token_store.get(key="upstream-id")
    assert before is not None
    refresh_deadline = before.refresh_token_expires_at
    assert refresh_deadline is not None
    deadline = int(time.time()) + 100
    validated = await transparently_refresh(
        proxy, token, refresh_response(deadline, rotate=False)
    )

    assert validated is not None
    token_set = await proxy._upstream_token_store.get(key="upstream-id")
    assert token_set is not None
    assert token_set.expires_at == deadline
    assert token_set.refresh_token_expires_at == refresh_deadline
    retained_until = await entry_ends_at(store, "upstream-id", "mcp-upstream-tokens")
    # Retained for the refresh token on its own, well past the short access
    # deadline, and not extended to the 30-day configured access lifetime.
    assert retained_until == pytest.approx(refresh_deadline, abs=1)
    assert retained_until > deadline + 3_000
    assert retained_until < time.time() + LONG_CONFIGURED_ACCESS / 2


@pytest.mark.parametrize("expires_at", ["past", True, math.inf])
async def test_a_transparent_refresh_with_a_bad_deadline_stores_nothing(
    proxy, expires_at
):
    token = await expired_session(proxy)
    before = await proxy._upstream_token_store.get(key="upstream-id")
    if expires_at == "past":
        expires_at = int(time.time()) - 1
    assert (
        await transparently_refresh(proxy, token, refresh_response(expires_at)) is None
    )
    assert await proxy._upstream_token_store.get(key="upstream-id") == before


# --------------------------------------------------------------------------
# Retention for outstanding access tokens
# --------------------------------------------------------------------------

RETENTION_CONFIG = {
    "fastmcp_access_token_expiry_seconds": 3_600,
    "fallback_access_token_expiry_seconds": 60,
    "fallback_refresh_token_expiry_seconds": 600,
}


class Clock:
    """A patched clock the test advances, shared by the proxy and MemoryStore.

    py-key-value stamps and expires entries with its own `datetime.now`, so it
    is patched too: entries then really expire at the patched time.
    """

    def __init__(self, monkeypatch) -> None:
        self.start = time.time()
        self.offset = 0.0
        real_time = time.time
        monkeypatch.setattr(time, "time", lambda: real_time() + self.offset)

        def now() -> datetime:
            return datetime.fromtimestamp(time.time(), tz=timezone.utc)

        def now_plus(seconds: float) -> datetime:
            return now() + timedelta(seconds=seconds)

        for module in (time_to_live, managed_entry):
            monkeypatch.setattr(module, "now", now)
            monkeypatch.setattr(module, "now_plus", now_plus)


def upstream_rejects(verifier, rejected: set[str]) -> None:
    """Make the verifier reject the upstream access tokens in `rejected`."""

    async def verify(token: str) -> AccessToken | None:
        if token in rejected:
            return None
        return AccessToken(token=token, client_id="test-client", scopes=["read"])

    verifier.verify_token.side_effect = verify


def refreshed_upstream(token: str, expires_at: int) -> dict:
    """A refresh response with an access deadline and no refresh expiry."""
    return {
        "access_token": token,
        "token_type": "Bearer",
        "expires_in": 60,
        "expires_at": expires_at,
    }


async def relative_session(proxy: OAuthProxy, client, response: dict):
    """Redeem a code from a real callback for `response`."""
    code, _ = await recorded_code(proxy, response)
    result = await exchange(proxy, client, code)
    access_jti, token_set = await stored_set(proxy, result.access_token)
    return result, access_jti, token_set


async def test_a_transparent_refresh_keeps_the_record_for_outstanding_tokens(
    verifier, store, monkeypatch
):
    """Fauna's scenario: a synthetic refresh expiry must not drop the record."""
    proxy = make_proxy(verifier, store, **RETENTION_CONFIG)
    client = await register(proxy)
    clock = Clock(monkeypatch)
    t = clock.start
    rejected = {"stale-upstream"}
    upstream_rejects(verifier, rejected)
    result, access_jti, token_set = await relative_session(
        proxy,
        client,
        {"access_token": "stale-upstream", "refresh_token": "upstream-refresh"},
    )
    # Relative issuance: a 3600 s access JWT and JTI, a synthetic refresh expiry.
    access = proxy.jwt_issuer.verify_token(result.access_token)
    assert access["exp"] - access["iat"] == 3_600
    assert token_set.refresh_token_expires_at == pytest.approx(t + 600, abs=1)
    assert not token_set.refresh_token_expires_at_authoritative
    jti_ends = await entry_ends_at(store, access_jti, "mcp-jti-mappings")

    upstream = AsyncMock()
    upstream.refresh_token = AsyncMock(
        side_effect=[
            refreshed_upstream("refreshed-upstream", int(t) + 121),
            refreshed_upstream("refreshed-upstream-2", int(t) + 661),
        ]
    )
    with patch.object(proxy, "_create_upstream_oauth_client", return_value=upstream):
        clock.offset = 61
        assert await proxy.load_access_token(result.access_token) is not None
        retained = await entry_ends_at(
            store, token_set.upstream_token_id, "mcp-upstream-tokens"
        )
        # The record outlives the outstanding JWT/JTI, not the synthetic t+600.
        assert retained >= jti_ends - 1

        # A request at t+601 still loads and refreshes with the refresh token.
        clock.offset = 601
        rejected.add("refreshed-upstream")
        validated = await proxy.load_access_token(result.access_token)
    assert validated is not None
    assert validated.token == "refreshed-upstream-2"
    assert upstream.refresh_token.call_count == 2


async def test_a_downstream_refresh_keeps_the_record_for_outstanding_tokens(
    verifier, store, monkeypatch
):
    proxy = make_proxy(verifier, store, **RETENTION_CONFIG)
    client = await register(proxy)
    clock = Clock(monkeypatch)
    result, access_jti, token_set = await relative_session(
        proxy,
        client,
        {"access_token": "valid-upstream", "refresh_token": "upstream-refresh"},
    )
    jti_ends = await entry_ends_at(store, access_jti, "mcp-jti-mappings")

    clock.offset = 61
    response = refreshed_upstream("refreshed-upstream", int(clock.start) + 121)
    await refresh(proxy, client, result, response)

    # The first access JWT (3600 s, relative) still references the record.
    retained = await entry_ends_at(
        store, token_set.upstream_token_id, "mcp-upstream-tokens"
    )
    assert retained >= jti_ends - 1


async def test_a_transparent_refresh_ends_retention_at_an_authoritative_deadline(
    verifier, store, monkeypatch
):
    proxy = make_proxy(verifier, store, **RETENTION_CONFIG)
    client = await register(proxy)
    clock = Clock(monkeypatch)
    upstream_rejects(verifier, {"stale-upstream"})
    result, access_jti, token_set = await relative_session(
        proxy,
        client,
        {
            "access_token": "stale-upstream",
            "refresh_token": "upstream-refresh",
            "refresh_expires_in": 600,
        },
    )
    refresh_deadline = token_set.refresh_token_expires_at
    assert refresh_deadline is not None
    assert token_set.refresh_token_expires_at_authoritative
    jti_ends = await entry_ends_at(store, access_jti, "mcp-jti-mappings")

    upstream = AsyncMock()
    upstream.refresh_token = AsyncMock(
        side_effect=[
            refreshed_upstream("refreshed-upstream", int(clock.start) + 121),
            refreshed_upstream("refreshed-upstream-2", int(clock.start) + 661),
        ]
    )
    with patch.object(proxy, "_create_upstream_oauth_client", return_value=upstream):
        clock.offset = 61
        assert await proxy.load_access_token(result.access_token) is not None
        retained = await entry_ends_at(
            store, token_set.upstream_token_id, "mcp-upstream-tokens"
        )
        # Past the authoritative refresh deadline the record serves no token.
        assert retained == pytest.approx(refresh_deadline, abs=1)
        assert retained < jti_ends - 2_000

        # At t+601 the JWT and its JTI are still valid, but the real refresh
        # authority is gone: the request fails closed, with no upstream refresh.
        clock.offset = 601
        assert await proxy._jti_mapping_store.get(key=access_jti) is not None
        assert await proxy.load_access_token(result.access_token) is None
    assert upstream.refresh_token.call_count == 1


async def legacy_session(proxy: OAuthProxy, store: MemoryStore, t: float) -> str:
    """An upstream record serialized before refresh-expiry provenance existed.

    Written as a raw dict through the store, read back by the real model
    adapter: it has a stored refresh expiry (t+600) but no provenance field.
    Returns a FastMCP access token (3600 s) that references it.
    """
    await store.put(
        "legacy-id",
        {
            "upstream_token_id": "legacy-id",
            "access_token": "stale-upstream",
            "refresh_token": "upstream-refresh",
            "refresh_token_expires_at": t + 600,
            "expires_at": t + 60,
            "token_type": "Bearer",
            "scope": "read",
            "client_id": "test-client",
            "created_at": t,
            "raw_token_data": {},
        },
        collection="mcp-upstream-tokens",
        ttl=3_600,
    )
    for jti in ("legacy-access", "legacy-refresh"):
        await store.put(
            jti,
            {"jti": jti, "upstream_token_id": "legacy-id", "created_at": t},
            collection="mcp-jti-mappings",
            ttl=3_600,
        )
    loaded = await proxy._upstream_token_store.get(key="legacy-id")
    assert loaded is not None
    assert loaded.refresh_token_expires_at_authoritative  # unknown: a hard bound
    return proxy.jwt_issuer.issue_access_token(
        client_id="test-client", scopes=["read"], jti="legacy-access", expires_in=3_600
    )


async def test_a_legacy_record_bounds_retention_on_transparent_refresh(
    verifier, store, monkeypatch
):
    proxy = make_proxy(verifier, store, **RETENTION_CONFIG)
    clock = Clock(monkeypatch)
    t = clock.start
    upstream_rejects(verifier, {"stale-upstream"})
    token = await legacy_session(proxy, store, t)

    clock.offset = 61
    validated = await transparently_refresh(
        proxy, token, refreshed_upstream("refreshed-upstream", int(t) + 121)
    )

    assert validated is not None
    retained = await entry_ends_at(store, "legacy-id", "mcp-upstream-tokens")
    # Never past max(real access deadline, stored refresh deadline).
    assert retained <= max(int(t) + 121, t + 600) + 0.5
    assert retained == pytest.approx(t + 600, abs=1)


async def test_a_legacy_record_bounds_retention_on_downstream_refresh(
    verifier, store, monkeypatch
):
    proxy = make_proxy(verifier, store, **RETENTION_CONFIG)
    client = await register(proxy)
    clock = Clock(monkeypatch)
    t = clock.start
    await legacy_session(proxy, store, t)
    refresh_token = proxy.jwt_issuer.issue_refresh_token(
        client_id="test-client", scopes=["read"], jti="legacy-refresh", expires_in=600
    )

    clock.offset = 61
    upstream = AsyncMock()
    upstream.refresh_token = AsyncMock(
        return_value=refreshed_upstream("refreshed-upstream", int(t) + 121)
    )
    with patch.object(proxy, "_create_upstream_oauth_client", return_value=upstream):
        refreshed = await proxy.exchange_refresh_token(
            client=client,
            refresh_token=RefreshToken(
                token=refresh_token, client_id="test-client", scopes=["read"]
            ),
            scopes=["read"],
        )

    retained = await entry_ends_at(store, "legacy-id", "mcp-upstream-tokens")
    assert retained <= max(int(t) + 121, t + 600) + 0.5
    assert retained == pytest.approx(t + 600, abs=1)
    # The rotated refresh token ends at the stored refresh deadline too.
    assert refreshed.refresh_token is not None
    rotated = proxy.jwt_issuer.verify_token(
        refreshed.refresh_token, expected_token_use="refresh"
    )
    assert rotated["exp"] == math.floor(t + 600)
