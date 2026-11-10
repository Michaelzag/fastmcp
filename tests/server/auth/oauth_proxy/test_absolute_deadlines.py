"""Absolute upstream deadlines are honoured at every use site, never re-based.

authlib records ``expires_at`` (absolute) when a token response arrives. A
redemption or refresh that happens later must end the FastMCP JWT, its JTI
mapping and the stored upstream token at that deadline, not at
``now + expires_in``. Responses without an absolute deadline keep the
relative behaviour; a present but invalid one is refused.
"""

import math
import time
from unittest.mock import AsyncMock, Mock, patch
from urllib.parse import parse_qs, urlparse

import pytest
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


@pytest.fixture
def proxy(verifier, store) -> OAuthProxy:
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
    )
    proxy.set_mcp_path("/mcp")
    return proxy


async def register(proxy: OAuthProxy) -> OAuthClientInformationFull:
    client = OAuthClientInformationFull(
        client_id="test-client",
        client_secret="test-secret",
        redirect_uris=[AnyUrl(REDIRECT)],
    )
    await proxy.register_client(client)
    return client


async def redeem(
    proxy: OAuthProxy,
    client: OAuthClientInformationFull,
    idp_tokens: dict,
    *,
    refresh_token_expires_at: float | None = None,
):
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
            created_at=time.time(),
            refresh_token_expires_at=refresh_token_expires_at,
        ),
    )
    return await proxy.exchange_authorization_code(
        client=client,
        authorization_code=AuthorizationCode(
            code="code",
            scopes=["read"],
            expires_at=time.time() + 300,
            client_id="test-client",
            code_challenge="challenge",
            redirect_uri=AnyUrl(REDIRECT),
            redirect_uri_provided_explicitly=True,
        ),
    )


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


async def published(store: MemoryStore) -> dict:
    """Every token record a redemption could publish."""
    collections = ("mcp-upstream-tokens", "mcp-jti-mappings", "mcp-refresh-tokens")
    return {name: await store.keys(name) for name in collections}


# --------------------------------------------------------------------------
# Helpers: presence, validity and floor
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("token_data", "expected"),
    [
        ({"expires_at": 1_000}, 1_000),
        ({"expires_at": 1_000.9}, 1_000),
        ({"expires_at": None}, None),
        ({"expires_in": 300}, None),
    ],
)
def test_an_absolute_deadline_is_a_finite_number_or_absent(token_data, expected):
    # Imported here so the behavioural tests below still collect without the fix.
    from fastmcp.server.auth.oauth_proxy.proxy import _absolute_deadline

    assert _absolute_deadline(token_data) == expected


@pytest.mark.parametrize(
    "value", [True, False, "1000", "soon", math.nan, math.inf, -math.inf, [1_000]]
)
def test_a_present_but_invalid_deadline_fails_closed(value):
    from fastmcp.server.auth.oauth_proxy.proxy import _absolute_deadline

    with pytest.raises(TokenError) as refused:
        _absolute_deadline({"expires_in": 300, "expires_at": value})
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
    assert ClientCode.model_validate(stored).refresh_token_expires_at is None


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


async def test_redeeming_after_the_access_deadline_is_refused(proxy, store):
    client = await register(proxy)
    with pytest.raises(TokenError) as refused:
        await redeem(
            proxy,
            client,
            {
                "access_token": "valid-upstream",
                "expires_in": 300,
                "expires_at": int(time.time()) - 1,
                "token_type": "Bearer",
            },
        )
    assert refused.value.error == "invalid_grant"
    assert await published(store) == {
        "mcp-upstream-tokens": [],
        "mcp-jti-mappings": [],
        "mcp-refresh-tokens": [],
    }


# NaN and infinity do not survive the ClientCode storage round trip (they
# become null); the callback refuses them on receipt instead (tested below).
@pytest.mark.parametrize("value", [True, "1000"])
async def test_redeeming_with_an_invalid_access_deadline_is_refused(
    proxy, store, value
):
    client = await register(proxy)
    with pytest.raises(TokenError) as refused:
        await redeem(
            proxy,
            client,
            {
                "access_token": "valid-upstream",
                "expires_in": 300,
                "expires_at": value,
                "token_type": "Bearer",
            },
        )
    assert refused.value.error == "invalid_grant"
    assert await published(store) == {
        "mcp-upstream-tokens": [],
        "mcp-jti-mappings": [],
        "mcp-refresh-tokens": [],
    }


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
                "expires_at": int(time.time()) + 300,
                "refresh_expires_in": 3_600,
                "token_type": "Bearer",
            },
            refresh_token_expires_at=time.time() - 1,
        )
    assert refused.value.error == "invalid_grant"
    assert await published(store) == {
        "mcp-upstream-tokens": [],
        "mcp-jti-mappings": [],
        "mcp-refresh-tokens": [],
    }


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
    tokens = {
        "access_token": "valid-upstream",
        "expires_in": 300,
        "expires_at": deadline,
        "token_type": "Bearer",
    }
    if not accepted:
        with pytest.raises(TokenError):
            await redeem(proxy, client, tokens)
        return
    result = await redeem(proxy, client, tokens)
    assert result.expires_in == 1
    assert proxy.jwt_issuer.verify_token(result.access_token)["exp"] == deadline


async def test_without_an_absolute_deadline_redemption_is_unchanged(proxy):
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
# The callback records the absolute refresh deadline
# --------------------------------------------------------------------------


async def callback(proxy: OAuthProxy, token_response: dict):
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


async def test_the_callback_records_an_absolute_refresh_deadline(proxy):
    before = time.time()
    response = await callback(
        proxy,
        {
            "access_token": "valid-upstream",
            "refresh_token": "upstream-refresh",
            "expires_in": 300,
            "expires_at": int(time.time()) + 300,
            "refresh_expires_in": 3_600,
            "token_type": "Bearer",
        },
    )
    after = time.time()

    assert response.status_code == 302
    code = parse_qs(urlparse(response.headers["location"]).query)["code"][0]
    stored = await proxy._code_store.get(key=code)
    assert stored is not None
    assert stored.refresh_token_expires_at is not None
    assert before + 3_600 <= stored.refresh_token_expires_at <= after + 3_600


@pytest.mark.parametrize("value", [math.nan, math.inf, "soon", True])
async def test_the_callback_refuses_an_invalid_access_deadline(proxy, store, value):
    response = await callback(
        proxy,
        {
            "access_token": "valid-upstream",
            "expires_in": 300,
            "expires_at": value,
            "token_type": "Bearer",
        },
    )
    assert response.status_code == 500
    assert await store.keys("mcp-authorization-codes") == []


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
            "expires_at": int(time.time()) + 300,
            "refresh_expires_in": 3_600,
            "token_type": "Bearer",
        },
        refresh_token_expires_at=refresh_deadline or time.time() + 3_600,
    )
    return client, result


def refresh_response(expires_at: object, *, rotate: bool = True) -> dict:
    response: dict = {
        "access_token": "refreshed-upstream",
        "token_type": "Bearer",
        "expires_in": 300,
        "expires_at": expires_at,
        "scope": "read",
    }
    if rotate:
        response["refresh_token"] = "upstream-refresh-2"
        response["refresh_expires_in"] = 3_600
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


async def test_a_downstream_refresh_keeps_the_recorded_refresh_deadline(proxy):
    refresh_deadline = time.time() + 1_800.5
    client, result = await signed_in(proxy, refresh_deadline=refresh_deadline)
    deadline = int(time.time()) + 300
    refreshed = await refresh(
        proxy, client, result, refresh_response(deadline, rotate=False)
    )

    rotated = proxy.jwt_issuer.verify_token(
        refreshed.refresh_token, expected_token_use="refresh"
    )
    assert rotated["exp"] == math.floor(refresh_deadline)
    _, token_set = await stored_set(proxy, refreshed.access_token)
    assert token_set.refresh_token_expires_at == refresh_deadline


@pytest.mark.parametrize("expires_at", ["past", True, math.nan, "1000"])
async def test_a_downstream_refresh_with_a_bad_deadline_publishes_nothing(
    proxy, store, expires_at
):
    client, result = await signed_in(proxy)
    before = await everything(store)
    if expires_at == "past":
        expires_at = int(time.time()) - 1
    with pytest.raises(TokenError) as refused:
        await refresh(proxy, client, result, refresh_response(expires_at))
    assert refused.value.error == "invalid_grant"
    assert await everything(store) == before


async def test_a_downstream_refresh_past_the_recorded_refresh_deadline_is_refused(
    proxy, store
):
    client, result = await signed_in(proxy)
    # The upstream still answers, but the refresh deadline recorded for this
    # session has passed: no refresh token with a fabricated lifetime is issued.
    _, token_set = await stored_set(proxy, result.access_token)
    token_set.refresh_token_expires_at = time.time() - 1
    await proxy._upstream_token_store.put(
        key=token_set.upstream_token_id, value=token_set, ttl=3_600
    )
    before = await everything(store)
    response = refresh_response(int(time.time()) + 300, rotate=False)
    with pytest.raises(TokenError) as refused:
        await refresh(proxy, client, result, response)
    assert refused.value.error == "invalid_grant"
    assert await everything(store) == before


# --------------------------------------------------------------------------
# Transparent refresh
# --------------------------------------------------------------------------


async def expired_session(proxy: OAuthProxy) -> str:
    await proxy._upstream_token_store.put(
        key="upstream-id",
        value=UpstreamTokenSet(
            upstream_token_id="upstream-id",
            access_token="expired-upstream",
            refresh_token="upstream-refresh",
            refresh_token_expires_at=time.time() + 3_600,
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


async def test_a_transparent_refresh_ends_at_the_response_deadline(proxy, store):
    token = await expired_session(proxy)
    before = await proxy._upstream_token_store.get(key="upstream-id")
    assert before is not None
    deadline = int(time.time()) + 300 - DELAY
    upstream = AsyncMock()
    upstream.refresh_token = AsyncMock(
        return_value=refresh_response(deadline, rotate=False)
    )
    with patch.object(proxy, "_create_upstream_oauth_client", return_value=upstream):
        validated = await proxy.load_access_token(token)

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


@pytest.mark.parametrize("expires_at", ["past", True, math.inf])
async def test_a_transparent_refresh_with_a_bad_deadline_stores_nothing(
    proxy, expires_at
):
    token = await expired_session(proxy)
    before = await proxy._upstream_token_store.get(key="upstream-id")
    if expires_at == "past":
        expires_at = int(time.time()) - 1
    upstream = AsyncMock()
    upstream.refresh_token = AsyncMock(return_value=refresh_response(expires_at))
    with patch.object(proxy, "_create_upstream_oauth_client", return_value=upstream):
        assert await proxy.load_access_token(token) is None
    assert await proxy._upstream_token_store.get(key="upstream-id") == before
