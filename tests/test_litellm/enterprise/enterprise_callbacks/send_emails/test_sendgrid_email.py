import base64
import os
import sys
import unittest.mock as mock
from unittest.mock import patch

import pytest
from httpx import Response

sys.path.insert(0, os.path.abspath("../../.."))

import litellm
from litellm_enterprise.enterprise_callbacks.send_emails.sendgrid_email import (
    SendGridEmailLogger,
)


@pytest.fixture(autouse=True)
def clear_client_cache():
    """
    Clear the HTTP client cache before each test to ensure mocks are used.
    This prevents cached real clients from being reused across tests.
    """
    cache = getattr(litellm, "in_memory_llm_clients_cache", None)
    if cache is not None:
        cache.flush_cache()
    yield
    if cache is not None:
        cache.flush_cache()


@pytest.fixture
def mock_env_vars():
    # Store original values
    original_api_key = os.environ.get("SENDGRID_API_KEY")
    original_sender_email = os.environ.get("SENDGRID_SENDER_EMAIL")
    
    # Set test API key and remove SENDGRID_SENDER_EMAIL to ensure isolation
    os.environ["SENDGRID_API_KEY"] = "test_api_key"
    if "SENDGRID_SENDER_EMAIL" in os.environ:
        del os.environ["SENDGRID_SENDER_EMAIL"]
    
    try:
        yield
    finally:
        # Restore original values
        if original_api_key is not None:
            os.environ["SENDGRID_API_KEY"] = original_api_key
        elif "SENDGRID_API_KEY" in os.environ:
            del os.environ["SENDGRID_API_KEY"]
        
        if original_sender_email is not None:
            os.environ["SENDGRID_SENDER_EMAIL"] = original_sender_email


@pytest.fixture
def mock_async_client():
    """
    Create a mock async httpx client that can be injected directly
    into the logger instance, bypassing any caching issues.
    """
    mock_response = mock.Mock(spec=Response)
    mock_response.status_code = 202
    mock_response.text = "accepted"
    mock_response.raise_for_status.return_value = None

    mock_client = mock.AsyncMock()
    mock_client.post.return_value = mock_response
    return mock_client


@pytest.mark.asyncio
async def test_send_email_success(mock_env_vars, mock_async_client):
    logger = SendGridEmailLogger()
    # Directly replace the httpx client to ensure the mock is used
    # This bypasses any caching or initialization timing issues
    logger.async_httpx_client = mock_async_client

    from_email = "test@example.com"
    to_email = ["recipient@example.com"]
    subject = "Test Subject"
    html_body = "<p>Test email body</p>"

    await logger.send_email(
        from_email=from_email, to_email=to_email, subject=subject, html_body=html_body
    )

    mock_async_client.post.assert_called_once()
    call_args = mock_async_client.post.call_args
    assert call_args[1]["url"] == "https://api.sendgrid.com/v3/mail/send"

    payload = call_args[1]["json"]
    assert payload["from"] == {"email": from_email}
    assert payload["personalizations"][0]["to"] == [{"email": to_email[0]}]
    assert payload["personalizations"][0]["subject"] == subject
    assert payload["content"][0]["type"] == "text/html"
    assert payload["content"][0]["value"] == html_body

    assert call_args[1]["headers"] == {"Authorization": "Bearer test_api_key"}


@pytest.mark.asyncio
async def test_send_email_missing_api_key():
    original_key = os.environ.pop("SENDGRID_API_KEY", None)

    try:
        logger = SendGridEmailLogger()

        with pytest.raises(ValueError, match='SENDGRID_API_KEY is not set'):
            await logger.send_email(
                from_email="test@example.com",
                to_email=["recipient@example.com"],
                subject="Test Subject",
                html_body="<p>Test email body</p>",
            )
    finally:
        if original_key is not None:
            os.environ["SENDGRID_API_KEY"] = original_key


@pytest.mark.asyncio
async def test_send_email_inline_logo_attachment(
    mock_env_vars, mock_async_client, tmp_path
):
    logo_path = tmp_path / "logo.png"
    logo_path.write_bytes(b"fake-png-bytes")
    os.environ["EMAIL_LOGO_PATH"] = str(logo_path)

    try:
        with patch("litellm.proxy.proxy_server.premium_user", True):
            logger = SendGridEmailLogger()
        logger.async_httpx_client = mock_async_client

        html_body = (
            '<img src="https://litellm-listing.s3.amazonaws.com/litellm_logo.png" '
            'alt="LiteLLM Logo">'
        )

        await logger.send_email(
            from_email="test@example.com",
            to_email=["recipient@example.com"],
            subject="Test Subject",
            html_body=html_body,
        )

        payload = mock_async_client.post.call_args[1]["json"]

        assert "cid:logo" in payload["content"][0]["value"]
        assert (
            "https://litellm-listing.s3.amazonaws.com/litellm_logo.png"
            not in payload["content"][0]["value"]
        )

        assert len(payload["attachments"]) == 1
        attachment = payload["attachments"][0]
        assert attachment["content_id"] == "logo"
        assert attachment["disposition"] == "inline"
        assert base64.b64decode(attachment["content"]) == b"fake-png-bytes"
    finally:
        del os.environ["EMAIL_LOGO_PATH"]


@pytest.mark.asyncio
async def test_send_email_inline_logo_attachment_non_premium_falls_back_to_external_url(
    mock_env_vars, mock_async_client, tmp_path
):
    logo_path = tmp_path / "logo.png"
    logo_path.write_bytes(b"fake-png-bytes")
    os.environ["EMAIL_LOGO_PATH"] = str(logo_path)

    try:
        with patch("litellm.proxy.proxy_server.premium_user", False):
            logger = SendGridEmailLogger()
        logger.async_httpx_client = mock_async_client

        html_body = (
            '<img src="https://litellm-listing.s3.amazonaws.com/litellm_logo.png" '
            'alt="LiteLLM Logo">'
        )

        await logger.send_email(
            from_email="test@example.com",
            to_email=["recipient@example.com"],
            subject="Test Subject",
            html_body=html_body,
        )

        payload = mock_async_client.post.call_args[1]["json"]

        assert (
            "https://litellm-listing.s3.amazonaws.com/litellm_logo.png"
            in payload["content"][0]["value"]
        )
        assert "cid:logo" not in payload["content"][0]["value"]
        assert "attachments" not in payload
    finally:
        del os.environ["EMAIL_LOGO_PATH"]


@pytest.mark.asyncio
async def test_send_email_no_logo_path_uses_external_url(
    mock_env_vars, mock_async_client
):
    logger = SendGridEmailLogger()
    logger.async_httpx_client = mock_async_client

    html_body = (
        '<img src="https://litellm-listing.s3.amazonaws.com/litellm_logo.png" '
        'alt="LiteLLM Logo">'
    )

    await logger.send_email(
        from_email="test@example.com",
        to_email=["recipient@example.com"],
        subject="Test Subject",
        html_body=html_body,
    )

    payload = mock_async_client.post.call_args[1]["json"]
    assert (
        "https://litellm-listing.s3.amazonaws.com/litellm_logo.png"
        in payload["content"][0]["value"]
    )
    assert "attachments" not in payload


@pytest.mark.asyncio
async def test_send_email_multiple_recipients(mock_env_vars, mock_async_client):
    logger = SendGridEmailLogger()
    # Directly replace the httpx client to ensure the mock is used
    # This bypasses any caching or initialization timing issues
    logger.async_httpx_client = mock_async_client

    from_email = "test@example.com"
    to_email = ["recipient1@example.com", "recipient2@example.com"]
    subject = "Test Subject"
    html_body = "<p>Test email body</p>"

    await logger.send_email(
        from_email=from_email, to_email=to_email, subject=subject, html_body=html_body
    )

    mock_async_client.post.assert_called_once()
    payload = mock_async_client.post.call_args[1]["json"]

    assert payload["personalizations"][0]["to"] == [
        {"email": "recipient1@example.com"},
        {"email": "recipient2@example.com"},
    ]
