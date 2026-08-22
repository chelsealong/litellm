"""
LiteLLM x SendGrid email integration.

Docs: https://docs.sendgrid.com/api-reference/mail-send/mail-send
"""

import base64
import os
from typing import List, Optional

from typing_extensions import ReadOnly, TypedDict

from litellm._logging import verbose_logger
from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,
    httpxSpecialProvider,
)
from litellm.types.integrations.slack_alerting import LITELLM_LOGO_URL

from .base_email import BaseEmailLogger


SENDGRID_API_ENDPOINT = "https://api.sendgrid.com/v3/mail/send"
LOGO_CONTENT_ID = "logo"


class SendGridInlineAttachment(TypedDict):
    content: ReadOnly[str]
    type: ReadOnly[str]
    filename: ReadOnly[str]
    disposition: ReadOnly[str]
    content_id: ReadOnly[str]


class SendGridEmailLogger(BaseEmailLogger):
    """
    Send emails using SendGrid's Mail Send API.

    Required env vars:
    - SENDGRID_API_KEY
    """

    def __init__(self, internal_usage_cache=None, **kwargs):
        super().__init__(internal_usage_cache=internal_usage_cache, **kwargs)
        self.async_httpx_client = get_async_httpx_client(
            llm_provider=httpxSpecialProvider.LoggingCallback
        )
        self.sendgrid_api_key = os.getenv("SENDGRID_API_KEY")
        self.sendgrid_sender_email = os.getenv("SENDGRID_SENDER_EMAIL")
        self.logo_attachment = self._build_logo_attachment()
        verbose_logger.debug("SendGrid Email Logger initialized.")

    def _build_logo_attachment(self) -> Optional[SendGridInlineAttachment]:
        logo_path = os.getenv("EMAIL_LOGO_PATH")
        if not logo_path:
            return None

        from litellm.proxy.proxy_server import premium_user

        if premium_user is not True:
            verbose_logger.warning(
                "EMAIL_LOGO_PATH is a premium feature. Ignoring it for non-premium users."
            )
            return None
        try:
            with open(logo_path, "rb") as logo_file:
                encoded_logo = base64.b64encode(logo_file.read()).decode("utf-8")
        except OSError as e:
            verbose_logger.warning(
                f"Could not read EMAIL_LOGO_PATH={logo_path}, falling back to external logo URL: {e}"
            )
            return None
        return SendGridInlineAttachment(
            content=encoded_logo,
            type="image/png",
            filename=os.path.basename(logo_path),
            disposition="inline",
            content_id=LOGO_CONTENT_ID,
        )

    async def send_email(
        self,
        from_email: str,
        to_email: List[str],
        subject: str,
        html_body: str,
    ):
        """
        Send an email via SendGrid.
        """
        if not self.sendgrid_api_key:
            raise ValueError("SENDGRID_API_KEY is not set")

        sender_email = self.sendgrid_sender_email or from_email
        verbose_logger.debug(
            f"Sending email via SendGrid from {sender_email} to {to_email} with subject {subject}"
        )

        if self.logo_attachment is not None:
            logo_url = os.getenv("EMAIL_LOGO_URL", LITELLM_LOGO_URL)
            html_body = html_body.replace(logo_url, f"cid:{LOGO_CONTENT_ID}")

        payload = {
            "from": {"email": sender_email},
            "personalizations": [
                {
                    "to": [{"email": email} for email in to_email],
                    "subject": subject,
                }
            ],
            "content": [
                {
                    "type": "text/html",
                    "value": html_body,
                }
            ],
        }

        if self.logo_attachment is not None:
            payload["attachments"] = [self.logo_attachment]

        response = await self.async_httpx_client.post(
            url=SENDGRID_API_ENDPOINT,
            json=payload,
            headers={"Authorization": f"Bearer {self.sendgrid_api_key}"},
        )

        verbose_logger.debug(
            f"SendGrid response status={response.status_code}, body={response.text}"
        )
        return
