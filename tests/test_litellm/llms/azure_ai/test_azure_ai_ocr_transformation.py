from unittest.mock import patch

import pytest

from litellm.llms.azure_ai.ocr.transformation import AzureAIOCRConfig


def test_azure_ai_ocr_validate_environment_with_api_key():
    config = AzureAIOCRConfig()
    headers = config.validate_environment(
        headers={},
        model="mistral-ocr-2503",
        api_key="test-api-key",
        api_base="https://my-endpoint.services.ai.azure.com",
    )
    assert headers["Authorization"] == "Bearer test-api-key"
    assert headers["Content-Type"] == "application/json"


def test_azure_ai_ocr_validate_environment_with_azure_ad_token():
    """
    When no api_key is provided but Azure AD credentials are available, the OCR
    path should fall back to the Entra ID token like azure_ai chat already does.

    Regression test for https://github.com/BerriAI/litellm/issues/37727
    """
    config = AzureAIOCRConfig()
    with (
        patch(
            "litellm.llms.azure_ai.ocr.transformation.get_azure_ad_token",
            return_value="fake-azure-ad-token",
        ),
        patch(
            "litellm.llms.azure_ai.ocr.transformation.get_secret_str",
            return_value=None,
        ),
    ):
        headers = config.validate_environment(
            headers={},
            model="mistral-document-ai-2512",
            api_key=None,
            api_base="https://my-endpoint.services.ai.azure.com",
            litellm_params={},
        )
    assert headers["Authorization"] == "Bearer fake-azure-ad-token"
    assert headers["Content-Type"] == "application/json"


def test_azure_ai_ocr_validate_environment_raises_without_any_credential():
    config = AzureAIOCRConfig()
    with (
        patch(
            "litellm.llms.azure_ai.ocr.transformation.get_azure_ad_token",
            return_value=None,
        ),
        patch(
            "litellm.llms.azure_ai.ocr.transformation.get_secret_str",
            return_value=None,
        ),
        pytest.raises(ValueError, match="Missing Azure AI API Key"),
    ):
        config.validate_environment(
            headers={},
            model="mistral-ocr-2503",
            api_key=None,
            api_base="https://my-endpoint.services.ai.azure.com",
            litellm_params={},
        )
