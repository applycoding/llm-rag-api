from pydantic import SecretStr

from app.errors import ProviderConfigError


def require_api_key(value: SecretStr | None, provider: str) -> str:
    if value is None or not value.get_secret_value().strip():
        raise ProviderConfigError(f"{provider} is selected but no API key is configured.")
    return value.get_secret_value()
