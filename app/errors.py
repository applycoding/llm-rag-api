class AppError(Exception):
    """Error that maps to an HTTP response."""

    def __init__(self, message: str, *, status_code: int = 500) -> None:
        super().__init__(message)
        self.status_code = status_code


class IngestError(AppError):
    def __init__(self, message: str, *, status_code: int = 400) -> None:
        super().__init__(message, status_code=status_code)


class NotFoundError(AppError):
    def __init__(self, message: str) -> None:
        super().__init__(message, status_code=404)


class ProviderConfigError(AppError):
    def __init__(self, message: str) -> None:
        super().__init__(message, status_code=503)


class ProviderError(AppError):
    def __init__(self, message: str) -> None:
        super().__init__(message, status_code=502)


class StoreError(AppError):
    def __init__(self, message: str) -> None:
        super().__init__(message, status_code=500)
