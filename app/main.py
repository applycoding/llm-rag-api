"""Uvicorn entrypoint. `app` is created at import so the server sees process env."""

import logging

import uvicorn

from app.api import create_app

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

app = create_app()


def main() -> None:
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
