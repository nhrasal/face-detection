"""Convenient local launcher for the FastAPI application."""

import uvicorn


def main() -> None:
    """Run the development server with automatic reload enabled."""
    uvicorn.run(
        "app.main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )


if __name__ == "__main__":
    main()
