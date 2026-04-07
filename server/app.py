"""Re-export the FastAPI app from the root module and explicitly define main()
to pass the OpenEnv static validator which looks for 'def main()' and '__main__'."""
from app import app


def main():
    """Start the uvicorn server."""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
