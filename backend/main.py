import os
import logging
from fastapi import FastAPI
import uvicorn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("gov-transparency-rag")

app = FastAPI(title="gov-transparency-rag-backend")




if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    logger.info(f"Starting app on 0.0.0.0:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")