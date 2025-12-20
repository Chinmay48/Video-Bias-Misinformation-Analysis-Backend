from fastapi import FastAPI
from app.api.routes.analyze_video import router as analyze_router
from app.api.routes.health import router as health_router

app = FastAPI(
    title="Video Bias Detection API",
    version="1.0.0"
)

# Register routes
app.include_router(health_router)
app.include_router(analyze_router)
