from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Use absolute imports instead of relative imports
from routes.placement import router as placement_router
from routes.retrieval import router as retrieval_router
from routes.waste import router as waste_router
from database import router as database_router
from routes.simulation import router as simulation_router
from routes.log import router as log_router

def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.
    
    Returns:
        FastAPI: Configured FastAPI application instance
    """
    # Initialize the FastAPI application
    app = FastAPI(
        title="Cargo Management System",
        description="Advanced cargo placement and management system",
        version="0.1.0"
    )
    

    

    # Add CORS middleware to allow cross-origin requests
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Allow all origins in development
        allow_credentials=True,
        allow_methods=["*"],  # Allow all methods
        allow_headers=["*"],  # Allow all headers
    )

    # Include routers
    app.include_router(placement_router, prefix="/api/placement", tags=["placement"])
    app.include_router(database_router, prefix="/api", tags=["import"])
    app.include_router(retrieval_router, prefix="/api", tags=["retrieval"])
    app.include_router(waste_router, prefix="/api/waste", tags=["waste"])
    app.include_router(simulation_router, prefix="/api", tags=["simulation"])
    app.include_router(log_router, prefix="/api", tags=["log"])
    return app

# Create the application instance
app = create_app()


# Optional: Add root endpoint for health check
@app.get("/", tags=["Root"])
async def root():
    """
    Health check and basic application information endpoint.
    
    Returns:
        dict: Basic application status information
    """
    return {
        "status": "healthy",
        "application": "Cargo Management System",
        "version": "0.1.0",
        "description": "Cargo placement and management API"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)