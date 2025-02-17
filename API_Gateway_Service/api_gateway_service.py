from fastapi import FastAPI, Request, Response
import httpx

app = FastAPI()

# Define routes for different microservices
MICROSERVICES = {
    "ui_service": "http://localhost:5000",
    "authentication_service": "http://localhost:5001",
    "load_balancer": "http://localhost:5002",
    "edge_inference_service": "http://localhost:5003",
    "fine_tuning_service": "http://localhost:5004",
    "inference_service": "http://localhost:5005",
    "database_service": "http://localhost:5006"
    
}

@app.api_route("/{service}/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def gateway(service: str, path: str, request: Request):
    if service not in MICROSERVICES:
        return {"error": "Service not found"}

    # Construct the target URL
    target_url = f"{MICROSERVICES[service]}/{path}"
    print(f"Forwarding request to: {target_url}")
    
    async with httpx.AsyncClient(follow_redirects=True) as client:
        # Forward the request to the respective microservice
        response = await client.request(
            method=request.method,
            url=target_url,
            headers=dict(request.headers),
            params=dict(request.query_params),
            content=await request.body()
        )
        return Response(
        content=response.content,
        status_code=response.status_code,
        media_type=response.headers.get("content-type"))

# Run the gateway using: uvicorn api_gateway_service:app --host 0.0.0.0 --port 8000
