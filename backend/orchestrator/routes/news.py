# orchestrator/routes/news.py
from fastapi import APIRouter, HTTPException, Query
import os, httpx

router = APIRouter()

# Point this to your retrieval_agent (where /news lives)
RETRIEVAL_URL = os.getenv("RETRIEVAL_URL", "http://127.0.0.1:8002")

@router.get("/route/news")
async def route_news(limit: int = Query(12, ge=1, le=50)):
    """Proxy: orchestrator → retrieval_agent /news"""
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.get(f"{RETRIEVAL_URL}/news", params={"limit": limit})
        if r.status_code != 200:
            raise HTTPException(status_code=r.status_code, detail=r.text)
        return r.json()   # { status, count, news: [...] }
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"route/news proxy error: {e}")

@router.get("/route/news/sources")
async def route_news_sources():
    """Proxy: orchestrator → retrieval_agent /news/sources"""
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.get(f"{RETRIEVAL_URL}/news/sources")
        if r.status_code != 200:
            raise HTTPException(status_code=r.status_code, detail=r.text)
        return r.json()   # { sources: [...] }
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"route/news/sources proxy error: {e}")
