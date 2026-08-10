"""
Server-rendered page routes (Jinja2), as distinct from the JSON API
routers. Kept separate so the API surface (app/routers/products.py etc.)
stays purely programmatic, and page routes can freely mix in template
rendering, cookies, redirects, etc. without cluttering the API layer.
"""
from fastapi import APIRouter, Request
from fastapi.templating import Jinja2Templates
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi import Depends

from app.db.session import get_db
from app.models.product import Product

router = APIRouter(tags=["pages"])
templates = Jinja2Templates(directory="app/templates")


@router.get("/")
async def catalog_page(request: Request, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Product))
    products = result.scalars().all()
    return templates.TemplateResponse("catalog.html", {"request": request, "products": products})

@router.get("/login")
async def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@router.get("/recommendations")
async def recommendations_page(request: Request):
    return templates.TemplateResponse("recommendations.html", {"request": request})