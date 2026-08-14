"""
Server-rendered page routes (Jinja2), as distinct from the JSON API
routers. Kept separate so the API surface (app/routers/products.py etc.)
stays purely programmatic, and page routes can freely mix in template
rendering, cookies, redirects, etc. without cluttering the API layer.
"""
from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.templating import Jinja2Templates
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.models.product import Product
from app.routers.deps import require_admin
from app.services import trigger_service

router = APIRouter(tags=["pages"])
templates = Jinja2Templates(directory="app/templates")

_COVER_VARIANTS = 6


def cover_index(category: str) -> int:
    return sum(ord(c) for c in category) % _COVER_VARIANTS


@router.get("/")
async def catalog_page(request: Request, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Product).order_by(Product.created_at.desc()))
    products = result.scalars().all()
    categories = sorted({p.category for p in products})
    return templates.TemplateResponse(
        "catalog.html",
        {
            "request": request,
            "products": products,
            "categories": categories,
            "cover_index": cover_index,
        },
    )


@router.get("/products/{product_id}")
async def product_detail_page(request: Request, product_id: str, db: AsyncSession = Depends(get_db)):
    product = await db.get(Product, product_id)
    if not product:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Product not found")

    related_result = await db.execute(
        select(Product).where(Product.category == product.category, Product.id != product.id).limit(3)
    )
    related = related_result.scalars().all()

    return templates.TemplateResponse(
        "product_detail.html",
        {
            "request": request,
            "product": product,
            "related": related,
            "cover_index": cover_index,
        },
    )


@router.get("/login")
async def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})


@router.get("/how-it-works")
async def how_it_works_page(request: Request):
    return templates.TemplateResponse("how_it_works.html", {"request": request})


@router.get("/password-reset")
async def password_reset_request_page(request: Request):
    return templates.TemplateResponse("password_reset_request.html", {"request": request})


@router.get("/password-reset/confirm")
async def password_reset_confirm_page(request: Request):
    return templates.TemplateResponse("password_reset_confirm.html", {"request": request})


@router.get("/register")
async def register_page(request: Request):
    return templates.TemplateResponse("register.html", {"request": request})


@router.get("/recommendations")
async def recommendations_page(request: Request):
    return templates.TemplateResponse(
        "recommendations.html",
        {
            "request": request,
            "event_threshold": trigger_service.EVENT_THRESHOLD,
            "min_score_threshold": trigger_service.MIN_SCORE_THRESHOLD,
            "cooldown_seconds": trigger_service.MIN_SECONDS_BETWEEN_RUNS,
        },
    )


@router.get("/admin")
async def admin_page(request: Request, _admin=Depends(require_admin)):
    return templates.TemplateResponse("admin.html", {"request": request})


@router.get("/cart")
async def cart_page(request: Request):
    return templates.TemplateResponse("cart.html", {"request": request})


@router.get("/console")
async def console_page(request: Request, _admin=Depends(require_admin)):
    return templates.TemplateResponse("console.html", {"request": request})