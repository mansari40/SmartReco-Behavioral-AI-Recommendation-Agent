"""
Admin product CRUD with dual-write to SQL + the vector store. This is one
of the explicitly graded hackathon requirements (kept even though we're no
longer submitting — it's just good design): sync_status/sync_error is not
decorative, it's how you can tell at a glance whether the two stores ever
drifted apart.
"""
import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.models.product import Product, SyncStatus
from app.routers.deps import require_admin
from app.schemas.product import ProductCreate, ProductOut, ProductUpdate
from app.services import vector_store
from app.services.catalog_meta import infer_level, seed_rating
from app.services.seeder import sync_product_to_vector_store

router = APIRouter(prefix="/api/products", tags=["products"])


def _apply_store_metadata(product: Product, title_changed: bool = True) -> None:
    """Level is content-derived from the title; rating is seeded once per
    product id and kept stable afterwards."""
    if title_changed or not product.level:
        product.level = infer_level(product.title)
    if product.rating is None:
        rating, count = seed_rating(product.id)
        product.rating, product.rating_count = rating, count


@router.post("", response_model=ProductOut, status_code=status.HTTP_201_CREATED)
async def create_product(
    payload: ProductCreate, db: AsyncSession = Depends(get_db), _admin=Depends(require_admin)
):
    clash = (
        await db.execute(select(Product).where(Product.title == payload.title))
    ).scalar_one_or_none()
    if clash:
        raise HTTPException(status.HTTP_409_CONFLICT, "A product with this title already exists")

    product = Product(**payload.model_dump(), vector_id=str(uuid.uuid4()), sync_status=SyncStatus.PENDING)
    db.add(product)
    await db.flush()  # get product.id before the vector-store call, without committing yet
    _apply_store_metadata(product)

    product.sync_status, product.sync_error = await sync_product_to_vector_store(product)

    await db.commit()
    await db.refresh(product)
    return product


@router.get("", response_model=list[ProductOut])
async def list_products(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Product))
    return result.scalars().all()


@router.get("/{product_id}", response_model=ProductOut)
async def get_product(product_id: str, db: AsyncSession = Depends(get_db)):
    product = await db.get(Product, product_id)
    if not product:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Product not found")
    return product


@router.patch("/{product_id}", response_model=ProductOut)
async def update_product(
    product_id: str,
    payload: ProductUpdate,
    db: AsyncSession = Depends(get_db),
    _admin=Depends(require_admin),
):
    product = await db.get(Product, product_id)
    if not product:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Product not found")

    title_changed = "title" in payload.model_dump(exclude_unset=True)
    if title_changed:
        clash = (
            await db.execute(
                select(Product).where(Product.title == payload.title, Product.id != product_id)
            )
        ).scalar_one_or_none()
        if clash:
            raise HTTPException(status.HTTP_409_CONFLICT, "A product with this title already exists")
    for field, value in payload.model_dump(exclude_unset=True).items():
        setattr(product, field, value)
    _apply_store_metadata(product, title_changed=title_changed)
    product.sync_status = SyncStatus.PENDING
    product.sync_error = None

    # Re-embed on any edit — the text representation may have changed.
    product.sync_status, product.sync_error = await sync_product_to_vector_store(product)

    await db.commit()
    await db.refresh(product)
    return product


@router.delete("/{product_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_product(
    product_id: str, db: AsyncSession = Depends(get_db), _admin=Depends(require_admin)
):
    product = await db.get(Product, product_id)
    if not product:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Product not found")

    try:
        await vector_store.delete_product(product.vector_id)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            f"Failed to remove product from the vector store: {str(exc)}",
        ) from exc

    await db.delete(product)
    await db.commit()