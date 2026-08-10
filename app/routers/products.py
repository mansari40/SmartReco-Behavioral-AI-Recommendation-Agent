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
from app.services import llm_client, vector_store

router = APIRouter(prefix="/api/products", tags=["products"])


async def _sync_to_vector_store(product: Product) -> tuple[SyncStatus, str | None]:
    try:
        embedding = await llm_client.get_embedding(product.to_embedding_text())
        await vector_store.upsert_product(
            vector_id=product.vector_id,
            embedding=embedding,
            document=product.to_embedding_text(),
            metadata={"category": product.category, "price": product.price, "sql_id": product.id},
        )
        return SyncStatus.SYNCED, None
    except Exception as exc:  # noqa: BLE001 — deliberately broad: any failure must surface, not vanish
        return SyncStatus.FAILED, str(exc)


@router.post("", response_model=ProductOut, status_code=status.HTTP_201_CREATED)
async def create_product(
    payload: ProductCreate, db: AsyncSession = Depends(get_db), _admin=Depends(require_admin)
):
    product = Product(**payload.model_dump(), vector_id=str(uuid.uuid4()))
    db.add(product)
    await db.flush()  # get product.id before the vector-store call, without committing yet

    product.sync_status, product.sync_error = await _sync_to_vector_store(product)

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

    for field, value in payload.model_dump(exclude_unset=True).items():
        setattr(product, field, value)

    # Re-embed on any edit — the text representation may have changed.
    product.sync_status, product.sync_error = await _sync_to_vector_store(product)

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
    except Exception:  # noqa: BLE001
        # Don't block the SQL delete on a vector-store hiccup, but this is
        # exactly the kind of drift sync_status is meant to catch elsewhere —
        # logged here so it's visible, not silent.
        import logging

        logging.getLogger(__name__).exception("Vector store delete failed for %s", product.vector_id)

    await db.delete(product)
    await db.commit()