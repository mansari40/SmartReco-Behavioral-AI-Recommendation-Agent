from datetime import datetime

from pydantic import BaseModel, ConfigDict

from app.models.product import SyncStatus


class ProductCreate(BaseModel):
    title: str
    description: str
    category: str
    price: float


class ProductUpdate(BaseModel):
    title: str | None = None
    description: str | None = None
    category: str | None = None
    price: float | None = None


class ProductOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    title: str
    description: str
    category: str
    price: float
    sync_status: SyncStatus
    created_at: datetime
    updated_at: datetime