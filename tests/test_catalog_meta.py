"""Catalog metadata: content-derived course levels and deterministic seeded
ratings. Levels must follow title keywords (never random); ratings must be
stable per product id and within the 4.5-4.9 band."""
import pytest

from tests.conftest import auth

from app.services.catalog_meta import infer_level, seed_rating


@pytest.mark.parametrize("title,expected", [
    ("Agentic AI Fundamentals", "Beginner"),
    ("Introduction to Baking", "Beginner"),
    ("Python Programming from Scratch", "Beginner"),
    ("Kubernetes for Beginners", "Beginner"),
    ("UI/UX Design Fundamentals", "Beginner"),
    ("AWS Cloud Practitioner", "Beginner"),
    ("Production RAG at Scale", "Advanced"),
    ("Advanced Python Development", "Advanced"),
    ("Artisan Bread Masterclass", "Advanced"),
    ("Fine-Tuning Language Models", "Advanced"),
    ("Clean Code and Software Architecture", "Advanced"),
    ("Prompt Engineering for Enterprises", "Advanced"),
    ("FastAPI Backend Development", "Intermediate"),
    ("Streaming Data with Kafka", "Intermediate"),
    ("Machine Learning with Scikit-Learn", "Intermediate"),
])
def test_infer_level_from_title(title, expected):
    assert infer_level(title) == expected


def test_infer_level_is_deterministic():
    assert infer_level("Production RAG at Scale") == infer_level("production rag at scale")


def test_seed_rating_is_stable_and_in_band():
    r1, c1 = seed_rating("abc-123")
    r2, c2 = seed_rating("abc-123")
    assert (r1, c1) == (r2, c2)
    assert 4.5 <= r1 <= 4.9
    assert 40 <= c1 <= 1200


def test_seed_rating_differs_across_products():
    r1, _ = seed_rating("prod-one")
    r2, _ = seed_rating("prod-two")
    assert r1 != r2


async def test_create_product_gets_level_and_rating(client, admin_token):
    resp = await client.post("/api/products", json={
        "title": "Advanced Agent Systems",
        "description": "Expert-level agent orchestration",
        "category": "AI",
        "price": 89.99,
    }, headers=auth(admin_token))
    assert resp.status_code == 201
    body = resp.json()
    assert body["level"] == "Advanced"
    assert 4.5 <= body["rating"] <= 4.9
    assert body["rating_count"] and body["rating_count"] > 0

    # Level follows the title: renaming updates it, rating stays stable.
    patch = await client.patch(f"/api/products/{body['id']}", json={"title": "Agent Fundamentals"},
                               headers=auth(admin_token))
    patched = patch.json()
    assert patched["level"] == "Beginner"
    assert patched["rating"] == body["rating"]
