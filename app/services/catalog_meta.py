"""
Deterministic catalog metadata: course level (Beginner/Intermediate/Advanced)
and seeded ratings (4.5-4.9 with a plausible review count).

Both are derived from the product itself (title/content keywords and a stable
hash of the product id) so they never change between renders and are identical
for the same product in every environment. Levels are content-based — never
random — while ratings are seeded per product so the store feels lived-in
without fabricating per-review data.
"""
import hashlib

BEGINNER_KEYWORDS = [
    "fundamentals", "introduction", "from scratch", "basics", "101",
    "beginners", "essentials", "starter", "core", "getting started",
    "practitioner",
]

ADVANCED_KEYWORDS = [
    "advanced", "production", "at scale", "masterclass", "master",
    "expert", "deep", "fine-tuning", "architecture", "enterprises",
    "risk", "professional",
]


def infer_level(title: str) -> str:
    """Content-based level: explicit keywords in the title decide the level,
    everything else is Intermediate. Deterministic — same title, same level."""
    lowered = (title or "").lower()
    if any(k in lowered for k in ADVANCED_KEYWORDS):
        return "Advanced"
    if any(k in lowered for k in BEGINNER_KEYWORDS):
        return "Beginner"
    return "Intermediate"


def seed_rating(product_id: str) -> tuple[float, int]:
    """Deterministic per-product rating in [4.50, 4.90] plus a plausible
    review count (40-1200). Same product id always yields the same numbers."""
    digest = hashlib.sha256((product_id or "").encode("utf-8")).hexdigest()
    step = int(digest[:8], 16) % 41                    # 0..40 -> 4.50..4.90
    rating = round(4.5 + step / 100, 2)
    count = 40 + (int(digest[8:16], 16) % 1161)        # 40..1200
    return rating, count