"""
One-time, reversible migration: collapse duplicate product rows in the local
database into the canonical 52-course catalog.

    python scripts/dedupe_catalog.py            # dry-run: show the full plan, change nothing
    python scripts/dedupe_catalog.py --apply    # backup -> one-transaction migrate -> verify
    python scripts/dedupe_catalog.py --apply --yes   # non-interactive apply

WHY
---
seed_catalog() used to look courses up by `products.title`, which had no
uniqueness constraint, so repeated seeding inserted duplicate rows for 5
titles (58 rows for 52 courses). Every duplicate row is referenced — by
events.product_id and by the JSON columns of recommendations — so plain
deletes are not safe. This script rewrites those references to a single
canonical row per title, deletes the extras, and enforces title uniqueness
(uq_products_title) so it cannot recur.

RETENTION RULE (deterministic, per canonical title)
1. the row carrying the canonical _stable_vector_id (uuid5 of the title), if any;
2. else the most-referenced row (events.product_id + recommendation JSON refs);
3. else the oldest row (created_at).

SAFETY
- Dry-run by default; --apply is required to write anything.
- --apply snapshots the DB (VACUUM INTO, consistent even in WAL mode) and
  copies the local Chroma persist dir before working, so the migration is
  fully reversible by file restore.
- All SQL changes run in ONE transaction; any failure rolls back cleanly.
- Chroma deletions happen only after the DB commit, are guarded by
  product_exists(), and are non-fatal — a leftover vector is simply dropped
  at retrieval time, never referenced.
- Idempotent: re-running --apply after success is a no-op.

RESTORE (if ever needed): stop the app, replace the DB file with the
snapshot, and restore the Chroma directory from its snapshot copy.
"""
import argparse
import asyncio
import re
import shutil
import sqlite3
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from app.config import settings  # noqa: E402
from scripts.seed_products import CATALOG  # noqa: E402

JSON_COLUMNS = ("products", "reasoning_chain", "alternatives_considered", "behavior_explanation")
PRODUCT_ID_PATTERN = re.compile(r'"product_id"\s*:\s*"([0-9a-fA-F-]{36})"')


def _stable_vector_id(title: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"upulse-course/{title}"))


def db_path() -> Path:
    url = settings.database_url
    if not url.startswith("sqlite"):
        raise SystemExit(
            f"Only sqlite databases can be backed up automatically; found {url!r}. "
            "Take a manual snapshot first, then re-run."
        )
    return Path(url.split("///", 1)[1]).resolve()


def open_ro() -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{db_path()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def reference_counts(conn: sqlite3.Connection) -> dict[str, tuple[int, int]]:
    events = dict(conn.execute("SELECT product_id, COUNT(*) FROM events GROUP BY product_id"))
    recs: dict[str, int] = {}
    for col in JSON_COLUMNS:
        for (raw,) in conn.execute(f"SELECT {col} FROM recommendations WHERE {col} IS NOT NULL"):
            for pid in PRODUCT_ID_PATTERN.findall(raw or ""):
                recs[pid] = recs.get(pid, 0) + 1
    return {pid: (events.get(pid, 0), recs.get(pid, 0)) for pid in set(events) | set(recs)}


def build_mapping(conn: sqlite3.Connection) -> dict[str, dict]:
    rows = {
        r["id"]: dict(r)
        for r in conn.execute(
            "SELECT id, title, vector_id, created_at FROM products ORDER BY created_at ASC"
        )
    }
    by_title: dict[str, list[dict]] = {}
    for r in rows.values():
        by_title.setdefault(r["title"], []).append(r)

    refs = reference_counts(conn)
    catalog_titles = {item["title"] for item in CATALOG}

    def ref_total(pid: str) -> int:
        events_n, recs_n = refs.get(pid, (0, 0))
        return events_n + recs_n

    mapping: dict[str, dict] = {}
    for title, group in sorted(by_title.items()):
        if title not in catalog_titles:
            print(f"[warn] title not in canonical CATALOG: {title!r} — left untouched")
        if len(group) == 1:
            mapping[title] = {"canonical": group[0], "removed": []}
            continue
        stable_hits = [r for r in group if r["vector_id"] == _stable_vector_id(title)]
        if stable_hits:
            canonical = stable_hits[0]
        else:
            canonical = sorted(group, key=lambda r: (-ref_total(r["id"]), r["created_at"]))[0]
        mapping[title] = {
            "canonical": canonical,
            "removed": [r for r in group if r["id"] != canonical["id"]],
        }
    return mapping


def print_plan(mapping: dict[str, dict], conn: sqlite3.Connection) -> None:
    refs = reference_counts(conn)
    total_removed = 0
    for title, info in sorted(mapping.items()):
        if not info["removed"]:
            continue
        canon = info["canonical"]
        ev, rc = refs.get(canon["id"], (0, 0))
        print(f"\n{title!r}")
        print(f"  keep    {canon['id']}  vector={canon['vector_id']}  "
              f"events={ev} recs={rc} created={canon['created_at']}")
        for dup in info["removed"]:
            dev, drc = refs.get(dup["id"], (0, 0))
            total_removed += 1
            print(f"  remove  {dup['id']}  vector={dup['vector_id']}  "
                  f"events={dev} recs={drc} created={dup['created_at']}")
    print(f"\n{total_removed} duplicate rows to remove.")


def backup_db(stamp: str) -> Path:
    src = db_path()
    backup = src.with_name(f"{src.name}.{stamp}.pre-dedupe.db")
    conn = sqlite3.connect(str(src), isolation_level=None)
    try:
        conn.execute(f"VACUUM INTO {backup.as_posix()!r}")
    finally:
        conn.close()
    print(f"DB snapshot: {backup}")
    return backup


def backup_chroma(stamp: str) -> Path | None:
    src = Path(settings.chroma_persist_dir).resolve()
    if not src.is_dir():
        print("[warn] no local Chroma dir to back up — skipped")
        return None
    backup = src.with_name(f"{src.name}.{stamp}.pre-dedupe")
    shutil.copytree(src, backup)
    print(f"Chroma snapshot: {backup}")
    return backup


async def apply_migration(mapping: dict[str, dict]) -> int:
    from sqlalchemy import text
    from app.db.session import AsyncSessionLocal

    dup_to_canon = {
        dup["id"]: info["canonical"]["id"]
        for info in mapping.values() for dup in info["removed"]
    }
    json_changed = 0

    async with AsyncSessionLocal() as db:
        for info in mapping.values():
            for dup in info["removed"]:
                await db.execute(
                    text("UPDATE events SET product_id = :canon WHERE product_id = :dup"),
                    {"canon": info["canonical"]["id"], "dup": dup["id"]},
                )
                metas = (
                    await db.execute(
                        text("SELECT id, event_metadata FROM events WHERE event_metadata LIKE :pat"),
                        {"pat": f"%{dup['id']}%"},
                    )
                ).all()
                for m in metas:
                    new_raw = re.sub(
                        r'"product_id"\s*:\s*"' + re.escape(dup["id"]) + r'"',
                        '"product_id": "' + info["canonical"]["id"] + '"',
                        m.event_metadata or "",
                    )
                    if new_raw != m.event_metadata:
                        await db.execute(
                            text("UPDATE events SET event_metadata = :v WHERE id = :id"),
                            {"v": new_raw, "id": m.id},
                        )

        recs = (
            await db.execute(
                text(f"SELECT id, {', '.join(JSON_COLUMNS)} FROM recommendations")
            )
        ).all()
        for rec in recs:
            for col in JSON_COLUMNS:
                raw = getattr(rec, col) or ""
                new_raw = raw
                for dup_id, canon_id in dup_to_canon.items():
                    new_raw = re.sub(
                        r'"product_id"\s*:\s*"' + re.escape(dup_id) + r'"',
                        '"product_id": "' + canon_id + '"',
                        new_raw,
                    )
                if new_raw != raw:
                    await db.execute(
                        text(f"UPDATE recommendations SET {col} = :v WHERE id = :id"),
                        {"v": new_raw, "id": rec.id},
                    )
                    json_changed += 1

        for info in mapping.values():
            for dup in info["removed"]:
                await db.execute(text("DELETE FROM products WHERE id = :id"), {"id": dup["id"]})

        await db.execute(text("CREATE UNIQUE INDEX IF NOT EXISTS uq_products_title ON products (title)"))
        await db.commit()

    from app.services import vector_store

    deleted_vectors = 0
    for info in mapping.values():
        for dup in info["removed"]:
            vid = dup["vector_id"]
            try:
                if await vector_store.product_exists(vid):
                    await vector_store.delete_product(vid)
                    deleted_vectors += 1
            except Exception as exc:  # noqa: BLE001 — orphan vector is harmless
                print(f"[warn] could not delete vector {vid}: {exc}")
    print(f"Vector cleanup: {deleted_vectors} removed (leftovers are harmless).")
    return json_changed


def verify(conn: sqlite3.Connection, mapping: dict[str, dict]) -> None:
    total = conn.execute("SELECT COUNT(*) FROM products").fetchone()[0]
    dup_titles = conn.execute(
        "SELECT title, COUNT(*) FROM products GROUP BY title HAVING COUNT(*) > 1"
    ).fetchall()
    dangling_events = conn.execute(
        "SELECT COUNT(*) FROM events WHERE product_id NOT IN (SELECT id FROM products)"
    ).fetchone()[0]
    product_ids = {r[0] for r in conn.execute("SELECT id FROM products")}
    dangling_recs = 0
    for col in JSON_COLUMNS:
        for (raw,) in conn.execute(f"SELECT {col} FROM recommendations WHERE {col} IS NOT NULL"):
            for pid in PRODUCT_ID_PATTERN.findall(raw or ""):
                if pid.lower() not in {p.lower() for p in product_ids}:
                    dangling_recs += 1
    has_index = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'index' AND name = 'uq_products_title'"
    ).fetchone()

    print("\nVerification")
    print(f"  products rows:            {total} (expected {len(mapping)})")
    print(f"  duplicate titles:         {len(dup_titles)}")
    print(f"  events with dangling pid: {dangling_events}")
    print(f"  rec JSON dangling refs:   {dangling_recs}")
    print(f"  uq_products_title index:  {'present' if has_index else 'MISSING'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--apply", action="store_true", help="execute the migration (default: dry-run)")
    parser.add_argument("--yes", action="store_true", help="skip the interactive confirmation")
    args = parser.parse_args()

    conn = open_ro()
    mapping = build_mapping(conn)
    total = conn.execute("SELECT COUNT(*) FROM products").fetchone()[0]
    removed = sum(len(info["removed"]) for info in mapping.values())
    print(f"Products: {total} rows across {len(mapping)} titles; {removed} duplicate rows.")
    print_plan(mapping, conn)
    missing = [item["title"] for item in CATALOG if item["title"] not in mapping]
    if missing:
        print(f"[warn] {len(missing)} catalog titles are missing from the DB — they will be "
              "re-created by seed_catalog() on next startup.")
    conn.close()

    if not args.apply:
        print("\nDry-run complete — nothing was changed. Re-run with --apply to execute.")
        return

    if not args.yes:
        reply = input(f"\nThis will modify {db_path()} and its vector store. Type APPLY to continue: ")
        if reply.strip() != "APPLY":
            print("Aborted — nothing was changed.")
            return

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    backup_db(stamp)
    backup_chroma(stamp)
    json_changed = asyncio.run(apply_migration(mapping))
    print(f"Rewrote {json_changed} recommendation JSON columns.")

    conn = open_ro()
    verify(conn, mapping)
    conn.close()
    print("\nRestore if ever needed: stop the app, replace the DB file and Chroma "
          "directory with the *.pre-dedupe* snapshots taken above.")


if __name__ == "__main__":
    main()