"""
One-off utility: promote a user to admin by email.
Usage: python -m scripts.make_admin your@email.com
"""
import asyncio
import sys

from sqlalchemy import select

from app.db.session import AsyncSessionLocal
from app.models.user import User, UserRole


async def make_admin(email: str) -> None:
    async with AsyncSessionLocal() as db:
        result = await db.execute(select(User).where(User.email == email))
        user = result.scalar_one_or_none()

        if not user:
            print(f"No user found with email: {email}")
            return

        if user.role == UserRole.ADMIN:
            print(f"{email} is already an admin.")
            return

        user.role = UserRole.ADMIN
        await db.commit()
        print(f"{email} is now an admin.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python -m scripts.make_admin your@email.com")
        sys.exit(1)

    asyncio.run(make_admin(sys.argv[1]))