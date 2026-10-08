from fastapi import Depends, HTTPException, Request, status

from app.auth.deps import require_jwt
from app.db.database import open_db
from app.db.repos.users import UserRepo


async def current_user_id(request: Request, sub: str = Depends(require_jwt)) -> int:
    """Resolve the JWT subject (a username) to users.id once per request."""
    async with open_db(request.app.state.settings.db_path) as db:
        row = await UserRepo(db).get_by_username(sub)
    if row is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "unknown subject")
    return row.id
