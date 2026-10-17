from pathlib import Path
import os

root = Path(__file__).parent
repo = root.parent


def load_env(env_path: Path | None = None) -> None:
    """Load key=value pairs from the repo's .env file into os.environ.

    Existing environment variables are NOT overridden.
    """
    env_path = env_path or (repo / ".env")
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())
