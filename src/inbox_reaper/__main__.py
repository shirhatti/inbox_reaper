"""Allow running inbox_reaper as a module with python -m inbox_reaper."""

from .cli import main

if __name__ == "__main__":
    main()
