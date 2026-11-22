import sys

from bot import CareerBot


def setup_console():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")


if __name__ == "__main__":
    setup_console()
    CareerBot().run()
