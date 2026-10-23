"""Legacy launcher. Use the installed grounded-stream CLI; see docs/migration.md."""
from grounded_stream.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
