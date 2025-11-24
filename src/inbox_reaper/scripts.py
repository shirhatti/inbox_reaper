"""Development setup scripts."""

import subprocess
import sys


def setup_dev():
    """Set up development environment with pre-commit hooks."""
    print("🔧 Setting up development environment...")

    # Install pre-commit hooks
    try:
        print("\n📦 Installing pre-commit hooks...")
        subprocess.run(
            ["pre-commit", "install"],
            check=True,
            capture_output=True,
            text=True,
        )
        print("✅ Pre-commit hooks installed successfully!")

        # Run pre-commit on all files to ensure everything is set up correctly
        print("\n🔍 Running pre-commit checks on all files...")
        result = subprocess.run(
            ["pre-commit", "run", "--all-files"],
            capture_output=True,
            text=True,
        )

        # pre-commit returns 1 if it made changes, which is okay for initial setup
        if result.returncode == 0:
            print("✅ All pre-commit checks passed!")
        else:
            print("ℹ️  Pre-commit made some changes. This is normal for initial setup.")
            print("\nYou may need to review and commit these changes.")

        print("\n✨ Development environment is ready!")
        print("\nNext steps:")
        print("  - Make your changes")
        print("  - Pre-commit hooks will run automatically on git commit")
        print("  - Or run manually: pre-commit run --all-files")

    except FileNotFoundError:
        print(
            "❌ Error: pre-commit not found. "
            "Install dev dependencies first: uv pip install -e '.[dev]'"
        )
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing pre-commit hooks: {e}")
        sys.exit(1)


if __name__ == "__main__":
    setup_dev()
