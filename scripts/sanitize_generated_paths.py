from __future__ import annotations

import argparse
from pathlib import Path


TEXT_SUFFIXES = {
    ".csv",
    ".json",
    ".md",
    ".txt",
    ".py",
    ".ini",
    ".toml",
    ".h",
    ".cpp",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replace local absolute project paths in text artifacts with relative paths.")
    parser.add_argument("--root", default=Path("."), type=Path)
    parser.add_argument("--project-root", default=Path.cwd(), type=Path)
    return parser.parse_args()


def replacements(project_root: Path) -> list[tuple[str, str]]:
    resolved = project_root.resolve()
    win = str(resolved)
    posix = resolved.as_posix()
    escaped_win = win.replace("\\", "\\\\")
    rel_prefix = ""
    return [
        (win + "\\", rel_prefix),
        (win + "/", rel_prefix),
        (posix + "/", rel_prefix),
        (escaped_win + "\\\\", rel_prefix),
        (escaped_win + "/", rel_prefix),
        ("\\\\?\\" + win + "\\", rel_prefix),
        ("\\\\?\\" + win + "/", rel_prefix),
        ("\\\\?\\\\" + escaped_win + "\\\\", rel_prefix),
        (win, "."),
        (posix, "."),
        (escaped_win, "."),
    ]


def should_skip(path: Path) -> bool:
    parts = set(path.parts)
    if ".git" in parts or ".venv" in parts or ".pio" in parts or ".vscode" in parts or "__pycache__" in parts:
        return True
    return path.suffix.lower() not in TEXT_SUFFIXES


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    reps = replacements(args.project_root)
    changed: list[str] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or should_skip(path):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        updated = text
        for old, new in reps:
            updated = updated.replace(old, new)
        if updated != text:
            path.write_text(updated, encoding="utf-8")
            changed.append(path.relative_to(root).as_posix())
    print(f"sanitized_files={len(changed)}")
    for name in changed:
        print(name)


if __name__ == "__main__":
    main()
