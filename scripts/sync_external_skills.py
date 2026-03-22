#!/usr/bin/env python3
"""Sync external skills from configured repositories."""

import subprocess
import sys
import tempfile
import yaml
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set, Optional

script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(script_dir))

from sync_types import ExternalSource, Skill, ConflictInfo


def load_config(config_path: str) -> List[ExternalSource]:
    """Load external sources from YAML config file.

    Args:
        config_path: Path to YAML configuration file.

    Returns:
        List of ExternalSource configurations.

    Raises:
        FileNotFoundError: If config file doesn't exist.
        yaml.YAMLError: If config file contains invalid YAML.
    """
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    config_data = yaml.safe_load(config_file.read_text(encoding="utf-8"))

    sources = []
    for source_data in config_data.get("sources", []):
        source = ExternalSource(
            name=source_data["name"],
            url=source_data["url"],
            branch=source_data.get("branch", "main"),
            enabled=source_data.get("enabled", True),
        )
        sources.append(source)

    return sources


def clone_external_repo(source: ExternalSource) -> Path:
    """Clone external repo to temp directory with --depth 1.

    Args:
        source: ExternalSource configuration with url, branch, etc.

    Returns:
        Path to the cloned temporary directory.

    Raises:
        subprocess.CalledProcessError: If git clone fails.
    """
    temp_dir = tempfile.mkdtemp(prefix=f"sync-{source.name}-")
    subprocess.run(
        [
            "git",
            "clone",
            "--depth",
            "1",
            "-b",
            source.branch,
            source.url,
            temp_dir,
        ],
        check=True,
        capture_output=True,
    )
    return Path(temp_dir)


def parse_skill_md(skill_path: Path) -> Dict:
    """Parse SKILL.md frontmatter and return as dict.

    Args:
        skill_path: Path to the skill directory.

    Returns:
        Dictionary containing parsed YAML frontmatter, or empty dict
        if no frontmatter found.
    """
    skill_md = skill_path / "SKILL.md"
    content = skill_md.read_text(encoding="utf-8")
    if content.startswith("---"):
        parts = content.split("---", 2)
        if len(parts) >= 3:
            return yaml.safe_load(parts[1]) or {}
    return {}


def find_skills(repo_path: Path, source: ExternalSource) -> List[Skill]:
    """Find all skills (dirs with SKILL.md) in repo.

    Args:
        repo_path: Path to the cloned repository root.
        source: ExternalSource this repository comes from.

    Returns:
        List of Skill objects for directories containing SKILL.md.
    """
    skills = []
    for item in repo_path.iterdir():
        if item.is_dir() and (item / "SKILL.md").exists():
            skills.append(
                Skill(
                    name=item.name,
                    path=item,
                    source=source,
                    has_skill_md=True,
                )
            )
    return skills


def get_local_skills() -> Set[str]:
    """Get skill names in repo root (excluding external/)."""
    skills = set()
    for item in Path(".").iterdir():
        if item.is_dir() and (item / "SKILL.md").exists() and item.name != "external":
            skills.add(item.name)
    return skills


def get_synced_skills() -> Set[str]:
    """Get skill names already synced in external/."""
    skills = set()
    external_dir = Path("external")
    if external_dir.exists():
        for source_dir in external_dir.iterdir():
            if source_dir.is_dir():
                for skill_dir in source_dir.iterdir():
                    if skill_dir.is_dir() and (skill_dir / "SKILL.md").exists():
                        skills.add(skill_dir.name)
    return skills


def detect_conflicts(
    skill: Skill, local_skills: Set[str], synced_skills: Set[str]
) -> Optional[ConflictInfo]:
    """Check if skill conflicts with local or synced skills."""
    if skill.name in local_skills:
        return ConflictInfo(
            skill_name=skill.name, local_path=f"./{skill.name}", external_source="local"
        )
    if skill.name in synced_skills:
        return ConflictInfo(
            skill_name=skill.name,
            local_path=f"./external/*/{skill.name}",
            external_source="synced",
        )
    return None


def inject_attribution(skill: Skill, commit_sha: str) -> str:
    """Inject source attribution into SKILL.md frontmatter.

    Args:
        skill: The Skill object containing source information
        commit_sha: The Git commit SHA to attribute

    Returns:
        Modified content string with injected attribution fields.
        Does NOT write to file.

    The function:
    - Preserves existing frontmatter fields
    - Adds attribution fields only if they don't exist
    - Does NOT modify the body content
    """
    skill_md_path = skill.path / "SKILL.md"
    content = skill_md_path.read_text(encoding="utf-8")

    # Parse existing frontmatter
    fm = {}
    body = content
    if content.startswith("---"):
        parts = content.split("---", 2)
        if len(parts) >= 3:
            fm = yaml.safe_load(parts[1]) or {}
            body = parts[2]

    # Inject attribution fields (don't overwrite existing)
    if "synced-from" not in fm:
        fm["synced-from"] = skill.source.url
    if "synced-date" not in fm:
        fm["synced-date"] = datetime.now().strftime("%Y-%m-%d")
    if "synced-commit" not in fm:
        fm["synced-commit"] = commit_sha
    if "license" not in fm:
        fm["license"] = "UNKNOWN"

    # Reassemble
    new_frontmatter = yaml.dump(fm, sort_keys=False, allow_unicode=True)
    return f"---\n{new_frontmatter}---\n{body}"


def main():
    """Main entry point."""
    # Default config path (relative to project root)
    config_path = ".github/external-sources.yml"

    try:
        sources = load_config(config_path)
        print(f"Loaded {len(sources)} external sources")
        for source in sources:
            print(f"  - {source.name}: {source.url} (branch: {source.branch})")
        print("\n✅ Configuration loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load config: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
