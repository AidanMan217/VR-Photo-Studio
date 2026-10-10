"""Checks of the user-facing documentation: links and anchors resolve, key facts stay in sync."""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
DOCS = [
    ROOT / "README.md",
    ROOT / "CHANGELOG.md",
    ROOT / "THIRD_PARTY_NOTICES.md",
    ROOT / "docs" / "USER_GUIDE.md",
    ROOT / "docs" / "QUEST_CHECKLIST.md",
    ROOT / "docs" / "ARCHITECTURE.md",
]

_LINK_RE = re.compile(r"(?<!!)\[[^\]]*\]\(([^)\s]+)\)")
_CODE_FENCE_RE = re.compile(r"```.*?```", re.S)


def github_slug(heading: str) -> str:
    """GitHub's heading anchor algorithm (lowercase, drop punctuation, spaces -> hyphens)."""
    text = re.sub(r"`([^`]*)`", r"\1", heading.strip()).lower()
    out = []
    for ch in text:
        cat = unicodedata.category(ch)
        if ch in "-_" or cat[0] in ("L", "N"):
            out.append(ch)
        elif ch == " ":
            out.append("-")
    return "".join(out)


def anchors(path: Path) -> set[str]:
    text = _CODE_FENCE_RE.sub("", path.read_text(encoding="utf-8"))
    seen: dict[str, int] = {}
    result = set()
    for m in re.finditer(r"^#{1,6}\s+(.+?)\s*#*\s*$", text, re.M):
        slug = github_slug(m.group(1))
        n = seen.get(slug, 0)
        result.add(slug if n == 0 else f"{slug}-{n}")
        seen[slug] = n + 1
    return result


def _gui_mode_titles() -> dict[str, str]:
    """``job kind -> title`` exactly as the GUI's "What do you want to make?" cards show it."""
    from vrphoto.gui.options_meta import MODES

    return {m.kind: m.title for m in MODES}


def _table_rows(text: str, header_start: str) -> list[str]:
    """The body rows of the first Markdown table whose header row starts with ``header_start``."""
    lines = text.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith(header_start))
    rows = []
    for line in lines[start + 2 :]:
        if not line.startswith("|"):
            break
        rows.append(line)
    return rows


def test_github_slug():
    assert (
        github_slug("4.7 Capture (many photos → walk-around 3D)") == "47-capture-many-photos--walk-around-3d"
    )
    assert github_slug('"Windows protected your PC"') == "windows-protected-your-pc"
    assert github_slug("4.6 Object / Person (walk-around AR)") == "46-object--person-walk-around-ar"


@pytest.mark.parametrize("doc", DOCS, ids=lambda p: p.name)
def test_doc_exists_and_is_utf8(doc):
    text = doc.read_text(encoding="utf-8")
    assert text.startswith("# ")
    assert "\t" not in text


@pytest.mark.parametrize("doc", DOCS, ids=lambda p: p.name)
def test_relative_links_and_anchors_resolve(doc):
    text = _CODE_FENCE_RE.sub("", doc.read_text(encoding="utf-8"))
    problems = []
    for target in _LINK_RE.findall(text):
        if re.match(r"^[a-z]+:", target):  # http(s):, mailto:
            continue
        file_part, _, anchor = target.partition("#")
        dest = (doc.parent / file_part).resolve() if file_part else doc
        if not dest.exists():
            problems.append(f"{target}: file not found")
            continue
        if anchor and dest.suffix == ".md" and anchor not in anchors(dest):
            problems.append(f"{target}: no heading with anchor #{anchor} in {dest.name}")
    assert not problems, "\n".join(problems)


def test_readme_mentions_essentials():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    for needle in (
        "More info",
        "Run anyway",
        "Send to Quest",
        "Movies",
        "USER_GUIDE.md",
        "QUEST_CHECKLIST.md",
    ):
        assert needle in text, needle


def test_user_guide_covers_all_job_modes_and_tokens():
    from vrphoto.pipeline.jobs import JOB_TYPES

    text = (ROOT / "docs" / "USER_GUIDE.md").read_text(encoding="utf-8")
    titles = _gui_mode_titles()
    assert set(titles) == set(JOB_TYPES), "a new job kind needs a User Guide section"
    for kind, title in titles.items():
        # the section heading uses exactly the name shown on the GUI's mode card
        assert re.search(rf"^### 4\.\d+ {re.escape(title)}( \(|$)", text, re.M), (kind, title)
    for token in ("_LR", "_TB", "_180_LR", "_360_TB", "`_360`"):
        assert token in text
    for player in ("Meta Quest TV", "Skybox", "DeoVR", "HereSphere"):
        assert player in text
    for word in ("Allow access to data", "Advanced", "Proceed", "Enter VR", "Enter AR", "adb", "Developer"):
        assert word in text


def test_ports_in_docs_match_settings():
    from vrphoto.core.config import Settings

    s = Settings()
    guide = (ROOT / "docs" / "USER_GUIDE.md").read_text(encoding="utf-8")
    assert f":{s.https_port}" in guide and f"localhost:{s.http_port}" in guide


def test_readme_feature_table_uses_gui_mode_names():
    rows = _table_rows((ROOT / "README.md").read_text(encoding="utf-8"), "| Mode |")
    names = {re.match(r"\|\s*\*\*(.+?)\*\*", r).group(1) for r in rows}
    assert names == set(_gui_mode_titles().values())


def _section(text: str, heading_re: str) -> str:
    m = re.search(heading_re, text, re.M)
    assert m, heading_re
    nxt = re.search(r"^#{2,3} ", text[m.end() :], re.M)
    return text[m.end() : m.end() + nxt.start()] if nxt else text[m.end() :]


def _option_labels(section: str) -> list[str]:
    """First-column labels of the ``| Option |`` table in a guide section (markers removed)."""
    if "| Option |" not in section:
        return []
    labels = []
    for row in _table_rows(section, "| Option |"):
        cell = row.split("|")[1]
        cell = re.sub(r"\*\((Advanced|More)\)\*", "", cell).strip()
        labels.extend(part.strip() for part in cell.split(" / "))
    return labels


def test_user_guide_option_names_match_the_gui():
    """Every option a mode table names is a field label the GUI shows for that mode."""
    from vrphoto.gui.options_meta import COMMON_META, JOB_META

    text = (ROOT / "docs" / "USER_GUIDE.md").read_text(encoding="utf-8")
    problems = []
    for kind, title in _gui_mode_titles().items():
        section = _section(text, rf"^### 4\.\d+ {re.escape(title)}\b.*$")
        gui = {m.label for m in JOB_META.get(kind, {}).values()}
        problems += [f"{title}: {lab!r}" for lab in _option_labels(section) if lab not in gui]
    common = _section(text, r"^### In every mode.*$")
    labels = _option_labels(common)
    assert labels, "the 'In every mode' table is missing"
    gui_common = {m.label for m in COMMON_META.values()}
    problems += [f"common: {lab!r}" for lab in labels if lab not in gui_common]
    assert not problems, "User Guide names options the GUI does not show:\n" + "\n".join(problems)


def test_docs_name_buttons_and_settings_that_exist():
    """Button / setting names the docs tell the user to click exist in the GUI source."""
    gui = ROOT / "src" / "vrphoto" / "gui"
    source = "\n".join(p.read_text(encoding="utf-8") for p in gui.rglob("*.py"))
    for label in (
        "Send to Quest",
        "Add photos…",
        "Add folder…",
        "Allow through firewall",
        "Check again",
        "Open on this PC",
        "Start viewer",
        "Stop viewer",
        "Install USB tools",
        "Set up USB connection",
        "Copy files to Quest",
        "View in headset",
        "Open folder",
        "Clear finished",
        "Show experimental add-ons",
        "Show experimental features",
        '"Device"',
        '"Headset"',
        '"Video encoder"',
        '"Hugging Face token"',
        '"Theme"',
        '"Your scenes"',
        '"USB cable"',
        '"Preset"',
        '"Depth, AI models and output"',
        '"AI models"',
    ):
        assert label in source, f"docs mention {label}, which the GUI no longer has"
    stale = (
        "Quest model*",
        "Video to 3D",
        "Object / Person",
        "USB cable (adb)",
        "Camera path",
        "Video preset",
    )
    for doc in ("README.md", "docs/USER_GUIDE.md", "docs/QUEST_CHECKLIST.md"):
        text = (ROOT / doc).read_text(encoding="utf-8")
        for word in stale:
            assert word not in text, f"{doc}: outdated name {word!r}"


def _short_title(title: str) -> str:
    return title.split(" (")[0].strip()


def _manifests() -> list:
    from vrphoto.addons.manifest import load_manifest, manifests_dir

    return [load_manifest(p) for p in sorted(manifests_dir().glob("*.toml"))]


def test_user_guide_addon_table_matches_manifests():
    """Name, experimental/non-commercial flags, VRAM and disk figures come from the manifests."""
    text = (ROOT / "docs" / "USER_GUIDE.md").read_text(encoding="utf-8")
    rows = _table_rows(text, "| Add-on |")
    problems = []
    manifests = _manifests()
    assert manifests
    for m in manifests:
        short = _short_title(m.title)
        row = next((r for r in rows if re.match(rf"\|\s*\*\*{re.escape(short)}\*\*", r)), None)
        if row is None:
            problems.append(f"{m.id}: no row starting with **{short}**")
            continue
        low = row.lower()
        if m.experimental and "experimental" not in low:
            problems.append(f"{m.id}: not marked experimental")
        if m.noncommercial and "non-commercial" not in low:
            problems.append(f"{m.id}: not marked non-commercial")
        if m.region_restricted and "not licensed in" not in low:
            problems.append(f"{m.id}: region restriction not mentioned")
        if f"{m.min_vram_gb:g} GB+" not in row:
            problems.append(f"{m.id}: minimum VRAM should read '{m.min_vram_gb:g} GB+'")
        if f"~{m.disk_gb:g} GB" not in row:
            problems.append(f"{m.id}: disk should read '~{m.disk_gb:g} GB'")
    assert len(rows) == len(manifests), "one table row per add-on manifest"
    assert not problems, "\n".join(problems)


def test_notices_list_every_addon():
    text = (ROOT / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
    for m in _manifests():
        assert _short_title(m.title).split(" for ")[0] in text, m.id


def test_notices_cover_required_components():
    text = (ROOT / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
    for needle in (
        "LGPL-3.0",
        "GPL-3.0",
        "PySide6",
        "FFmpeg",
        "ONNX Runtime",
        "OpenCV",
        "NumPy",
        "Pillow",
        "pillow-heif",
        "x265",
        "cryptography",
        "qrcode",
        "uv",
        "three.js",
        "Spark",
        "WebXR Input Profiles",
        "Spatial Media",
        "Depth Anything V2 Small",
        "CC-BY-NC-4.0",
        "MoGe",
        "LaMa",
        "BiRefNet",
        "MI-GAN",
        "TRELLIS",
        "nvdiffrast",
        "Hunyuan3D",
        "SHARP",
        "COLMAP",
        "Brush",
        "CreativeML Open RAIL++-M",
    ):
        assert needle in text, needle


def test_license_is_mit_2026():
    text = (ROOT / "LICENSE").read_text(encoding="utf-8")
    assert text.startswith("MIT License")
    assert "Copyright (c) 2026 AidanMan217" in text


def test_changelog_version_matches_package():
    import vrphoto

    text = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    assert f"## [{vrphoto.__version__}]" in text
