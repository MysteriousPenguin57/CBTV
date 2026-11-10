"""mkdocs hooks: publish the repo-root README / CHANGELOG / CONTRIBUTING / CODE_OF_CONDUCT as pages
and point the docs'
../ links at them (or at GitHub for non-markdown files). Content is not changed otherwise."""
import re
from pathlib import Path

from mkdocs.structure.files import File

ROOT = Path(__file__).resolve().parents[2]
# repo file -> page path. README is renamed: mkdocs would treat README.md as the folder index and replace index.md.
ROOT_PAGES = {"README.md": "project-readme.md", "CHANGELOG.md": "CHANGELOG.md", "CONTRIBUTING.md": "CONTRIBUTING.md",
              "CODE_OF_CONDUCT.md": "CODE_OF_CONDUCT.md"}
BLOB = "https://github.com/MysteriousPenguin57/CBTV/blob/main/"


def on_files(files, config):
    for name, page in ROOT_PAGES.items():
        files.append(File.generated(config, page, content=(ROOT / name).read_text(encoding="utf-8")))
    return files


# a link in a repo-root page (README, ...) to another repo markdown file: relative to the repo root, not to docs/
_ROOT_MD_LINK = re.compile(r"\]\((?!https?:|#|/)([^)\s#]+\.md)(#[^)\s]*)?\)")


def _root_page_link(m):
    target, anchor = m.group(1), m.group(2) or ""
    if target in ROOT_PAGES:
        return "](%s%s)" % (ROOT_PAGES[target], anchor)
    if target.startswith("docs/"):
        return "](%s%s)" % (target[len("docs/"):], anchor)
    return "](%s%s%s)" % (BLOB, target, anchor)  # not a docs page: link to GitHub


def on_page_markdown(markdown, page, config, files):
    if page.file.src_uri in ROOT_PAGES.values():
        markdown = _ROOT_MD_LINK.sub(_root_page_link, markdown)
        for name in ["LICENSE", "CITATION.cff"]:  # repo-root files linked from a repo-root page
            markdown = markdown.replace("](%s)" % name, "](%s%s)" % (BLOB, name))
    for name, page in ROOT_PAGES.items():
        markdown = markdown.replace("](../%s" % name, "](%s" % page)
    for name in ["LICENSE", "CITATION.cff"]:
        markdown = markdown.replace("](../%s)" % name, "](%s%s)" % (BLOB, name))
    return markdown
