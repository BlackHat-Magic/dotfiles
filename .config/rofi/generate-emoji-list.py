#!/usr/bin/env python3
"""Generate the Rofi emoji list from pinned Unicode and CLDR data."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from urllib.request import Request, urlopen


CLDR_URLS = (
    "https://raw.githubusercontent.com/unicode-org/cldr/refs/tags/release-48-2/common/"
    "annotations/en.xml",
    "https://raw.githubusercontent.com/unicode-org/cldr/refs/tags/release-48-2/common/"
    "annotationsDerived/en.xml",
)
EMOJI_TEST_URL = "https://unicode.org/Public/17.0.0/emoji/emoji-test.txt"


def fetch(url: str) -> bytes:
    request = Request(url, headers={"User-Agent": "dotfiles-emoji-generator"})
    with urlopen(request, timeout=30) as response:
        return response.read()


def parse_annotations(xml: bytes) -> dict[str, tuple[str, ...]]:
    import xml.etree.ElementTree as element_tree

    annotations: dict[str, tuple[str, ...]] = {}
    root = element_tree.fromstring(xml)
    for annotation_group in root.findall("annotations"):
        for annotation in annotation_group.findall("annotation"):
            sequence = annotation.attrib["cp"]
            if annotation.get("type") == "tts":
                name = annotation.text or ""
                keywords = annotations.get(sequence, ("",))[1:]
                annotations[sequence] = (name, *keywords)
            elif sequence not in annotations:
                keywords = tuple((annotation.text or "").split(" | "))
                annotations[sequence] = ("", *keywords)
    return annotations


def parse_emoji_test(text: str) -> list[str]:
    emoji: list[str] = []
    for line in text.splitlines():
        if ";" not in line:
            continue
        codepoints, status = line.split(";", 1)
        if not status.lstrip().startswith("fully-qualified"):
            continue
        sequence = "".join(chr(int(codepoint, 16)) for codepoint in codepoints.split())
        if sequence not in emoji:
            emoji.append(sequence)
    return emoji


def annotation_for(
    sequence: str, annotations: dict[str, tuple[str, ...]]
) -> tuple[str, ...] | None:
    annotation = annotations.get(sequence)
    if annotation and annotation[0]:
        return annotation

    # CLDR annotation data may omit variation selectors from its keys.
    return annotations.get(sequence.replace("\ufe0f", ""))


def build_list(emoji_test: str, annotations: dict[str, tuple[str, ...]]) -> list[str]:
    rows: list[str] = []
    missing: list[str] = []
    for sequence in parse_emoji_test(emoji_test):
        annotation = annotation_for(sequence, annotations)
        if annotation is None:
            missing.append(sequence)
            continue
        name, *keywords = annotation
        rows.append(f"{sequence}\t{name}\t{' '.join(keywords)}")

    if missing:
        print(
            f"warning: skipped {len(missing)} emoji without CLDR names",
            file=sys.stderr,
        )
    return rows


def load_sources(cache_dir: Path | None) -> tuple[str, dict[str, tuple[str, ...]]]:
    if cache_dir is None:
        sources = [fetch(url) for url in (*CLDR_URLS, EMOJI_TEST_URL)]
    else:
        cache_dir.mkdir(parents=True, exist_ok=True)
        source_paths = [
            cache_dir / "annotations.xml",
            cache_dir / "annotationsDerived.xml",
            cache_dir / "emoji-test.txt",
        ]
        for path, url in zip(source_paths, (*CLDR_URLS, EMOJI_TEST_URL), strict=True):
            if not path.exists():
                path.write_bytes(fetch(url))
        sources = [path.read_bytes() for path in source_paths]

    annotations = parse_annotations(sources[0])
    annotations.update(parse_annotations(sources[1]))
    return sources[2].decode("utf-8"), annotations


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path(__file__).with_name("emoji.txt"),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        help="Cache the pinned upstream source files in this directory.",
    )
    args = parser.parse_args()

    emoji_test, annotations = load_sources(args.cache_dir)
    rows = build_list(emoji_test, annotations)
    args.output.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"wrote {len(rows)} emoji to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
