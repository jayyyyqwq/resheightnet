"""M0.5: split parity & leakage audit.

Verifies the committed data/splits/*.txt files (copied verbatim from
DepthWizard's stage_a1 manifests) have not been altered, are internally
consistent, and preserve the invariants the rest of the project
depends on -- most importantly that column 2 (the upstream HF folder)
must NOT be mistaken for the experiment split.
"""
from __future__ import annotations

import hashlib
import os
import re

import pytest

from src.dataset import parse_split_file, SplitRow
from tests.helpers import write_split_file

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_SPLIT = os.path.join(REPO_ROOT, "data", "splits", "stage_a1_train.txt")
VAL_SPLIT = os.path.join(REPO_ROOT, "data", "splits", "stage_a1_val.txt")

# Pinned at copy time (see PowerShell sha256sum output in the build log).
# A mismatch means the split file was edited -- e.g. an editor silently
# converting tabs to spaces -- which would corrupt the 5-column parse
# and break comparability with DepthWizard's reported numbers.
EXPECTED_TRAIN_SHA256 = "c046a863c1db6f1a0bfcf24a80c830b001add1a8e01a3408d612857f0e19c06b"
EXPECTED_VAL_SHA256 = "725f232e2c135d3746168c43a3b65a1ab436a5302a45c95ea6c23df3ee3b8050"

RELPATH_RE = re.compile(r"^(images|heights|classes)/(train|val|test)/.+\.h5$")


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def test_split_files_exist():
    assert os.path.isfile(TRAIN_SPLIT), f"missing {TRAIN_SPLIT}"
    assert os.path.isfile(VAL_SPLIT), f"missing {VAL_SPLIT}"


def test_split_file_sha256_pinned():
    train_hash = _sha256(TRAIN_SPLIT)
    val_hash = _sha256(VAL_SPLIT)
    assert train_hash == EXPECTED_TRAIN_SHA256, (
        f"stage_a1_train.txt content changed (sha256={train_hash}); "
        f"re-copy from DepthWizard's data/gamus/splits/ if this is unintended"
    )
    assert val_hash == EXPECTED_VAL_SHA256, (
        f"stage_a1_val.txt content changed (sha256={val_hash}); "
        f"re-copy from DepthWizard's data/gamus/splits/ if this is unintended"
    )


def test_row_counts():
    train_rows = parse_split_file(TRAIN_SPLIT)
    val_rows = parse_split_file(VAL_SPLIT)
    assert len(train_rows) == 1000
    assert len(val_rows) == 200


def test_every_row_has_five_tab_fields():
    for split_path in (TRAIN_SPLIT, VAL_SPLIT):
        with open(split_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                fields = line.rstrip("\n").split("\t")
                assert len(fields) == 5, f"{split_path}:{line_num} has {len(fields)} fields"


def test_relpaths_match_expected_pattern():
    for split_path in (TRAIN_SPLIT, VAL_SPLIT):
        for row in parse_split_file(split_path):
            for relpath in (row.rgb_relpath, row.height_relpath, row.cls_relpath):
                assert RELPATH_RE.match(relpath), f"unexpected relpath shape: {relpath}"


def test_zero_sample_id_overlap_between_train_and_val():
    train_ids = {r.sample_id for r in parse_split_file(TRAIN_SPLIT)}
    val_ids = {r.sample_id for r in parse_split_file(VAL_SPLIT)}
    overlap = train_ids & val_ids
    assert overlap == set(), f"{len(overlap)} sample_ids appear in both splits: {sorted(overlap)[:10]}"


def test_zero_rgb_path_overlap_between_train_and_val():
    train_paths = {r.rgb_relpath for r in parse_split_file(TRAIN_SPLIT)}
    val_paths = {r.rgb_relpath for r in parse_split_file(VAL_SPLIT)}
    assert (train_paths & val_paths) == set()


def test_city_balance():
    train_rows = parse_split_file(TRAIN_SPLIT)
    val_rows = parse_split_file(VAL_SPLIT)
    train_dc = sum(1 for r in train_rows if r.sample_id.startswith("DC_"))
    train_phl = sum(1 for r in train_rows if r.sample_id.startswith("PHL_"))
    val_dc = sum(1 for r in val_rows if r.sample_id.startswith("DC_"))
    val_phl = sum(1 for r in val_rows if r.sample_id.startswith("PHL_"))
    assert (train_dc, train_phl) == (500, 500)
    assert (val_dc, val_phl) == (100, 100)


def test_column_2_is_not_the_experiment_split():
    """Column 2 (hf_folder) is the upstream HF repo folder, independent
    of which FILE (train/val) a row belongs to. This guards against a
    future "fix" that mistakenly filters by column 2 instead of by
    split-file membership -- stage_a1_val.txt legitimately contains
    rows whose column 2 is "train"."""
    val_rows = parse_split_file(VAL_SPLIT)
    hf_folders_in_val = {r.hf_folder for r in val_rows}
    assert "train" in hf_folders_in_val, (
        "expected the val split to contain rows whose hf_folder column is "
        "'train' (proving column 2 != experiment split); if this no longer "
        "holds, some assumption about the split format has changed"
    )


def test_nyc_style_row_parses_without_crashing(tmp_path):
    """NYC rows carry an _IMG.h5 suffix on the sample_id itself (unlike
    DC/PHL rows). NYC is out of scope for stage_a1, but the parser must
    not choke if such a row ever appears."""
    nyc_row = {
        "sample_id": "NYC_00735_IMG.h5",
        "rgb_relpath": "images/test/NYC_00735_IMG.h5",
        "height_relpath": "heights/test/NYC_00735_AGL.h5",
        "cls_relpath": "classes/test/NYC_00735_CLS.h5",
    }
    split_file = tmp_path / "nyc_test.txt"
    write_split_file(str(split_file), [nyc_row], hf_folder="test")

    rows = parse_split_file(str(split_file))
    assert len(rows) == 1
    assert rows[0].sample_id == "NYC_00735_IMG.h5"
    assert rows[0].rgb_relpath == "images/test/NYC_00735_IMG.h5"


def test_parse_split_file_rejects_wrong_field_count(tmp_path):
    bad_file = tmp_path / "bad.txt"
    bad_file.write_text("only\tfour\tfields\there\n", encoding="utf-8")
    with pytest.raises(ValueError, match="expected 5 tab-separated fields"):
        parse_split_file(str(bad_file))


def test_parse_split_file_skips_blank_lines(tmp_path):
    f = tmp_path / "with_blanks.txt"
    row = "A\ttrain\timages/train/A_RGB.h5\theights/train/A_AGL.h5\tclasses/train/A_CLS.h5\n"
    f.write_text(f"{row}\n{row}", encoding="utf-8")
    rows = parse_split_file(str(f))
    assert len(rows) == 2


def _tile_grid_coords(sample_id: str) -> tuple[str, int, int] | None:
    """DC/PHL sample_ids encode a tile grid position as CITY_row_col."""
    m = re.match(r"^(DC|PHL)_(\d+)_(\d+)$", sample_id)
    if not m:
        return None
    return m.group(1), int(m.group(2)), int(m.group(3))


def test_spatial_adjacency_disclosed_in_limitations():
    """Regression guard on a disclosed limitation: some val tiles are
    grid-adjacent to train tiles (same city, row/col within 1), which
    means absolute MAE is somewhat optimistic vs. a geographically
    disjoint holdout. This does not break comparability with DepthWizard
    (both systems use the identical split) but must stay documented.
    """
    train_coords = {}
    for row in parse_split_file(TRAIN_SPLIT):
        c = _tile_grid_coords(row.sample_id)
        if c:
            train_coords.setdefault(c[0], set()).add((c[1], c[2]))

    val_rows = parse_split_file(VAL_SPLIT)
    adjacent_count = 0
    for row in val_rows:
        c = _tile_grid_coords(row.sample_id)
        if not c:
            continue
        city, r, col = c
        neighbors = {(r + dr, col + dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)}
        if train_coords.get(city, set()) & neighbors:
            adjacent_count += 1

    limitations_path = os.path.join(REPO_ROOT, "docs", "limitations.md")
    assert os.path.isfile(limitations_path), (
        f"found {adjacent_count} val tiles grid-adjacent to a train tile, "
        f"but docs/limitations.md does not exist yet to disclose this"
    )
    with open(limitations_path, "r", encoding="utf-8") as f:
        content = f.read()
    assert "adjac" in content.lower(), (
        "docs/limitations.md exists but does not appear to mention the "
        "spatial-adjacency finding"
    )
