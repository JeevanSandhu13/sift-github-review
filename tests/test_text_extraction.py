"""Tests for privacy-preserving unstructured-text extraction.

Analysis uses local, deterministic keyword classification and lexicon
sentiment.

Three layers, matching test_python_runtime_sanitizer.py's pattern:

1. Pure unit tests on the tokenizer / sentiment / classifier helpers
   in sift.runtime.sift -- no I/O, no sanitizer.
2. Runtime -> sanitizer round trips using the same env-var + reload
   fixture test_python_runtime_sanitizer.py established.
3. The core security property, exercised twice: a category below the
   suppression threshold must have BOTH its count suppressed AND its
   sentiment score withheld -- a sentiment map that "leaked through"
   for a suppressed category would re-identify it just as surely as
   the count would.
"""

from __future__ import annotations

import importlib
import json
import os
import sys
from pathlib import Path

import pandas as pd
import pytest

from sift.sanitizer import sanitize


_TEST_TOKEN = "deadbeef" * 8


@pytest.fixture
def runtime(tmp_path: Path):
    result_path = tmp_path / "result.json"
    prev_token = os.environ.get("SIFT_RUN_TOKEN")
    prev_path = os.environ.get("SIFT_RESULT_PATH")
    os.environ["SIFT_RUN_TOKEN"] = _TEST_TOKEN
    os.environ["SIFT_RESULT_PATH"] = str(result_path)
    sys.modules.pop("sift.runtime.sift", None)
    try:
        mod = importlib.import_module("sift.runtime.sift")
        yield mod, result_path
    finally:
        sys.modules.pop("sift.runtime.sift", None)
        if prev_token is None:
            os.environ.pop("SIFT_RUN_TOKEN", None)
        else:
            os.environ["SIFT_RUN_TOKEN"] = prev_token
        if prev_path is None:
            os.environ.pop("SIFT_RESULT_PATH", None)
        else:
            os.environ["SIFT_RESULT_PATH"] = prev_path


def _read_payload_strip_token(result_path: Path) -> dict:
    raw = json.loads(result_path.read_text(encoding="utf-8"))
    assert raw.get("_token") == _TEST_TOKEN
    return {k: v for k, v in raw.items() if k != "_token"}


# ---------------------------------------------------------------------------
# 1. Pure unit tests: tokenizer / sentiment / classifier
#
# sift.runtime.sift refuses import without SIFT_RUN_TOKEN set (a
# module-level guard against direct ``python script.py`` invocation
# outside the executor) — so these use the same ``runtime`` fixture
# as every other test here and reach the helpers off the loaded
# module, rather than importing them at file-collection time.
# ---------------------------------------------------------------------------

def test_tokenize_lowercases_and_splits_on_punctuation(runtime) -> None:
    mod, _path = runtime
    assert mod._text_extract_tokenize("Great, FAST service!") == [
        "great", "fast", "service",
    ]


def test_tokenize_empty_string(runtime) -> None:
    mod, _path = runtime
    assert mod._text_extract_tokenize("") == []


def test_sentiment_positive(runtime) -> None:
    mod, _path = runtime
    s = mod._text_extract_sentiment("This was a great and wonderful experience")
    assert s is not None and s > 0


def test_sentiment_negative(runtime) -> None:
    mod, _path = runtime
    s = mod._text_extract_sentiment("Terrible, broken, and a complete waste")
    assert s is not None and s < 0


def test_sentiment_no_lexicon_hits_returns_none(runtime) -> None:
    mod, _path = runtime
    assert mod._text_extract_sentiment("The package arrived on a Tuesday") is None


def test_sentiment_mixed_words_partially_cancel(runtime) -> None:
    mod, _path = runtime
    s = mod._text_extract_sentiment("good service but a terrible delay")
    assert s is not None
    assert -1.0 <= s <= 1.0


def test_sentiment_bounded_in_range(runtime) -> None:
    mod, _path = runtime
    s = mod._text_extract_sentiment("great great great awful")
    assert s is not None
    assert -1.0 <= s <= 1.0


def test_classify_first_match_wins(runtime) -> None:
    mod, _path = runtime
    categories = {
        "shipping": ["shipping", "delivery"],
        "billing": ["charge", "invoice"],
    }
    assert mod._text_extract_classify(
        "my delivery was late and the charge was wrong",
        categories, "uncategorized",
    ) == "shipping"


def test_classify_case_insensitive(runtime) -> None:
    mod, _path = runtime
    categories = {"shipping": ["SHIPPING"]}
    assert mod._text_extract_classify(
        "Shipping delay again", categories, "uncategorized",
    ) == "shipping"


def test_classify_no_match_falls_to_uncategorized(runtime) -> None:
    mod, _path = runtime
    categories = {"billing": ["invoice"]}
    assert mod._text_extract_classify(
        "the app crashed on login", categories, "uncategorized",
    ) == "uncategorized"


# ---------------------------------------------------------------------------
# 2. from_text_extract -> sanitizer round trip
# ---------------------------------------------------------------------------

def _feedback_df() -> pd.DataFrame:
    # 30 rows: 20 shipping complaints (negative), 10 billing (mixed).
    # Well above the default suppression threshold (10) on both cells.
    texts = (
        ["my package was delayed and the delivery was terrible"] * 15
        + ["shipping was slow but the product itself was great"] * 5
        + ["the invoice charge was wrong, very frustrating"] * 7
        + ["billing was resolved quickly, thanks for the help"] * 3
    )
    return pd.DataFrame({"feedback": texts})


def test_from_text_extract_through_sanitizer(runtime) -> None:
    mod, path = runtime
    df = _feedback_df()
    mod.from_text_extract(
        df, "feedback",
        categories={
            "shipping": ["shipping", "delivery", "package"],
            "billing": ["invoice", "charge", "billing"],
        },
    )
    payload = _read_payload_strip_token(path)
    # The core privacy claim, checked directly on the wire payload
    # BEFORE it even reaches the sanitizer: no raw text anywhere.
    payload_str = json.dumps(payload)
    assert "package was delayed" not in payload_str
    assert "invoice charge" not in payload_str

    res = sanitize(payload)
    assert res.ok, f"sanitizer rejected: {res.rejection_reason}"
    assert res.analysis_type == "text_extraction"
    assert res.sanitized["categories"]["shipping"] == 20
    assert res.sanitized["categories"]["billing"] == 10
    assert res.sanitized["n"] == 30
    assert "overall_sentiment_mean" in res.sanitized


def test_from_text_extract_missing_column_raises(runtime) -> None:
    mod, _path = runtime
    df = pd.DataFrame({"other": ["x"]})
    with pytest.raises(ValueError):
        mod.from_text_extract(df, "feedback", categories={"a": ["x"]})


def test_from_text_extract_empty_categories_raises(runtime) -> None:
    mod, _path = runtime
    df = pd.DataFrame({"feedback": ["hello"]})
    with pytest.raises(ValueError):
        mod.from_text_extract(df, "feedback", categories={})


def test_from_text_extract_counts_missing_values(runtime) -> None:
    mod, path = runtime
    df = pd.DataFrame({"feedback": ["great service"] * 12 + [None] * 3})
    mod.from_text_extract(df, "feedback", categories={"a": ["great"]})
    payload = _read_payload_strip_token(path)
    assert payload["n"] == 12
    assert payload["missing_count"] == 3


# ---------------------------------------------------------------------------
# 3. Sanitizer-level security properties (direct payload construction,
#    no runtime helper — exercises the suppression/alignment logic
#    against adversarial and edge-case shapes)
# ---------------------------------------------------------------------------

def _well_formed(**overrides) -> dict:
    p = {
        "type": "text_extraction",
        "text_column": "feedback",
        "categories": {"shipping": 40, "billing": 25},
        "category_sentiment": {"shipping": -0.4, "billing": 0.1},
        "n": 65,
        "missing_count": 0,
    }
    p.update(overrides)
    return p


def test_well_formed_payload_passes() -> None:
    r = sanitize(_well_formed())
    assert r.ok, r.rejection_reason
    assert r.sanitized["categories"] == {"shipping": 40, "billing": 25}
    assert r.sanitized["category_sentiment"] == {"shipping": -0.4, "billing": 0.1}


def test_missing_required_field_rejected() -> None:
    p = _well_formed()
    del p["categories"]
    r = sanitize(p)
    assert not r.ok
    assert "missing required" in (r.rejection_reason or "")


def test_categories_must_be_nonempty_dict() -> None:
    r = sanitize(_well_formed(categories={}))
    assert not r.ok


def test_negative_count_rejected() -> None:
    r = sanitize(_well_formed(categories={"a": -1}))
    assert not r.ok


def test_non_int_count_rejected() -> None:
    r = sanitize(_well_formed(categories={"a": 4.5}))
    assert not r.ok


def test_too_many_categories_rejected() -> None:
    huge = {f"cat{i}": 20 for i in range(60)}
    r = sanitize(_well_formed(categories=huge, category_sentiment={}))
    assert not r.ok
    assert "structural cap" in (r.rejection_reason or "")


def test_suppressed_category_has_count_and_sentiment_both_withheld() -> None:
    """THE core security property. Two rare categories (3 and 2),
    both below the default threshold of 10 -- both counts AND both
    sentiment scores must vanish from the sanitized output, replaced
    by the anonymous [suppressed] bucket. Two primary-suppressed
    cells means ``enforce_back_calc_safety`` no-ops (its secondary
    rule only fires for exactly ONE primary-suppressed cell), so this
    isolates primary suppression's effect on the sentiment map from
    secondary's — that combination is covered separately below."""
    p = _well_formed(
        categories={"shipping": 40, "rare_a": 3, "rare_b": 2},
        category_sentiment={
            "shipping": -0.4, "rare_a": -0.9, "rare_b": 0.7,
        },
        n=45,
    )
    r = sanitize(p)
    assert r.ok, r.rejection_reason
    assert "rare_a" not in r.sanitized["categories"]
    assert "rare_b" not in r.sanitized["categories"]
    assert "rare_a" not in r.sanitized["category_sentiment"]
    assert "rare_b" not in r.sanitized["category_sentiment"]
    assert "[suppressed]" in r.sanitized["categories"]
    assert r.sanitized["suppressed_cell_count"] == 2
    # The surviving category keeps its sentiment.
    assert r.sanitized["category_sentiment"] == {"shipping": -0.4}


def test_publishing_total_n_triggers_secondary_suppression_too() -> None:
    """Same setup as above but WITH n present: exactly one primary-
    suppressed category plus a published total is back-calculable, so
    the sanitizer also sacrifices the smallest surviving category
    (billing, 25 < shipping's 40) -- same back-calc-safety rule
    frequency_table follows. Its sentiment must vanish right along
    with its count."""
    p = _well_formed(
        categories={"shipping": 40, "billing": 25, "rare_complaint": 3},
        category_sentiment={
            "shipping": -0.4, "billing": 0.1, "rare_complaint": -0.9,
        },
        n=68,
    )
    r = sanitize(p)
    assert r.ok, r.rejection_reason
    assert r.sanitized["suppressed_cell_count"] == 2
    assert r.sanitized["categories"] == {"shipping": 40, "[suppressed]": "<10"}
    assert r.sanitized["category_sentiment"] == {"shipping": -0.4}


def test_sentiment_for_category_not_in_counts_is_dropped() -> None:
    """An adversarial/malformed payload where category_sentiment names
    a category that doesn't even appear in categories -- must not
    survive into the output (there's nothing for it to attach to,
    and it can't be used to smuggle an extra signal past the
    suppression gate)."""
    p = _well_formed(
        categories={"shipping": 40},
        category_sentiment={"shipping": -0.4, "ghost_category": 0.9},
    )
    r = sanitize(p)
    assert r.ok, r.rejection_reason
    assert "ghost_category" not in r.sanitized["category_sentiment"]


def test_sentiment_values_clamped_to_valid_range() -> None:
    p = _well_formed(category_sentiment={"shipping": 5.0, "billing": -7.0})
    r = sanitize(p)
    assert r.ok, r.rejection_reason
    assert r.sanitized["category_sentiment"]["shipping"] == 1.0
    assert r.sanitized["category_sentiment"]["billing"] == -1.0


def test_non_finite_sentiment_dropped_not_rejected() -> None:
    p = _well_formed(category_sentiment={"shipping": float("nan"), "billing": 0.2})
    r = sanitize(p)
    assert r.ok, r.rejection_reason
    assert "shipping" not in r.sanitized["category_sentiment"]
    assert r.sanitized["category_sentiment"]["billing"] == 0.2


def test_overall_sentiment_dropped_below_min_n() -> None:
    p = _well_formed(
        categories={"shipping": 15},
        category_sentiment={"shipping": -0.4},
        n=8,  # below min_n_descriptive (10)
        overall_sentiment_mean=-0.4,
    )
    r = sanitize(p)
    assert r.ok, r.rejection_reason
    assert "overall_sentiment_mean" not in r.sanitized


def test_overall_sentiment_present_when_n_adequate() -> None:
    p = _well_formed(
        categories={"shipping": 40, "billing": 25},
        n=65,
        overall_sentiment_mean=-0.15,
    )
    r = sanitize(p)
    assert r.ok, r.rejection_reason
    assert r.sanitized["overall_sentiment_mean"] == -0.15


def test_single_suppressed_cell_strips_total_n() -> None:
    """The true degenerate case: a SINGLE category, below threshold,
    with a total published. There's no second cell available to
    sacrifice for secondary suppression, so the only way to prevent
    back-calculating the suppressed value from the margin is to drop
    the margin itself -- n and missing_count are stripped."""
    p = _well_formed(
        categories={"rare": 3},
        category_sentiment={"rare": 0.1},
        n=3,
    )
    r = sanitize(p)
    assert r.ok, r.rejection_reason
    assert "n" not in r.sanitized
    assert "missing_count" not in r.sanitized
    assert r.sanitized["categories"] == {"[suppressed]": "<10"}
    assert r.sanitized["category_sentiment"] == {}


def test_colliding_category_names_rejected() -> None:
    p = _well_formed(categories={"a\nb": 20, "a b": 15})
    r = sanitize(p)
    assert not r.ok
    assert "collid" in (r.rejection_reason or "").lower()


def test_small_missing_count_coarsened() -> None:
    p = _well_formed(missing_count=3)
    r = sanitize(p)
    assert r.ok, r.rejection_reason
    assert isinstance(r.sanitized["missing_count"], str)


def test_unknown_type_still_rejects_as_before() -> None:
    """Registering text_extraction must not change unknown-type
    handling for anything else."""
    r = sanitize({"type": "not_a_real_type"})
    assert not r.ok


def test_supported_types_includes_text_extraction() -> None:
    from sift.sanitizer import supported_types
    assert "text_extraction" in supported_types()


# ---------------------------------------------------------------------------
# 4. End-to-end through submit_script (real sandboxed executor)
# ---------------------------------------------------------------------------

def _python_ready() -> bool:
    from sift.env_detect import detect_environment
    e = detect_environment()
    if e.python is None or not e.has_sandbox_backend():
        return False
    return not ({"pandas", "numpy"} & set(e.python.missing_packages))


_skip_no_python = pytest.mark.skipif(
    not _python_ready(),
    reason="needs python3 + pandas + numpy + a sandbox backend (sandbox-exec or bwrap)",
)


@_skip_no_python
def test_submit_script_text_extraction_end_to_end(tmp_path: Path) -> None:
    """Real subprocess execution: a script builds a free-text
    DataFrame, calls from_text_extract, and the tool response +
    stored row must never carry a single raw sentence -- only the
    aggregated counts and floats the sanitizer allows."""
    import asyncio
    import json as _json

    from sift.config import set_cwd
    from sift.store import get_store, reset_store_for_tests
    from sift.tools import submit_script

    set_cwd(tmp_path)
    reset_store_for_tests()

    code = (
        "import pandas as pd\n"
        "import sift\n"
        "texts = (\n"
        "    ['the delivery was late and the package was damaged'] * 15\n"
        "    + ['shipping was fine, no complaints here'] * 5\n"
        "    + ['the invoice was wrong and support was unhelpful'] * 12\n"
        ")\n"
        "df = pd.DataFrame({'feedback': texts})\n"
        "sift.from_text_extract(\n"
        "    df, 'feedback',\n"
        "    categories={\n"
        "        'shipping': ['delivery', 'shipping', 'package'],\n"
        "        'billing': ['invoice', 'support'],\n"
        "    },\n"
        ")\n"
    )
    response = asyncio.run(submit_script.handler({
        "language": "Python",
        "code": code,
        "label": "customer feedback triage",
        "source_dataset": "",
    }))
    text_block = next(b for b in response["content"] if b.get("type") == "text")
    body = _json.loads(text_block["text"])
    assert body["status"] == "ok", body

    result_id = body["results"][0]["result_id"]
    store = get_store(tmp_path)
    row = store.get(result_id)
    assert row is not None
    assert row.analysis_type == "text_extraction"

    # The privacy claim, checked on every surface this row touches:
    # the tool response envelope, the stored sanitized payload, and
    # the script_code column (which legitimately DOES carry the raw
    # sentences, since that's the researcher's own source code kept
    # for audit -- but the SANITIZED PAYLOAD specifically must not).
    payload_str = _json.dumps(row.sanitized_payload)
    assert "delivery was late" not in payload_str
    assert "package was damaged" not in payload_str
    assert "invoice was wrong" not in payload_str
    response_str = _json.dumps(body)
    assert "delivery was late" not in response_str

    assert row.sanitized_payload["categories"]["shipping"] == 20
    assert row.sanitized_payload["categories"]["billing"] == 12


@_skip_no_python
def test_submit_script_text_extraction_suppresses_below_threshold(tmp_path: Path) -> None:
    """A taxonomy where every category is below the suppression
    threshold still comes back ``ok`` (suppression is a soft
    transformation, not a rejection -- same as frequency_table), but
    BOTH category names and BOTH sentiment scores must be gone,
    replaced by the anonymous bucket -- proving the sanitizer gate
    actually engages on this shape through the real pipeline, not
    just in the direct-payload unit tests above."""
    import asyncio
    import json as _json

    from sift.config import set_cwd
    from sift.store import get_store, reset_store_for_tests
    from sift.tools import submit_script

    set_cwd(tmp_path)
    reset_store_for_tests()

    code = (
        "import pandas as pd\n"
        "import sift\n"
        "texts = ['a great review'] * 3 + ['a terrible review'] * 2\n"
        "df = pd.DataFrame({'feedback': texts})\n"
        "sift.from_text_extract(\n"
        "    df, 'feedback',\n"
        "    categories={'positive': ['great'], 'negative': ['terrible']},\n"
        ")\n"
    )
    response = asyncio.run(submit_script.handler({
        "language": "Python",
        "code": code,
        "label": "tiny sample canary",
        "source_dataset": "",
    }))
    text_block = next(b for b in response["content"] if b.get("type") == "text")
    body = _json.loads(text_block["text"])
    assert body["status"] == "ok", body

    result_id = body["results"][0]["result_id"]
    row = get_store(tmp_path).get(result_id)
    assert row is not None
    assert "positive" not in row.sanitized_payload["categories"]
    assert "negative" not in row.sanitized_payload["categories"]
    assert row.sanitized_payload["categories"] == {"[suppressed]": "<10"}
    assert row.sanitized_payload["category_sentiment"] == {}
