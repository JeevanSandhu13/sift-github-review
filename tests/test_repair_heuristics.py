"""Pure unit tests for ``sift.repair_heuristics.normalize_gremlins``.

No subprocess, no async — string in, string out.

End-to-end coverage (actually running a repaired script through
``submit_script``) lives in ``tests/test_submit_script_local_repair.py``.

Coverage includes the full quotation-mark family (curly quotes + guillemets +
fullwidth quotes), fullwidth ASCII punctuation (CJK input-method
artifacts), the complete zero-width/format/bidi-control character set
(a superset of the original 5-character allowlist, including the
"Trojan Source" bidi-override family), and the complete Unicode
Space_Separator category (a superset of plain NBSP). Every new
character table was generated against ``unicodedata`` rather than
hand-transcribed — the tests below cross-check that generation
directly rather than trusting it.
"""

from __future__ import annotations

import unicodedata

import pytest

from sift.repair_heuristics import (
    _FULLWIDTH_PUNCTUATION,
    _SMART_QUOTES,
    _UNICODE_SPACE_CODEPOINTS,
    _UNICODE_SPACES,
    _ZERO_WIDTH_RE,
    normalize_gremlins,
)


def test_clean_code_is_untouched() -> None:
    code = "import pandas as pd\ndf = pd.read_csv('x.csv')\n"
    r = normalize_gremlins(code)
    assert r.changed is False
    assert r.code == code
    assert r.descriptions == []


def test_empty_string_is_untouched() -> None:
    r = normalize_gremlins("")
    assert r.changed is False
    assert r.code == ""


def test_curly_single_quotes_replaced() -> None:
    r = normalize_gremlins("x <- \u2018hello\u2019")
    assert r.changed is True
    assert r.code == "x <- 'hello'"
    assert any("quotation mark" in d for d in r.descriptions)


def test_curly_double_quotes_replaced() -> None:
    r = normalize_gremlins("x <- \u201Chello\u201D")
    assert r.changed is True
    assert r.code == 'x <- "hello"'


def test_low9_and_reversed_variants_replaced() -> None:
    r = normalize_gremlins("\u201A\u201B\u201E\u201F\u2032\u2033")
    assert r.changed is True
    assert r.code == "''\"\"'\""


def test_zero_width_characters_stripped_not_replaced() -> None:
    code = "x\u200B <- 1\u200C\u200D\u2060"
    r = normalize_gremlins(code)
    assert r.changed is True
    assert r.code == "x <- 1"
    assert any("zero-width" in d for d in r.descriptions)


def test_bom_stripped_when_leading() -> None:
    code = "\uFEFFimport pandas as pd\n"
    r = normalize_gremlins(code)
    assert r.changed is True
    assert r.code == "import pandas as pd\n"


def test_nbsp_replaced_with_regular_space() -> None:
    code = "x <- 1\u00A02"
    r = normalize_gremlins(code)
    assert r.changed is True
    assert r.code == "x <- 1 2"
    assert any("non-standard Unicode space" in d for d in r.descriptions)


def test_mixed_gremlins_all_reported() -> None:
    code = "x <- \u2018a\u2019\u00A0\u200B"
    r = normalize_gremlins(code)
    assert r.changed is True
    assert r.code == "x <- 'a' "
    joined = " | ".join(r.descriptions)
    assert "quotation mark" in joined
    assert "non-standard Unicode space" in joined
    assert "zero-width" in joined


def test_counts_are_accurate() -> None:
    r = normalize_gremlins("\u2018\u2018\u2018")
    assert r.changed is True
    assert "replaced 3 curly/typographic/fullwidth quotation marks" in r.descriptions[0]


def test_singular_wording_for_count_one() -> None:
    r = normalize_gremlins("\u2018")
    assert "replaced 1 curly/typographic/fullwidth quotation mark " in r.descriptions[0]
    assert "marks" not in r.descriptions[0]


def test_does_not_touch_intentional_ascii_content() -> None:
    # A script that legitimately uses straight quotes and regular
    # spaces throughout must come back byte-identical.
    code = (
        "import sift\n"
        "sift.from_summarize('income', n=100, mean=50000.0, sd=1200.0, "
        "missing_count=0)\n"
    )
    r = normalize_gremlins(code)
    assert r.changed is False
    assert r.code is code  # same object, not just equal


# ---------------------------------------------------------------------------
# Guillemets and fullwidth quote marks (quotation-mark family expansion)
# ---------------------------------------------------------------------------


def test_guillemets_replaced_as_double_quotes() -> None:
    """« » -- the French/German locale "smart quote" autocorrect
    substitution some word processors default to."""
    r = normalize_gremlins("x <- \u00ABhello\u00BB")
    assert r.changed is True
    assert r.code == 'x <- "hello"'


def test_single_angle_quotes_replaced_as_single_quotes() -> None:
    r = normalize_gremlins("x <- \u2039hi\u203A")
    assert r.changed is True
    assert r.code == "x <- 'hi'"


def test_fullwidth_quote_marks_replaced() -> None:
    """＂ / ＇ -- CJK IME default punctuation-width artifacts."""
    r = normalize_gremlins("x <- \uFF02hello\uFF07")
    assert r.changed is True
    assert r.code == 'x <- "hello\''
    assert any("quotation mark" in d for d in r.descriptions)


# ---------------------------------------------------------------------------
# Fullwidth ASCII punctuation
# ---------------------------------------------------------------------------


def test_fullwidth_parentheses_replaced() -> None:
    """The single highest-value fullwidth-punctuation case: a CJK IME
    left in fullwidth mode breaks every function call."""
    r = normalize_gremlins("print\uFF08x\uFF09")
    assert r.changed is True
    assert r.code == "print(x)"
    assert any("fullwidth ASCII symbol" in d for d in r.descriptions)


def test_fullwidth_brackets_and_braces_replaced() -> None:
    r = normalize_gremlins("x\uFF3B1\uFF3D <- \uFF5B1, 2\uFF5D")
    assert r.changed is True
    assert r.code == "x[1] <- {1, 2}"


def test_fullwidth_operators_replaced() -> None:
    r = normalize_gremlins("x \uFF1D 1 \uFF0B 2 \uFF0A 3")
    assert r.changed is True
    assert r.code == "x = 1 + 2 * 3"


def test_fullwidth_comma_and_semicolon_replaced() -> None:
    r = normalize_gremlins("f(a\uFF0C b)\uFF1B")
    assert r.changed is True
    assert r.code == "f(a, b);"


def test_fullwidth_backslash_replaced() -> None:
    """Reverse solidus is a two-character JSON/regex-adjacent escape
    hazard if mishandled -- confirm it maps to a single ASCII
    backslash, not something doubled or dropped."""
    r = normalize_gremlins("path <- \"C:\uFF3CUsers\"")
    assert r.changed is True
    assert r.code == 'path <- "C:\\Users"'


def test_fullwidth_letters_and_digits_are_not_touched() -> None:
    """Deliberately out of scope -- see the module docstring's point
    2: fullwidth punctuation mid-syntax is almost always an input-
    method accident, but fullwidth letters/digits inside a label are
    much more plausibly intentional content a researcher wants
    preserved verbatim (e.g. a Japanese full-width numeral in a
    label). Only punctuation/symbols are normalized."""
    code = "label <- \uFF21\uFF22\uFF23"  # fullwidth "ABC"
    r = normalize_gremlins(code)
    assert r.changed is False
    assert r.code == code


# ---------------------------------------------------------------------------
# Extended zero-width / bidi-control / invisible-format characters
# ---------------------------------------------------------------------------


def test_soft_hyphen_stripped() -> None:
    """U+00AD -- invisible except at a line break; a notorious
    artifact of copying justified text from a web page or PDF."""
    r = normalize_gremlins("in\u00ADcome <- 1")
    assert r.changed is True
    assert r.code == "income <- 1"


def test_bidi_embedding_and_override_controls_stripped() -> None:
    r = normalize_gremlins("x\u202A\u202B\u202C\u202D\u202E <- 1")
    assert r.changed is True
    assert r.code == "x <- 1"


def test_bidi_override_trojan_source_scenario_stripped() -> None:
    """The actual "Trojan Source" (CVE-2021-42574) attack shape:
    RIGHT-TO-LEFT OVERRIDE can make displayed token order diverge
    from parsed token order. Sift strips it before the script ever
    reaches an interpreter -- verified here by confirming the
    character is gone and the surrounding tokens are untouched, which
    is what closes the display/execution divergence regardless of
    whether the character arrived by accident or by design."""
    code = "safe = True \u202Eexec(x)\u202C # comment"
    r = normalize_gremlins(code)
    assert r.changed is True
    assert "\u202E" not in r.code
    assert "\u202C" not in r.code
    assert "exec(x)" in r.code  # the visible tokens survive untouched


def test_left_to_right_and_right_to_left_marks_stripped() -> None:
    r = normalize_gremlins("x\u200E <- 1\u200F")
    assert r.changed is True
    assert r.code == "x <- 1"


def test_bidi_isolate_controls_stripped() -> None:
    r = normalize_gremlins("x\u2066\u2067\u2068\u2069 <- 1")
    assert r.changed is True
    assert r.code == "x <- 1"


def test_invisible_math_operators_stripped() -> None:
    """Leak from copy-pasting a Word/MathML equation into a script --
    e.g. an invisible-times character between two coefficients."""
    r = normalize_gremlins("y <- 2\u2062x + 1")
    assert r.changed is True
    assert r.code == "y <- 2x + 1"


def test_variation_selectors_stripped() -> None:
    r = normalize_gremlins("x\uFE00 <- 1\uFE0F")
    assert r.changed is True
    assert r.code == "x <- 1"


def test_astral_variation_selector_supplement_stripped() -> None:
    r = normalize_gremlins("x\U000E0100 <- 1")
    assert r.changed is True
    assert r.code == "x <- 1"


def test_astral_tag_characters_stripped() -> None:
    r = normalize_gremlins("x\U000E0001\U000E007F <- 1")
    assert r.changed is True
    assert r.code == "x <- 1"


# ---------------------------------------------------------------------------
# Extended Unicode space separators
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cp", list(_UNICODE_SPACE_CODEPOINTS))
def test_every_unicode_space_codepoint_replaced_with_ascii_space(cp: int) -> None:
    ch = chr(cp)
    r = normalize_gremlins(f"x{ch}<-{ch}1")
    assert r.changed is True
    assert r.code == "x <- 1"


def test_em_space_and_ideographic_space_replaced() -> None:
    r = normalize_gremlins("x\u2003<-\u30001")
    assert r.changed is True
    assert r.code == "x <- 1"


# ---------------------------------------------------------------------------
# Table generation correctness -- cross-check against unicodedata
# ---------------------------------------------------------------------------


def test_unicode_space_codepoints_exactly_match_zs_category() -> None:
    """``_UNICODE_SPACE_CODEPOINTS`` must be EXACTLY the Unicode "Zs"
    (Space_Separator) category minus plain ASCII space -- neither
    missing an entry (a real gremlin the module would then fail to
    catch) nor including something that isn't actually in the
    category (scope creep into characters this module never audited).
    Scans the same range this file's own generation script did."""
    actual_zs = {
        cp for cp in range(0x0, 0x30000)
        if unicodedata.category(chr(cp)) == "Zs" and cp != 0x20
    }
    assert set(_UNICODE_SPACE_CODEPOINTS) == actual_zs


def test_fullwidth_punctuation_table_matches_generation_formula() -> None:
    """Every entry must satisfy ``chr(cp - 0xFEE0)`` for its ASCII
    target, and the table must contain EVERY fullwidth punctuation/
    symbol codepoint in U+FF01-FF5E except the ones intentionally
    excluded (letters, digits, and the two quote marks, which live in
    ``_SMART_QUOTES`` instead)."""
    letters_digits = set(
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    )
    quote_targets = {'"', "'"}
    expected: dict[str, str] = {}
    for cp in range(0xFF01, 0xFF5F):
        ascii_ch = chr(cp - 0xFEE0)
        if ascii_ch in letters_digits or ascii_ch in quote_targets:
            continue
        expected[chr(cp)] = ascii_ch
    assert _FULLWIDTH_PUNCTUATION == expected


def test_fullwidth_quote_marks_present_in_smart_quotes_table() -> None:
    """The two fullwidth codepoints excluded from the punctuation
    table above (quote marks) must actually be handled -- by
    ``_SMART_QUOTES`` instead, not silently dropped from both."""
    assert _SMART_QUOTES["\uFF02"] == '"'
    assert _SMART_QUOTES["\uFF07"] == "'"


def test_zero_width_regex_matches_every_documented_range() -> None:
    """Exhaustively checks every codepoint the module docstring
    claims is covered, one by one, against the actual compiled
    regex -- catches an off-by-one range boundary that a handful of
    spot-check tests above could miss."""
    expected_ranges = [
        range(0x200B, 0x2010),   # 200B-200F
        [0xFEFF],
        [0x2060],
        [0x00AD],
        range(0x202A, 0x202F),   # 202A-202E
        range(0x2066, 0x206A),   # 2066-2069
        range(0x2061, 0x2065),   # 2061-2064
        range(0xFE00, 0xFE10),   # FE00-FE0F
        range(0xE0100, 0xE01F0),  # E0100-E01EF
        range(0xE0000, 0xE0080),  # E0000-E007F
    ]
    for rng in expected_ranges:
        for cp in rng:
            assert _ZERO_WIDTH_RE.match(chr(cp)), f"U+{cp:04X} not matched"


def test_zero_width_regex_does_not_match_ordinary_characters() -> None:
    """Negative control: ordinary letters, digits, ASCII punctuation,
    NBSP (handled by the SEPARATE space table, not this one), and an
    en-dash (deliberately out of scope -- see below) must all be
    rejected by the zero-width class."""
    for ch in ("a", "Z", "1", " ", "-", "\u00A0", "\u2013", "\u2014"):
        assert not _ZERO_WIDTH_RE.match(ch), repr(ch)


# ---------------------------------------------------------------------------
# Deliberately out of scope -- documents what this module does NOT touch
# ---------------------------------------------------------------------------


def test_en_and_em_dashes_are_not_normalized() -> None:
    """En/em dashes are a well-known autocorrect gremlin too (a
    hyphen-minus operator silently becomes an en-dash), but unlike a
    quote-glyph swap, a dash INSIDE a string literal can carry
    genuinely different intended content (a label like "2020\u20132021"
    using an en-dash on purpose) -- normalizing it would change label
    content more substantively than any character this module DOES
    touch. Deliberately excluded; the researcher/model must fix a
    dash-as-operator typo explicitly rather than have it silently
    rewritten."""
    code = "x <- 1 \u2013 2"  # en-dash, NOT a hyphen-minus
    r = normalize_gremlins(code)
    assert r.changed is False
    assert r.code == code


def test_fullwidth_letters_never_touched_even_mixed_with_punctuation() -> None:
    """A mixed fullwidth string (punctuation + letters) only has its
    punctuation normalized; the letters ride through untouched."""
    r = normalize_gremlins("x \uFF1D \uFF21\uFF22\uFF23")  # x ＝ ＡＢＣ
    assert r.changed is True
    assert r.code == "x = \uFF21\uFF22\uFF23"


# ---------------------------------------------------------------------------
# Combined / realistic scenarios
# ---------------------------------------------------------------------------


def test_realistic_llm_copy_paste_scenario() -> None:
    """A plausible realistic failure: a model-authored R script that
    picked up curly quotes AND a non-breaking space from being
    rendered through markdown before being pasted back into the
    script text."""
    code = "df$income\u00A0<- \u2018clean\u2019"
    r = normalize_gremlins(code)
    assert r.changed is True
    assert r.code == "df$income <- 'clean'"


def test_realistic_cjk_ime_copy_paste_scenario() -> None:
    """A plausible realistic failure for a researcher working with a
    CJK input method left in fullwidth mode: fullwidth parens and
    comma break a Python function call that otherwise looks completely
    normal in the chat transcript."""
    code = "pd.read_csv\uFF08\uFF02data.csv\uFF02\uFF0C sep\uFF1D\uFF02,\uFF02\uFF09"
    r = normalize_gremlins(code)
    assert r.changed is True
    assert r.code == 'pd.read_csv("data.csv", sep=",")'


def test_all_four_categories_in_one_script() -> None:
    code = (
        "x\uFF08\u2018a\u2019, 1\u00A02\uFF09\u200B"
    )
    r = normalize_gremlins(code)
    assert r.changed is True
    assert r.code == "x('a', 1 2)"
    assert len(r.descriptions) == 4
