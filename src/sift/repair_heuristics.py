"""Sift -- deterministic local repair heuristics for failed scripts.

This module repairs a narrow class of failed scripts without spending another
provider round trip. A general repair model would have to guess at the
researcher's intent well enough to rewrite their analysis code. A statistical
analysis tool that silently rewrites a researcher's script based on a model's
best guess is a correctness and trust hazard: a wrong guess can change what was
actually estimated without anyone noticing.

What this module does instead is narrower and does not guess: it
detects and corrects a fixed allowlist of invisible, zero-width, or
typographic-equivalent characters that are never analytically
meaningful in R / Stata / Python source -- smart quotes, guillemets,
fullwidth CJK-input-method punctuation, zero-width characters, bidi
control characters, non-standard Unicode space separators, a leading
BOM -- and that overwhelmingly enter a script as a copy-paste artifact
from rendered markdown, rich text, a CJK input method, or a Word
equation editor, rather than as anything the model or researcher
intended. These are the single most common class of "the script LOOKS
right but the interpreter chokes on it" failures for LLM-generated
code, because every character in this module's allowlist is either
completely invisible or visually near-identical to the ASCII character
it replaces -- indistinguishable in a diff or a chat transcript.

Why this is safe to apply automatically:

  1. **The substitution set is fixed and exhaustive**, not learned or
     inferred. Every character this module ever touches is listed
     below; there's no probability, no "best guess", nothing that
     could produce a different edit on a different day.
  2. **The transformation never changes analytical meaning.** A
     straight quote and a curly quote are the same character to a
     human reader; a fullwidth comma and an ASCII comma are the same
     punctuation mark in two different Unicode compatibility widths
     (Unicode's own NFKC normalization treats them as canonically
     equivalent); a non-breaking space and a regular space render
     identically; every character in the zero-width set renders as
     literally nothing. This is normalization, not rewriting -- it can
     change what LABEL or STRING CONTENT looks like in a very minor
     typographic sense, but never a variable name, a formula, a
     comparison operator, or a statistical choice.
  3. **The repair is verified, not assumed.** The caller
     (``tools.submit_script``) only keeps a repaired script if
     actually RE-RUNNING it locally succeeds. If it doesn't fix the
     failure, the repair is discarded entirely and the researcher's
     original script -- and its original error -- is exactly what the
     model sees, unchanged.
  4. **It is always disclosed.** When a repair IS kept, the tool
     response says so explicitly (see ``tools.py``'s
     ``local_repair`` field): what changed, in plain language, and
     that the corrected script -- not the original submission -- is
     what actually produced the results. Nothing here is silent.
  5. **It only ever fires on failure**, and only when the script
     text actually contains one of the flagged characters -- a clean
     script that runs on the first try, or fails for an unrelated
     reason, costs nothing extra. There is no loop: at most one extra
     local subprocess run per ``submit_script`` call.

Security-adjacent bonus, not the primary motivation: the bidi-control
characters in the zero-width set (U+202A-202E, U+2066-2069) are the
exact character family behind the "Trojan Source" class of attack
(CVE-2021-42574) -- source text that RENDERS one way in an editor or
chat transcript while the interpreter parses a different token order.
Stripping them from a script Sift is about to execute is a real,
independent hardening, not just a convenience -- it removes a vector
for what's displayed in the chat UI to diverge from what actually
runs, regardless of whether that divergence would have been
accidental or deliberate.

This module itself is pure text transformation -- no subprocess, no
I/O, no async. That keeps it trivially unit-testable (string in,
string out) and keeps the execution / retry orchestration where it
already lives, in ``tools.submit_script``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Quotation-mark family -- curly quotes, angle quotes (guillemets), and
# fullwidth quote marks all substitute for the ASCII ' / " characters
# R / Stata / Python actually parse as string delimiters. A rich-text
# editor's "smart quotes" autocorrect is the most common source for
# the curly-quote entries; guillemets are the equivalent autocorrect
# behavior in some European locale settings (French/German word
# processors commonly default to angle quotes for smart quotes);
# fullwidth quote marks come from CJK input methods, where the IME's
# default punctuation mode renders ASCII-equivalent symbols at double
# width. All map to their straight-ASCII equivalents.
# ---------------------------------------------------------------------------
_SMART_QUOTES: dict[str, str] = {
    "\u2018": "\'",   # LEFT SINGLE QUOTATION MARK
    "\u2019": "\'",   # RIGHT SINGLE QUOTATION MARK
    "\u201A": "\'",   # SINGLE LOW-9 QUOTATION MARK
    "\u201B": "\'",   # SINGLE HIGH-REVERSED-9 QUOTATION MARK
    "\u201C": '"',   # LEFT DOUBLE QUOTATION MARK
    "\u201D": '"',   # RIGHT DOUBLE QUOTATION MARK
    "\u201E": '"',   # DOUBLE LOW-9 QUOTATION MARK
    "\u201F": '"',   # DOUBLE HIGH-REVERSED-9 QUOTATION MARK
    "\u2032": "\'",   # PRIME
    "\u2033": '"',   # DOUBLE PRIME
    "\u00AB": '"',   # LEFT-POINTING DOUBLE ANGLE QUOTATION MARK (guillemet)
    "\u00BB": '"',   # RIGHT-POINTING DOUBLE ANGLE QUOTATION MARK (guillemet)
    "\u2039": "\'",   # SINGLE LEFT-POINTING ANGLE QUOTATION MARK
    "\u203A": "\'",   # SINGLE RIGHT-POINTING ANGLE QUOTATION MARK
    "\uFF02": '"',   # FULLWIDTH QUOTATION MARK
    "\uFF07": "\'",   # FULLWIDTH APOSTROPHE
}

# ---------------------------------------------------------------------------
# Fullwidth ASCII symbols -- the rest of the U+FF01-FF5E "Fullwidth
# Forms" compatibility block (excluding letters and digits, which are
# deliberately left untouched -- see the module docstring's point 2:
# fullwidth punctuation landing mid-syntax is virtually always a CJK
# input-method accident, but fullwidth LETTERS or DIGITS inside a
# label are much more plausibly intentional content a researcher
# wants preserved verbatim). Parentheses, brackets, and braces are the
# single highest-value entries here -- a fullwidth paren typed by an
# IME left in its default (fullwidth) punctuation mode breaks every
# function call, index, and block delimiter in R / Stata / Python
# alike, and looks completely unremarkable in a chat transcript.
# Every key below is written as an explicit \\uXXXX escape (never a
# pasted glyph) specifically because these characters are visually
# similar to each other at a glance -- an escape sequence is
# unambiguous on review in a way a row of near-identical fullwidth
# glyphs is not. Cross-checked against
# ``chr(cp - 0xFEE0) for cp in range(0xFF01, 0xFF5F)`` via
# ``unicodedata`` rather than hand-transcribed.
# ---------------------------------------------------------------------------
_FULLWIDTH_PUNCTUATION: dict[str, str] = {
    "\uFF01": "!",   # FULLWIDTH EXCLAMATION MARK
    "\uFF03": "#",   # FULLWIDTH NUMBER SIGN
    "\uFF04": "$",   # FULLWIDTH DOLLAR SIGN
    "\uFF05": "%",   # FULLWIDTH PERCENT SIGN
    "\uFF06": "&",   # FULLWIDTH AMPERSAND
    "\uFF08": "(",   # FULLWIDTH LEFT PARENTHESIS
    "\uFF09": ")",   # FULLWIDTH RIGHT PARENTHESIS
    "\uFF0A": "*",   # FULLWIDTH ASTERISK
    "\uFF0B": "+",   # FULLWIDTH PLUS SIGN
    "\uFF0C": ",",   # FULLWIDTH COMMA
    "\uFF0D": "-",   # FULLWIDTH HYPHEN-MINUS
    "\uFF0E": ".",   # FULLWIDTH FULL STOP
    "\uFF0F": "/",   # FULLWIDTH SOLIDUS
    "\uFF1A": ":",   # FULLWIDTH COLON
    "\uFF1B": ";",   # FULLWIDTH SEMICOLON
    "\uFF1C": "<",   # FULLWIDTH LESS-THAN SIGN
    "\uFF1D": "=",   # FULLWIDTH EQUALS SIGN
    "\uFF1E": ">",   # FULLWIDTH GREATER-THAN SIGN
    "\uFF1F": "?",   # FULLWIDTH QUESTION MARK
    "\uFF20": "@",   # FULLWIDTH COMMERCIAL AT
    "\uFF3B": "[",   # FULLWIDTH LEFT SQUARE BRACKET
    "\uFF3C": "\\",  # FULLWIDTH REVERSE SOLIDUS
    "\uFF3D": "]",   # FULLWIDTH RIGHT SQUARE BRACKET
    "\uFF3E": "^",   # FULLWIDTH CIRCUMFLEX ACCENT
    "\uFF3F": "_",   # FULLWIDTH LOW LINE
    "\uFF40": "`",   # FULLWIDTH GRAVE ACCENT
    "\uFF5B": "{",   # FULLWIDTH LEFT CURLY BRACKET
    "\uFF5C": "|",   # FULLWIDTH VERTICAL LINE
    "\uFF5D": "}",   # FULLWIDTH RIGHT CURLY BRACKET
    "\uFF5E": "~",   # FULLWIDTH TILDE
}

# ---------------------------------------------------------------------------
# Zero-width / invisible / format characters -- render as NOTHING at
# all in any editor, chat transcript, or diff, but break tokenizers
# when they land inside an identifier or between tokens. Stripped
# entirely (never replaced with a visible character, since they carry
# no width to begin with).
#
# Built entirely from \\uXXXX / \\UXXXXXXXX escape sequences -- never a
# pasted invisible glyph -- because these characters are, by
# definition, indistinguishable from nothing on a screen; a corrupted
# or silently-duplicated invisible glyph in source code would be
# impossible to catch on review. Each range is verified against
# ``unicodedata`` in the module's own test suite.
#
# Deliberately a superset of ``text_safety.py``'s ``_CONTROL_AND_
# TRICKS`` set (that boundary file's job is preventing prompt
# injection via data-derived text reaching the model; this module's
# job is fixing scripts that fail to run) but built independently
# rather than imported from it, so a future change to either module's
# scope can't silently change the other's behavior. Ranges:
#
#   U+200B-200F  Zero-width space/joiner/non-joiner + LTR/RTL marks
#   U+FEFF       Zero-width no-break space (byte-order mark)
#   U+2060       Word joiner
#   U+00AD       Soft hyphen (invisible except at a line break -- a
#                notorious copy-paste artifact from justified web text
#                or PDF extraction)
#   U+202A-202E  Bidi embedding/override controls
#   U+2066-2069  Bidi isolate controls
#   (the two ranges above are the "Trojan Source" character family --
#   see the module docstring's security-adjacent note)
#   U+2061-2064  Invisible math operators (function application,
#                invisible times/separator/plus -- leak from Word
#                equation editor / MathML copy-paste)
#   U+FE00-FE0F  Variation selectors (basic plane)
#   U+E0100-E01EF Variation selectors (supplementary, astral plane)
#   U+E0000-E007F Tag characters (includes the ASCII-mirrored payload
#                range historically used to smuggle invisible text)
# ---------------------------------------------------------------------------
_ZERO_WIDTH_RE = re.compile(
    "["
    "\\u200B-\\u200F"
    "\\uFEFF"
    "\\u2060"
    "\\u00AD"
    "\\u202A-\\u202E"
    "\\u2066-\\u2069"
    "\\u2061-\\u2064"
    "\\uFE00-\\uFE0F"
    "\\U000E0100-\\U000E01EF"
    "\\U000E0000-\\U000E007F"
    "]"
)

# ---------------------------------------------------------------------------
# Non-standard Unicode space separators -- the full Unicode "Zs"
# (Space_Separator) general category minus ASCII space itself. All
# render as "just a gap" to a human eye, none is a valid token
# separator to R / Stata / Python outside a string literal (the exact
# same risk/safety profile the original non-breaking-space handling
# already accepted -- this is a direct completion of that category,
# not a new one).
#
# Built from explicit codepoint integers (never pasted glyphs, for the
# same "invisible characters can't be verified by reading them" reason
# as the zero-width regex above) and cross-checked against
# ``unicodedata.category(chr(cp)) == "Zs"`` in the test suite.
# ---------------------------------------------------------------------------
_UNICODE_SPACE_CODEPOINTS: tuple[int, ...] = (
    0x00A0,  # NO-BREAK SPACE
    0x1680,  # OGHAM SPACE MARK
    0x2000,  # EN QUAD
    0x2001,  # EM QUAD
    0x2002,  # EN SPACE
    0x2003,  # EM SPACE
    0x2004,  # THREE-PER-EM SPACE
    0x2005,  # FOUR-PER-EM SPACE
    0x2006,  # SIX-PER-EM SPACE
    0x2007,  # FIGURE SPACE
    0x2008,  # PUNCTUATION SPACE
    0x2009,  # THIN SPACE
    0x200A,  # HAIR SPACE
    0x202F,  # NARROW NO-BREAK SPACE
    0x205F,  # MEDIUM MATHEMATICAL SPACE
    0x3000,  # IDEOGRAPHIC SPACE
)
_UNICODE_SPACES: frozenset[str] = frozenset(
    chr(_cp) for _cp in _UNICODE_SPACE_CODEPOINTS
)


@dataclass
class RepairResult:
    """Outcome of :func:`normalize_gremlins`.

    ``code`` is always populated (equal to the input when nothing
    changed). ``changed`` is the single boolean callers should branch
    on. ``descriptions`` is empty when ``changed`` is False; otherwise
    each entry is a short, human-readable summary of one category of
    edit, suitable for surfacing to the researcher/model verbatim.
    """
    code: str
    changed: bool
    descriptions: list[str] = field(default_factory=list)


def normalize_gremlins(code: str) -> RepairResult:
    """Strip/replace invisible and typographic-equivalent characters.

    Single pass over ``code``: every character is classified against
    the fixed tables above -- quotation-mark family, fullwidth ASCII
    punctuation, zero-width/format (regex character class), non-
    standard Unicode spaces -- in that order, and either passed through
    unchanged, substituted, or dropped. Nothing here inspects language
    syntax or the error message -- it is purely a character-level
    normalization, applied identically regardless of R / Stata /
    Python.

    Returns a :class:`RepairResult`. When ``code`` contains none of
    the flagged characters, ``changed`` is False and ``code`` on the
    result is identical (same object) to the input -- callers use
    ``changed`` to decide whether a re-run is worth attempting at all.
    """
    if not code:
        return RepairResult(code=code, changed=False)

    quote_count = 0
    fullwidth_count = 0
    zero_width_count = 0
    space_count = 0
    out: list[str] = []

    for ch in code:
        if ch in _SMART_QUOTES:
            quote_count += 1
            out.append(_SMART_QUOTES[ch])
        elif ch in _FULLWIDTH_PUNCTUATION:
            fullwidth_count += 1
            out.append(_FULLWIDTH_PUNCTUATION[ch])
        elif _ZERO_WIDTH_RE.match(ch):
            zero_width_count += 1
            # Dropped -- these have no width, so there is no
            # replacement character to insert.
        elif ch in _UNICODE_SPACES:
            space_count += 1
            out.append(" ")
        else:
            out.append(ch)

    if (quote_count == 0 and fullwidth_count == 0
            and zero_width_count == 0 and space_count == 0):
        return RepairResult(code=code, changed=False)

    new_code = "".join(out)
    descriptions: list[str] = []
    if quote_count:
        descriptions.append(
            f"replaced {quote_count} curly/typographic/fullwidth "
            f"quotation mark{'s' if quote_count != 1 else ''} with "
            f"straight ASCII quotes"
        )
    if fullwidth_count:
        descriptions.append(
            f"replaced {fullwidth_count} fullwidth ASCII "
            f"symbol{'s' if fullwidth_count != 1 else ''} (parentheses, "
            f"brackets, punctuation) with their standard-width "
            f"equivalents"
        )
    if zero_width_count:
        descriptions.append(
            f"removed {zero_width_count} invisible zero-width/format "
            f"character{'s' if zero_width_count != 1 else ''}"
        )
    if space_count:
        descriptions.append(
            f"replaced {space_count} non-standard Unicode space "
            f"character{'s' if space_count != 1 else ''} with regular "
            f"spaces"
        )
    return RepairResult(code=new_code, changed=True, descriptions=descriptions)
