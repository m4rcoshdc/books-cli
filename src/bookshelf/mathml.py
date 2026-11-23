"""MathML -> LaTeX, covering the elements real ebooks use.

Equations stay text, so agents can read them and search can index them. If a
<semantics> block already carries a TeX annotation, that is used verbatim.
"""

import re
import unicodedata

_SYMBOLS = {
    "∈": r"\in", "∉": r"\notin", "≤": r"\le", "≥": r"\ge", "≠": r"\neq", "×": r"\times",
    "·": r"\cdot", "⋅": r"\cdot", "→": r"\to", "←": r"\leftarrow", "↦": r"\mapsto",
    "⇒": r"\Rightarrow", "⇔": r"\Leftrightarrow", "⟹": r"\implies", "∞": r"\infty",
    "∑": r"\sum", "∏": r"\prod", "∫": r"\int", "∂": r"\partial", "∇": r"\nabla", "±": r"\pm",
    "∓": r"\mp", "−": "-", "…": r"\ldots", "⋯": r"\cdots", "⋮": r"\vdots", "∀": r"\forall",
    "∃": r"\exists", "⊂": r"\subset", "⊆": r"\subseteq", "⊃": r"\supset", "∪": r"\cup",
    "∩": r"\cap", "∖": r"\setminus", "≡": r"\equiv", "≈": r"\approx", "∼": r"\sim",
    "≃": r"\simeq", "∝": r"\propto", "∘": r"\circ", "⊗": r"\otimes", "⊕": r"\oplus",
    "⟨": r"\langle", "⟩": r"\rangle", "∥": r"\|", "∣": r"\mid", "ℓ": r"\ell", "∅": r"\emptyset",
    "¬": r"\neg", "∧": r"\wedge", "∨": r"\vee", "′": "'", "″": "''", "⁎": "*", "∗": "*",
    "{": r"\{", "}": r"\}", "%": r"\%", "#": r"\#", "&": r"\&", "_": r"\_", "⁡": "",
    "⁢": "", "⁣": "", " ": " ",
}
_GREEK_CAPS_IN_TEX = {"Gamma", "Delta", "Theta", "Lambda", "Xi", "Pi", "Sigma", "Upsilon", "Phi", "Psi", "Omega"}
_ACCENTS_OVER = {"¯": r"\overline", "‾": r"\overline", "^": r"\hat", "ˆ": r"\hat", "~": r"\tilde",
                 "˜": r"\tilde", "˙": r"\dot", "¨": r"\ddot", "→": r"\vec", "⃗": r"\vec", "̄": r"\overline"}
_ACCENTS_UNDER = {"̲": r"\underline", "_": r"\underline", "‾": r"\underline", "¯": r"\underline"}
_VARIANTS = {"bold": r"\mathbf", "double-struck": r"\mathbb", "script": r"\mathcal",
             "bold-script": r"\mathcal", "fraktur": r"\mathfrak", "sans-serif": r"\mathsf",
             "bold-italic": r"\boldsymbol"}
_FUNCTIONS = {"sin", "cos", "tan", "log", "ln", "exp", "max", "min", "sup", "inf", "lim", "det", "arg"}


def _char(c: str) -> str:
    if c in _SYMBOLS:
        tex = _SYMBOLS[c]
        # "\equiv" + "V" must not become "\equivV": end letter-commands with a space.
        return tex + " " if tex.startswith("\\") and tex[-1].isalpha() else tex
    name = unicodedata.name(c, "")
    if name.startswith("GREEK SMALL LETTER "):
        return "\\" + name.removeprefix("GREEK SMALL LETTER ").lower().replace("final sigma", "varsigma") + " "
    if name.startswith("GREEK CAPITAL LETTER "):
        cap = name.removeprefix("GREEK CAPITAL LETTER ").capitalize()
        return ("\\" + cap + " ") if cap in _GREEK_CAPS_IN_TEX else c
    return c


def _chars(text: str) -> str:
    return "".join(_char(c) for c in text)


def _kids(el) -> list:
    return [c for c in el.children if getattr(c, "name", None)]


def _g(s: str) -> str:
    s = s.strip()
    return s if len(s) == 1 else "{" + s + "}"


def _conv(el) -> str:
    name = el.name.split(":")[-1]
    kids = _kids(el)
    text = el.get_text().strip()

    if name == "mi":
        variant = el.get("mathvariant")
        if len(text) > 1 and text.isalpha():
            body = rf"\{text}" if text in _FUNCTIONS else rf"\mathrm{{{text}}}"
        else:
            body = _chars(text)
        return f"{_VARIANTS[variant]}{{{body}}}" if variant in _VARIANTS else body
    if name == "mn":
        return text
    if name == "mo":
        return " " + _chars(text) + " " if len(text) == 1 and text in "=<>+" else _chars(text)
    if name == "mtext":
        return rf"\text{{{text}}}" if text else " "
    if name == "mspace":
        return r"\ "
    if name in ("msub", "msup", "msubsup") and kids:
        base = _g(_conv(kids[0]))
        if name == "msub" and len(kids) > 1:
            return f"{base}_{{{_conv(kids[1])}}}"
        if name == "msup" and len(kids) > 1:
            return f"{base}^{{{_conv(kids[1])}}}"
        if len(kids) > 2:
            return f"{base}_{{{_conv(kids[1])}}}^{{{_conv(kids[2])}}}"
    if name in ("munder", "mover", "munderover") and kids:
        base = _conv(kids[0])
        if name == "munder" and len(kids) > 1:
            u = kids[1].get_text().strip()
            if u in _ACCENTS_UNDER:
                return f"{_ACCENTS_UNDER[u]}{{{base}}}"
            return f"{_g(base)}_{{{_conv(kids[1])}}}" if base.strip().startswith("\\") else rf"\underset{{{_conv(kids[1])}}}{{{base}}}"
        if name == "mover" and len(kids) > 1:
            o = kids[1].get_text().strip()
            if o in _ACCENTS_OVER:
                return f"{_ACCENTS_OVER[o]}{{{base}}}"
            return rf"\overset{{{_conv(kids[1])}}}{{{base}}}"
        if len(kids) > 2:
            return f"{_g(base)}_{{{_conv(kids[1])}}}^{{{_conv(kids[2])}}}"
    if name == "mfrac" and len(kids) == 2:
        return rf"\frac{{{_conv(kids[0])}}}{{{_conv(kids[1])}}}"
    if name == "msqrt":
        return rf"\sqrt{{{''.join(_conv(k) for k in kids)}}}"
    if name == "mroot" and len(kids) == 2:
        return rf"\sqrt[{_conv(kids[1])}]{{{_conv(kids[0])}}}"
    if name == "mfenced":
        open_, close = el.get("open", "("), el.get("close", ")")
        sep = el.get("separators", ",")
        sep = sep.strip()[:1] if sep.strip() else ""
        inner = (sep + " ").join(_conv(k) for k in kids)
        return rf"\left{_fence(open_)} {inner} \right{_fence(close)}"
    if name == "mtable":
        rows = []
        for tr in kids:
            cells = [_conv(td) for td in _kids(tr)] if tr.name.endswith("tr") else [_conv(tr)]
            rows.append(" & ".join(cells))
        return r"\begin{aligned} " + r" \\ ".join(rows) + r" \end{aligned}"
    if name == "semantics" and kids:
        return _conv(kids[0])
    if name in ("annotation", "annotation-xml", "mphantom", "none"):
        return ""
    if name == "menclose":
        return rf"\boxed{{{''.join(_conv(k) for k in kids)}}}"
    # mrow, mstyle, mpadded, math, mtd, mmultiscripts, unknown: concatenate.
    if kids:
        return "".join(_conv(k) for k in kids)
    return _chars(text)


def _fence(c: str) -> str:
    if not c:
        return "."
    return {"{": r"\{", "}": r"\}", "⟨": r"\langle", "⟩": r"\rangle", "|": "|", "‖": r"\|"}.get(c, c)


def mathml_to_latex(math_el) -> str:
    """Return LaTeX for a <math> element (bs4 Tag), without $ delimiters."""
    for ann in math_el.find_all("annotation"):
        if "tex" in (ann.get("encoding") or "").lower() and ann.get_text().strip():
            return ann.get_text().strip()
    latex = _conv(math_el)
    latex = re.sub(r"[ \t]+", " ", latex).strip()
    return latex
