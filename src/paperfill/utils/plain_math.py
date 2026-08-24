"""
Rewrite LaTeX an LLM slipped into an answer as plain text a student would write.

Answers are stamped onto the page as literal glyphs, so a response of
``\\frac{5\\sqrt{6}}{\\sqrt{22}}`` lands on the worksheet as backslashes and
braces. Math sheets pull that out of the model constantly and prompting alone
never fully stops it, so every answer goes through here on the way to the page.
"""

import re

_SUPERSCRIPT = {
    "0": "\u2070", "1": "\u00b9", "2": "\u00b2", "3": "\u00b3", "4": "\u2074",
    "5": "\u2075", "6": "\u2076", "7": "\u2077", "8": "\u2078", "9": "\u2079",
    "+": "\u207a", "-": "\u207b", "\u2212": "\u207b", "=": "\u207c",
    "(": "\u207d", ")": "\u207e", "n": "\u207f", "i": "\u2071",
}
_SUBSCRIPT = {
    "0": "\u2080", "1": "\u2081", "2": "\u2082", "3": "\u2083", "4": "\u2084",
    "5": "\u2085", "6": "\u2086", "7": "\u2087", "8": "\u2088", "9": "\u2089",
    "+": "\u208a", "-": "\u208b", "\u2212": "\u208b", "=": "\u208c",
    "(": "\u208d", ")": "\u208e",
}

_SYMBOLS = {
    "alpha": "\u03b1", "beta": "\u03b2", "gamma": "\u03b3", "delta": "\u03b4",
    "Delta": "\u0394", "epsilon": "\u03b5", "varepsilon": "\u03b5",
    "zeta": "\u03b6", "eta": "\u03b7", "theta": "\u03b8", "Theta": "\u0398",
    "lambda": "\u03bb", "mu": "\u03bc", "nu": "\u03bd", "xi": "\u03be",
    "pi": "\u03c0", "Pi": "\u03a0", "rho": "\u03c1", "sigma": "\u03c3",
    "Sigma": "\u03a3", "tau": "\u03c4", "phi": "\u03c6", "varphi": "\u03c6",
    "Phi": "\u03a6", "chi": "\u03c7", "psi": "\u03c8", "omega": "\u03c9",
    "Omega": "\u03a9",
    "infty": "\u221e", "pm": "\u00b1", "mp": "\u2213", "times": "\u00d7",
    "div": "\u00f7", "cdot": "\u00b7", "ast": "*",
    "le": "\u2264", "leq": "\u2264", "ge": "\u2265", "geq": "\u2265",
    "ne": "\u2260", "neq": "\u2260", "approx": "\u2248", "equiv": "\u2261",
    "cong": "\u2245", "sim": "~", "propto": "\u221d", "therefore": "\u2234",
    "to": "\u2192", "rightarrow": "\u2192", "longrightarrow": "\u2192",
    "Rightarrow": "\u21d2", "leftarrow": "\u2190", "Leftarrow": "\u21d0",
    "leftrightarrow": "\u2194", "Leftrightarrow": "\u21d4", "mapsto": "\u21a6",
    "in": "\u2208", "notin": "\u2209", "subset": "\u2282", "subseteq": "\u2286",
    "supset": "\u2283", "cup": "\u222a", "cap": "\u2229",
    "emptyset": "\u2205", "varnothing": "\u2205", "forall": "\u2200",
    "exists": "\u2203", "neg": "\u00ac", "land": "\u2227", "lor": "\u2228",
    "sum": "\u03a3", "prod": "\u03a0", "int": "\u222b", "partial": "\u2202",
    "nabla": "\u2207", "angle": "\u2220", "triangle": "\u25b3",
    "perp": "\u22a5", "parallel": "\u2225",
    # \circ is composition in analysis, but on a worksheet it is nearly always
    # the degree sign in 30^\circ.
    "circ": "\u00b0", "degree": "\u00b0",
    "ldots": "\u2026", "dots": "\u2026", "cdots": "\u2026",
    "quad": " ", "qquad": " ",
}

# Commands whose brace argument is the answer text itself.
_PASSTHROUGH = {"text", "textbf", "textit", "textrm", "mathrm", "mathbf",
                "mathit", "mathsf", "mathbb", "mathcal", "operatorname",
                "mbox", "boxed", "overline", "underline", "bar", "hat", "vec"}

# Size/delimiter modifiers: drop the command, keep whatever bracket follows.
_SIZERS = {"left", "right", "big", "Big", "bigg", "Bigg", "bigl", "bigr",
           "Bigl", "Bigr", "biggl", "biggr", "displaystyle", "textstyle"}

_FRACTIONS = {"frac", "dfrac", "tfrac", "cfrac"}

_COMMAND = re.compile(r"\\([a-zA-Z]+|.)", re.DOTALL)
_COMPOUND = re.compile(r"[+\-\u2212*/\u00b7\u00d7\u00f7 ]")

# What makes a string worth rewriting: a TeX command, a script with a real
# argument, or a $...$ span with TeX inside it. Without this guard a plain
# answer like "$40" or "snake_case" would get mangled.
_TEX_SIGNAL = re.compile(r"\\[a-zA-Z(\[]|[\^_]\s*[{\d(]|\$[^$]*[\\{^_][^$]*\$")


def plain_math(text: str) -> str:
    """Convert any LaTeX in `text` to plain Unicode. Non-LaTeX text is returned
    untouched."""
    if not text or not _TEX_SIGNAL.search(text):
        return text
    return re.sub(r"[ \t]{2,}", " ", _convert(text)).strip()


def _argument(s: str, i: int) -> tuple[str, int]:
    """Read a command argument at s[i]: a braced group, a command, or one char.
    Returns the raw argument text and the index just past it."""
    while i < len(s) and s[i] == " ":
        i += 1
    if i >= len(s):
        return "", i
    if s[i] == "\\":
        m = _COMMAND.match(s, i)
        return (s[i:m.end()], m.end()) if m else ("", i + 1)
    if s[i] != "{":
        return s[i], i + 1
    depth = 0
    for j in range(i, len(s)):
        if s[j] == "{":
            depth += 1
        elif s[j] == "}":
            depth -= 1
            if depth == 0:
                return s[i + 1:j], j + 1
    return s[i + 1:], len(s)


def _grouped(part: str) -> str:
    """Parenthesize a fraction side or radicand that is more than one term, so
    (3+8)/(2-8) doesn't collapse into 3+8/2-8."""
    if len(part) <= 1 or not _COMPOUND.search(part):
        return part
    if part.startswith("(") and part.endswith(")"):
        return part
    return f"({part})"


def _radical(index: str) -> str:
    return {"": "\u221a", "2": "\u221a", "3": "\u221b", "4": "\u221c"}.get(
        index.strip(), _script(index.strip(), "^") + "\u221a")


def _script(text: str, kind: str) -> str:
    """Superscript/subscript `text` if every character has a Unicode form,
    otherwise leave a readable ^/_ behind."""
    table = _SUPERSCRIPT if kind == "^" else _SUBSCRIPT
    if text == "\u00b0":                      # 30^\circ, not 30 raised to a circle
        return text
    mapped = [table.get(ch) for ch in text]
    if text and all(mapped):
        return "".join(mapped)
    return f"{kind}({text})" if len(text) > 1 else f"{kind}{text}"


def _convert(s: str) -> str:
    out: list[str] = []
    i, n = 0, len(s)
    while i < n:
        ch = s[i]
        if ch == "\\":
            m = _COMMAND.match(s, i)
            if not m:
                i += 1
                continue
            name, i = m.group(1), m.end()
            if name in _FRACTIONS:
                num, i = _argument(s, i)
                den, i = _argument(s, i)
                out.append(f"{_grouped(_convert(num))}/{_grouped(_convert(den))}")
            elif name == "sqrt":
                index = ""
                j = i
                while j < n and s[j] == " ":
                    j += 1
                if j < n and s[j] == "[":
                    close = s.find("]", j)
                    if close != -1:
                        index, i = s[j + 1:close], close + 1
                arg, i = _argument(s, i)
                out.append(_radical(index) + _grouped(_convert(arg)))
            elif name in _PASSTHROUGH:
                arg, i = _argument(s, i)
                out.append(_convert(arg))
            elif name in _SIZERS:
                if i < n and s[i] == ".":     # \left. — an invisible delimiter
                    i += 1
            elif name in _SYMBOLS:
                # A space after the command name only terminates it: "2\pi r" is
                # 2πr. But when the command was already space-separated from what
                # came before ("x = 45 \pm 3"), the spacing is the author's and
                # dropping it runs the symbol into the next token.
                prev = out[-1] if out else ""
                attached = bool(prev) and not prev.endswith(" ")
                out.append(_SYMBOLS[name])
                if attached and i < n and s[i] == " ":
                    i += 1
            elif name in "(){}[]$%&#_":
                out.append("" if name in "()[]" else name)
            elif name in (",", ";", ":", " "):
                out.append(" ")
            elif name in ("!", "\\"):
                out.append("" if name == "!" else " ")
            else:
                out.append(name)              # \sin, \log, or a macro we don't know
        elif ch in "^_":
            arg, i = _argument(s, i + 1)
            out.append(_script(_convert(arg), ch))
        elif ch in "{}$":
            i += 1
        else:
            out.append(ch)
            i += 1
    return "".join(out)
