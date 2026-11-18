"""Rule-checkable instructions about the REASONING TRACE (ReasonIF-style), checked per 64-token window."""
import re

INSTR = {
    "uppercase": "Write your entire response, including all of your thinking, in CAPITAL LETTERS only.",
    "no_commas": "Do not use any commas anywhere in your response, including all of your thinking.",
    "french": "Write your entire response, including all of your thinking, in French only.",
}
REMIND = "\n(Reminder of the instruction: {})\n"

FR = set("le la les de des du un une et est que qui pour pas dans sur avec nous donc il elle ce cette sont mais ou on par plus alors".split())
EN = set("the and is that to of for we so this it are with but not then be in on as by let need total".split())


def window_ok(kind, text):
    """True / False, or None when the window has nothing checkable."""
    if kind == "no_commas":
        return "," not in text
    if kind == "uppercase":
        letters = [c for c in text if c.isalpha()]
        if len(letters) < 10:
            return None
        return sum(c.isupper() for c in letters) / len(letters) >= 0.95
    if kind == "french":
        w = re.findall(r"[a-zàâçéèêëîïôûùüÿœ']+", text.lower())
        fr, en = sum(x in FR for x in w), sum(x in EN for x in w)
        if fr + en < 2:
            return None
        return fr > en
    raise ValueError(kind)


def answer_of(text):
    m = re.findall(r"\\boxed\{([^{}]*)\}", text)
    s = m[-1] if m else text
    nums = re.findall(r"-?\d[\d,]*\.?\d*", s)
    return nums[-1].replace(",", "").rstrip(".") if nums else None


def correct(p, g):
    try:
        return p is not None and abs(float(p) - float(g)) < 1e-6
    except ValueError:
        return False
