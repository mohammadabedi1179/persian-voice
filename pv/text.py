"""Persian text normalisation: one form for scoring STT, one for feeding TTS."""
import re

_CHAR_MAP = str.maketrans({
    "ي": "ی", "ى": "ی", "ئ": "ی", "ك": "ک", "ة": "ه", "ۀ": "ه", "أ": "ا", "إ": "ا", "ٱ": "ا",
    **{d: str(i) for i, d in enumerate("۰۱۲۳۴۵۶۷۸۹")},
    **{d: str(i) for i, d in enumerate("٠١٢٣٤٥٦٧٨٩")},
})
_DIACRITICS = re.compile(r"[ً-ٰٟـ]")  # harakat, superscript alef, tatweel
_PUNCT = re.compile(r"[^\w\s]|_")
_SPACES = re.compile(r"\s+")


def unify(text: str) -> str:
    """Arabic → Persian letters, any digits → ASCII, drop diacritics/tatweel."""
    return _DIACRITICS.sub("", text.translate(_CHAR_MAP))


def normalize_for_scoring(text: str) -> str:
    """Form used for WER/CER: both reference and hypothesis go through this.

    Persian writes many words with a half-space (ZWNJ), a full space or none
    (می‌روم / می روم / میروم). Models differ in which they write, so the
    verbal prefixes and plural/comparative suffixes are re-attached to their
    word and all three spellings score the same. Punctuation is removed and
    numbers spelled out, since a transcript may write «۸» for «هشت».
    """
    text = _NUMBER.sub(_num_repl, unify(text)).replace("‌", " ")
    text = _SPACES.sub(" ", _PUNCT.sub(" ", text)).strip()
    text = _PREFIX.sub(r"\1", text)
    return _SUFFIX.sub(r"\1", text)


_PREFIX = re.compile(r"(?<!\S)(ن?می) (?=\S)")
_SUFFIX = re.compile(r"(?<=\S) (ها|های|هایی|هایم|هایت|هایش|هایمان|هایتان|هایشان|تر|ترین)(?!\S)")


# ---------------------------------------------------------------- numbers → words
_ONES = ["", "یک", "دو", "سه", "چهار", "پنج", "شش", "هفت", "هشت", "نه"]
_TEENS = ["ده", "یازده", "دوازده", "سیزده", "چهارده", "پانزده", "شانزده", "هفده", "هجده", "نوزده"]
_TENS = ["", "", "بیست", "سی", "چهل", "پنجاه", "شصت", "هفتاد", "هشتاد", "نود"]
_HUNDREDS = ["", "صد", "دویست", "سیصد", "چهارصد", "پانصد", "ششصد", "هفتصد", "هشتصد", "نهصد"]
_SCALES = ["", "هزار", "میلیون", "میلیارد", "تریلیون"]


def _below_1000(n: int) -> str:
    parts = []
    h, rest = divmod(n, 100)
    if h:
        parts.append(_HUNDREDS[h])
    if 10 <= rest < 20:
        parts.append(_TEENS[rest - 10])
    else:
        t, o = divmod(rest, 10)
        if t:
            parts.append(_TENS[t])
        if o:
            parts.append(_ONES[o])
    return " و ".join(parts)


def number_to_words(n: int) -> str:
    if n == 0:
        return "صفر"
    if n < 0:
        return "منفی " + number_to_words(-n)
    groups, i = [], 0
    while n and i < len(_SCALES):
        n, g = divmod(n, 1000)
        if g:
            words = _below_1000(g)
            if i:
                words = _SCALES[i] if (g == 1 and i == 1) else f"{words} {_SCALES[i]}"
            groups.append(words)
        i += 1
    return " و ".join(reversed(groups))


def _num_repl(m: re.Match) -> str:
    whole, frac = m.group(1).replace(",", ""), m.group(2)
    words = number_to_words(int(whole))
    if frac:
        words += " ممیز " + number_to_words(int(frac))
    return f" {words} "


_NUMBER = re.compile(r"(\d[\d,]*)(?:[./٫](\d+))?")


_TTS_MAP = str.maketrans({
    "ي": "ی", "ى": "ی", "ك": "ک", "ـ": "",
    **{d: str(i) for i, d in enumerate("۰۱۲۳۴۵۶۷۸۹")},
    **{d: str(i) for i, d in enumerate("٠١٢٣٤٥٦٧٨٩")},
})


def prepare_for_tts(text: str) -> str:
    """Form fed to the TTS model: spelled-out numbers, Persian letters, clean spacing.

    Unlike scoring, this keeps what affects pronunciation and is in the model's
    vocabulary: ئ أ إ ؤ ة, diacritics and ZWNJ. Numbers are spelled out because
    the model card warns that digits may be mispronounced.
    """
    text = _NUMBER.sub(_num_repl, text.translate(_TTS_MAP))
    text = text.replace("%", " درصد ").replace("٪", " درصد ")
    return _SPACES.sub(" ", text).strip()


def split_sentences(text: str, max_chars: int = 180) -> list[str]:
    """Split into sentence chunks no longer than max_chars (XTTS degrades on long inputs)."""
    pieces = [p.strip() for p in re.split(r"(?<=[.!?؟؛\n])\s+", text) if p.strip()]
    chunks = []
    for p in pieces:
        while len(p) > max_chars:  # break overlong sentences at a comma or space
            cut = max(p.rfind("،", 0, max_chars), p.rfind(",", 0, max_chars))
            if cut < max_chars // 3:
                cut = p.rfind(" ", 0, max_chars)
            cut = cut if cut > 0 else max_chars
            chunks.append(p[: cut + 1].strip())
            p = p[cut + 1:].strip()
        if p:
            chunks.append(p)
    return chunks
