import re
import html
import unicodedata

# Common leetspeak character substitutions (used only *inside* suspected
# obfuscated words — see normalize_leetspeak below).
LEET_DICT = {
    '@': 'a',
    '4': 'a',
    '8': 'b',
    '3': 'e',
    '1': 'i',
    '!': 'i',
    '0': 'o',
    '5': 's',
    '$': 's',
    '7': 't',
    '+': 't',
    '*': 'u',  # censored-style profanity: f*ck -> fuck, sh*t -> shit
}

_LEET_TABLE = str.maketrans(LEET_DICT)

# Matches a token that mixes letters with leetspeak digits/symbols, e.g.
# "sh1t", "h4te", "b!tch" (punctuation-subset only), "f*ck".
#   - digit-containing tokens need >= 2 letters ("1985", "123" are untouched);
#   - punctuation-only substitutions (!@*$+!) need >= 3 letters so stray
#     symbols aren't converted into letters.
_LEET_TOKEN_RE = re.compile(
    r'\b(?=[a-z]*[0-9])(?=(?:[a-z][0-9]?){2})[a-z0-9]{3,}\b'
    r'|\b(?=[a-z]*[!@*$+])(?=(?:[a-z][!@*$+]?){3})[a-z!@*$+]{4,}\b'
)


def normalize_leetspeak(text: str) -> str:
    """
    Decodes *word-level* leetspeak without corrupting digits or punctuation.

    e.g., 'f*ck' -> 'fuck', 'b!tch' -> 'bitch', 'sh1t' -> 'shit',
          'h4te' -> 'hate'

    Previously this function blindly replaced every occurrence of each
    leet character across the whole string, which mangled perfectly normal
    text: "I was born in 1985" became "I was born in i9bs" and "wow!!!"
    became "wowiii".  Now substitutions are applied only to tokens that
    look like intentionally obfuscated words; real numbers ("2024", "1985")
    and repeated punctuation ("!!!", "???") pass through unchanged.
    """
    if not isinstance(text, str):
        return ""
    return _LEET_TOKEN_RE.sub(lambda m: m.group(0).translate(_LEET_TABLE), text)

def clean_text(text: str, remove_urls: bool = True, normalize_whitespace: bool = True) -> str:
    """
    Cleans raw comment text for model inference.
    """
    if not isinstance(text, str):
        return ""
        
    # Unescape HTML entities (e.g., &amp; -> &)
    text = html.unescape(text)
    
    # Normalize unicode (accents, weird quotes, etc.) and drop combining
    # marks so "café" -> "cafe", "ｆｕｃｋ" -> "fuck".
    text = unicodedata.normalize('NFKD', text)
    text = ''.join(c for c in text if not unicodedata.combining(c))

    # Lowercase early so the leetspeak token regex can rely on [a-z].
    text = text.lower()

    # Remove URLs if specified
    if remove_urls:
        text = re.sub(r'https?://\S+|www\.\S+', '[URL]', text)

    # Remove user mentions / handles if desired or normalize
    text = re.sub(r'@\w+', '[USER]', text)

    # Decode word-level leetspeak ("sh1t" -> "shit", "f*ck" -> "fuck")
    # *before* collapsing repeated characters, so "!!!"/"???" emphasis is
    # never mistaken for leet letters.
    text = normalize_leetspeak(text)

    # Normalize excessive repetitive characters (e.g., "sooooo" -> "soo").
    # Letters only — never digits ("555") or punctuation ("!!!", "???"),
    # which carry meaning and were previously mangled by `(.)\1{2,}`.
    text = re.sub(r'([a-zA-Z])\1{2,}', r'\1\1', text)
    
    if normalize_whitespace:
        text = re.sub(r'\s+', ' ', text).strip()
        
    return text
