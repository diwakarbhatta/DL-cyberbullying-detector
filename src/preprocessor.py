import re
import html
import unicodedata

# Common leetspeak replacements to catch obfuscated toxic terms
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
}

def normalize_leetspeak(text: str) -> str:
    """
    Substitutes common leetspeak characters to help catch obfuscated words.
    e.g., 'f*ck' -> 'fuck', 'b!tch' -> 'bitch', 'sh1t' -> 'shit'
    """
    # Remove asterisks inside words like f*ck, sh*t
    text = re.sub(r'(\b[a-zA-Z]+)\*([a-zA-Z]+\b)', r'\1u\2', text)
    
    # Character substitutions
    for char, replacement in LEET_DICT.items():
        text = text.replace(char, replacement)
        
    return text

def clean_text(text: str, remove_urls: bool = True, normalize_whitespace: bool = True) -> str:
    """
    Cleans raw comment text for model inference.
    """
    if not isinstance(text, str):
        return ""
        
    # Unescape HTML entities (e.g., &amp; -> &)
    text = html.unescape(text)
    
    # Normalize unicode (accents, weird quotes, etc.)
    text = unicodedata.normalize('NFKD', text)
    
    # Remove URLs if specified
    if remove_urls:
        text = re.sub(r'https?://\S+|www\.\S+', '[URL]', text)
        
    # Remove user mentions / handles if desired or normalize
    text = re.sub(r'@\w+', '[USER]', text)
    
    # Normalize excessive repetitive characters (e.g., "sooooo" -> "soo", "!!!!" -> "!!")
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    
    if normalize_whitespace:
        text = re.sub(r'\s+', ' ', text).strip()
        
    return text
