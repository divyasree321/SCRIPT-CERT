"""
SCRIPT-CERT
SCRIPT-SHIFT dataset generation and validation

Conditions:
1. original
2. romanized
3. telugu_script
4. mixed_script
5. mild_noise
6. severe_noise

Uses indic-transliteration for Telugu <-> Roman transliteration.
"""

from pathlib import Path
import re
import random
import pandas as pd

from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42
random.seed(SEED)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "script_shift"
    / "script_shift_base.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "script_shift"
    / "script_shift_base.csv"
)


# ============================================================
# REGEX
# ============================================================

TELUGU_RE = re.compile(r"[\u0C00-\u0C7F]")

WORD_RE = re.compile(r"\S+")


# ============================================================
# BASIC HELPERS
# ============================================================

def contains_telugu(text):
    """Return True if text contains Telugu Unicode characters."""
    if pd.isna(text):
        return False

    return bool(TELUGU_RE.search(str(text)))


def safe_text(value):
    """Convert dataframe value safely to string."""
    if pd.isna(value):
        return ""

    return str(value).strip()


# ============================================================
# TELUGU -> ROMANIZED
# ============================================================

def telugu_to_romanized(text):
    """
    Convert Telugu Unicode text to ITRANS-style Roman text.

    English text is retained where possible.
    """
    text = safe_text(text)

    if not text:
        return text

    if not contains_telugu(text):
        return text

    try:
        converted = transliterate(
            text,
            sanscript.TELUGU,
            sanscript.ITRANS
        )

        return converted

    except Exception:
        return text


# ============================================================
# ROMANIZED -> TELUGU
# ============================================================

# Common English words that should remain English instead of
# being transliterated into Telugu.
ENGLISH_WORDS = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "but",
    "if",
    "then",
    "this",
    "that",
    "these",
    "those",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "very",
    "good",
    "bad",
    "best",
    "worst",
    "person",
    "persons",
    "people",
    "man",
    "woman",
    "men",
    "women",
    "boy",
    "girl",
    "boys",
    "girls",
    "hello",
    "hi",
    "youtube",
    "video",
    "videos",
    "follow",
    "future",
    "all",
    "data",
    "recharge",
    "police",
    "support",
    "free",
    "awesome",
    "extraordinary",
    "ordinary",
    "life",
    "love",
    "hate",
    "happy",
    "sad",
    "friend",
    "friends",
    "brother",
    "sister",
    "father",
    "mother",
    "sir",
    "madam",
    "please",
    "thank",
    "thanks",
    "congratulations",
    "government",
    "news",
    "school",
    "college",
    "student",
    "students",
    "doctor",
    "doctors",
    "police",
}


def preserve_punctuation(word):
    """
    Split punctuation from a word.

    Example:
        "brother." -> ("brother", ".")
    """
    match = re.match(r"^([^\w]*)(.*?)([^\w]*)$", word, flags=re.UNICODE)

    if not match:
        return "", word, ""

    prefix = match.group(1)
    core = match.group(2)
    suffix = match.group(3)

    return prefix, core, suffix


def looks_like_english(word):
    """
    Conservative English-word check.
    """
    clean = re.sub(r"[^A-Za-z]", "", word).lower()

    if not clean:
        return False

    return clean in ENGLISH_WORDS


def romanized_to_telugu(text):
    """
    Convert Romanized Telugu portions into Telugu script while
    retaining obvious English words.

    This is intentionally conservative because the dataset is
    Telugu-English code-mixed.
    """
    text = safe_text(text)

    if not text:
        return text

    # Already Telugu
    if contains_telugu(text):
        return text

    words = text.split()
    converted_words = []

    for word in words:

        prefix, core, suffix = preserve_punctuation(word)

        if not core:
            converted_words.append(word)
            continue

        # Keep obvious English words unchanged
        if looks_like_english(core):
            converted_words.append(word)
            continue

        # Keep URLs / usernames / email-like tokens unchanged
        if (
            "http" in core.lower()
            or "www." in core.lower()
            or "@" in core
            or "#" in core
        ):
            converted_words.append(word)
            continue

        # Try Roman -> Telugu
        try:
            converted = transliterate(
                core,
                sanscript.ITRANS,
                sanscript.TELUGU
            )

            if converted and converted != core:
                converted_words.append(
                    prefix + converted + suffix
                )
            else:
                converted_words.append(word)

        except Exception:
            converted_words.append(word)

    return " ".join(converted_words)


# ============================================================
# MIXED SCRIPT
# ============================================================

def create_mixed_script(original, romanized, telugu_script):
    """
    Create a Telugu-English mixed-script representation.

    Approximately half of the content is represented in Telugu
    script and the remainder in Roman/English script.
    """

    original = safe_text(original)
    romanized = safe_text(romanized)
    telugu_script = safe_text(telugu_script)

    if not telugu_script:
        return original

    telugu_words = telugu_script.split()
    roman_words = romanized.split()

    if not telugu_words:
        return romanized

    # Use Telugu representation for roughly half of tokens.
    result = []

    max_len = max(len(telugu_words), len(roman_words))

    for i in range(max_len):

        if i < len(telugu_words) and i < len(roman_words):

            if i % 2 == 0:
                result.append(telugu_words[i])
            else:
                result.append(roman_words[i])

        elif i < len(telugu_words):
            result.append(telugu_words[i])

        else:
            result.append(roman_words[i])

    return " ".join(result)


# ============================================================
# SPELLING NOISE
# ============================================================

VOWEL_REPLACEMENTS = {
    "a": "aa",
    "e": "ee",
    "i": "ii",
    "o": "oo",
    "u": "uu",
}


def add_mild_noise_word(word):
    """
    Apply a small spelling variation.
    """

    if len(word) < 4:
        return word

    chars = list(word)

    # Remove one repeated character occasionally
    for i in range(len(chars) - 1):

        if chars[i].lower() == chars[i + 1].lower():

            if random.random() < 0.5:
                del chars[i]
                return "".join(chars)

    # Mild vowel elongation
    for i, char in enumerate(chars):

        lower = char.lower()

        if lower in VOWEL_REPLACEMENTS:

            if random.random() < 0.15:

                replacement = VOWEL_REPLACEMENTS[lower]

                if char.isupper():
                    replacement = replacement.upper()

                chars[i] = replacement[0]

                chars.insert(i + 1, replacement[1])

                return "".join(chars)

    # Character duplication
    if random.random() < 0.20:

        index = random.randint(1, len(chars) - 2)

        chars.insert(index, chars[index])

    return "".join(chars)


def add_severe_noise_word(word):
    """
    Apply stronger spelling perturbation while retaining
    approximate readability.
    """

    if len(word) < 4:
        return word

    chars = list(word)

    operation = random.choice(
        [
            "duplicate",
            "remove",
            "swap",
            "elongate",
        ]
    )

    # Duplicate
    if operation == "duplicate":

        index = random.randint(1, len(chars) - 1)

        chars.insert(index, chars[index])

    # Remove
    elif operation == "remove":

        index = random.randint(1, len(chars) - 2)

        del chars[index]

    # Swap adjacent characters
    elif operation == "swap":

        index = random.randint(0, len(chars) - 2)

        chars[index], chars[index + 1] = (
            chars[index + 1],
            chars[index],
        )

    # Vowel elongation
    elif operation == "elongate":

        for i, char in enumerate(chars):

            if char.lower() in {"a", "e", "i", "o", "u"}:

                chars.insert(i + 1, char)

                break

    return "".join(chars)


def add_mild_noise(text):
    """
    Mild spelling noise.
    """

    words = text.split()

    result = []

    for word in words:

        if random.random() < 0.25:
            result.append(add_mild_noise_word(word))
        else:
            result.append(word)

    return " ".join(result)


def add_severe_noise(text):
    """
    Severe spelling noise.
    """

    words = text.split()

    result = []

    for word in words:

        if random.random() < 0.60:
            result.append(add_severe_noise_word(word))
        else:
            result.append(word)

    return " ".join(result)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("SCRIPT-CERT SCRIPT-SHIFT DATA GENERATION")
print("=" * 70)

print("\nReading:")
print(INPUT_FILE)

if not INPUT_FILE.exists():

    raise FileNotFoundError(
        f"Input file not found:\n{INPUT_FILE}"
    )


df = pd.read_csv(INPUT_FILE)


print("\nOriginal columns:")
print(list(df.columns))

print("\nRows:", len(df))


# ============================================================
# IDENTIFY ORIGINAL TEXT COLUMN
# ============================================================

if "original_text" in df.columns:

    text_column = "original_text"

elif "original" in df.columns:

    text_column = "original"

elif "text" in df.columns:

    text_column = "text"

else:

    raise ValueError(
        "Could not find original text column."
    )


# ============================================================
# NORMALIZE ORIGINAL
# ============================================================

df["original"] = df[text_column].apply(safe_text)


# Remove rows without text
df = df[df["original"].str.len() > 0].copy()

df.reset_index(drop=True, inplace=True)


print("\nValid original texts:", len(df))


# ============================================================
# PRESERVE LABEL
# ============================================================

if "label" not in df.columns:

    raise ValueError(
        "Dataset must contain a 'label' column."
    )


# ============================================================
# GENERATE SCRIPT-SHIFT CONDITIONS
# ============================================================

print("\nGenerating SCRIPT-SHIFT conditions...")

print("Generating romanized...")

df["romanized"] = df["original"].apply(
    telugu_to_romanized
)


print("Generating Telugu script...")

def generate_telugu_script(row):

    original = safe_text(row["original"])
    romanized = safe_text(row["romanized"])

    # If original already contains Telugu, retain the
    # native Telugu content.
    if contains_telugu(original):

        # Convert only if useful Roman representation exists.
        converted = romanized_to_telugu(romanized)

        if contains_telugu(converted):
            return converted

        return original

    # For Romanized/code-mixed text, convert Romanized
    # Telugu portions to Telugu script.
    return romanized_to_telugu(romanized)


df["telugu_script"] = df.apply(
    generate_telugu_script,
    axis=1
)


print("Generating mixed script...")

df["mixed_script"] = df.apply(
    lambda row: create_mixed_script(
        row["original"],
        row["romanized"],
        row["telugu_script"],
    ),
    axis=1
)


print("Generating mild spelling noise...")

random.seed(SEED)

df["mild_noise"] = df["romanized"].apply(
    add_mild_noise
)


print("Generating severe spelling noise...")

random.seed(SEED + 1)

df["severe_noise"] = df["romanized"].apply(
    add_severe_noise
)


# ============================================================
# FINAL COLUMN ORDER
# ============================================================

columns = [
    "id",
    "original_text",
    "label",
    "original",
    "romanized",
    "telugu_script",
    "mixed_script",
    "mild_noise",
    "severe_noise",
]


# Keep only columns that exist
columns = [
    c for c in columns
    if c in df.columns
]


df = df[columns]


# ============================================================
# QUALITY CHECK
# ============================================================

print("\n")
print("=" * 70)
print("TRANSFORMATION QUALITY CHECK")
print("=" * 70)


conditions = [
    "original",
    "romanized",
    "telugu_script",
    "mixed_script",
    "mild_noise",
    "severe_noise",
]


for condition in conditions:

    non_null = df[condition].notna().sum()

    non_empty = (
        df[condition]
        .astype(str)
        .str.strip()
        .ne("")
        .sum()
    )

    unique = df[condition].nunique()

    print(
        f"{condition:<15} | "
        f"non-null: {non_null:<4} | "
        f"non-empty: {non_empty:<4} | "
        f"unique: {unique}"
    )


# ============================================================
# DIFFERENCE FROM ORIGINAL
# ============================================================

print("\n")
print("=" * 70)
print("DIFFERENCE FROM ORIGINAL")
print("=" * 70)


for condition in conditions:

    if condition == "original":

        difference = 0

    else:

        difference = (
            df[condition].astype(str)
            != df["original"].astype(str)
        ).sum()

    percentage = (
        difference / len(df) * 100
        if len(df) > 0
        else 0
    )

    print(
        f"{condition:<15} | "
        f"different: {difference:<4} | "
        f"{percentage:>6.2f}%"
    )


# ============================================================
# TELUGU SCRIPT CHECK
# ============================================================

print("\n")
print("=" * 70)
print("TELUGU UNICODE CHECK")
print("=" * 70)


for condition in [
    "original",
    "romanized",
    "telugu_script",
    "mixed_script",
]:

    count = (
        df[condition]
        .astype(str)
        .apply(contains_telugu)
        .sum()
    )

    percentage = count / len(df) * 100

    print(
        f"{condition:<15} | "
        f"Telugu Unicode rows: {count:<4} | "
        f"{percentage:>6.2f}%"
    )


# ============================================================
# LABEL DISTRIBUTION
# ============================================================

print("\nLabel distribution:")

print(
    df["label"]
    .value_counts()
)


# ============================================================
# SAMPLE TRANSFORMATIONS
# ============================================================

print("\n")
print("=" * 70)
print("SAMPLE TRANSFORMATIONS")
print("=" * 70)

sample_count = min(10, len(df))

sample_df = df.sample(
    n=sample_count,
    random_state=SEED
)


for idx, row in sample_df.iterrows():

    print("\n" + "-" * 70)

    print("ORIGINAL:")
    print(row["original"])

    print("\nROMANIZED:")
    print(row["romanized"])

    print("\nTELUGU SCRIPT:")
    print(row["telugu_script"])

    print("\nMIXED SCRIPT:")
    print(row["mixed_script"])

    print("\nMILD NOISE:")
    print(row["mild_noise"])

    print("\nSEVERE NOISE:")
    print(row["severe_noise"])


# ============================================================
# SAVE
# ============================================================

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)


print("\n")
print("=" * 70)
print("SAVED")
print("=" * 70)

print(OUTPUT_FILE)

print("\nRows:", len(df))
print("Columns:", list(df.columns))

print("\nDONE")
print("=" * 70)