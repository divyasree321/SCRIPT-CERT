import re
import random
from pathlib import Path

import pandas as pd
from indic_transliteration import sanscript


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42
random.seed(SEED)


# ============================================================
# SCRIPT DETECTION
# ============================================================

def contains_telugu(text):
    """
    Return True if Telugu Unicode characters are present.
    """
    text = str(text)

    return any(
        "\u0C00" <= ch <= "\u0C7F"
        for ch in text
    )


def contains_latin(text):
    """
    Return True if Latin alphabet characters are present.
    """
    text = str(text)

    return any(
        ("A" <= ch <= "Z") or ("a" <= ch <= "z")
        for ch in text
    )


def classify_script(text):
    """
    Classify text into:
    Telugu
    Latin/Romanized
    Mixed
    Other
    """

    text = str(text)

    has_telugu = contains_telugu(text)
    has_latin = contains_latin(text)

    if has_telugu and has_latin:
        return "Mixed"

    if has_telugu:
        return "Telugu"

    if has_latin:
        return "Latin/Romanized"

    return "Other"


# ============================================================
# TOKENIZATION
# ============================================================

def tokenize_text(text):
    """
    Split text into whitespace-separated units while preserving
    whitespace exactly.

    This avoids splitting Telugu Unicode characters/combining
    marks into separate pieces.
    """

    return re.findall(
        r"\S+|\s+",
        str(text),
        flags=re.UNICODE
    )


# ============================================================
# TELUGU → ROMANIZED
# ============================================================

def telugu_to_romanized(text):
    """
    Convert Telugu-script portions into Roman/ITRANS.

    English words, numbers, punctuation, and whitespace
    are preserved.
    """

    tokens = tokenize_text(text)
    output = []

    for token in tokens:

        # Preserve whitespace
        if not token.strip():
            output.append(token)
            continue

        # Convert only tokens containing Telugu
        if contains_telugu(token):

            try:
                converted = sanscript.transliterate(
                    token,
                    sanscript.TELUGU,
                    sanscript.ITRANS
                )

                output.append(converted)

            except Exception:
                output.append(token)

        else:
            # English / numbers / punctuation remain unchanged
            output.append(token)

    return "".join(output)


# ============================================================
# ROMANIZED → TELUGU
# ============================================================

# Common English words that should remain English.
ENGLISH_WORDS = {
    "the",
    "a",
    "an",
    "is",
    "are",
    "was",
    "were",
    "and",
    "or",
    "but",
    "very",
    "good",
    "bad",
    "best",
    "life",
    "person",
    "persons",
    "people",
    "politician",
    "politics",
    "video",
    "videos",
    "follow",
    "house",
    "future",
    "waste",
    "behaviour",
    "youtube",
    "extraordinary",
}


def looks_like_english(token):
    """
    Heuristic to protect common English words.
    """

    return token.lower() in ENGLISH_WORDS


def romanized_to_telugu(text):
    """
    Convert likely Romanized Telugu tokens into Telugu script.

    Existing Telugu text, English words, numbers, punctuation,
    and whitespace are preserved.
    """

    tokens = tokenize_text(text)
    output = []

    for token in tokens:

        # Preserve whitespace
        if not token.strip():
            output.append(token)
            continue

        # Already Telugu
        if contains_telugu(token):
            output.append(token)
            continue

        # Only process purely alphabetic Roman tokens
        if not re.fullmatch(r"[A-Za-z]+", token):
            output.append(token)
            continue

        # Preserve known English words
        if looks_like_english(token):
            output.append(token)
            continue

        lower = token.lower()

        # Common Romanized Telugu patterns
        roman_telugu_patterns = (
            "aa",
            "ee",
            "ii",
            "oo",
            "uu",
            "ch",
            "dh",
            "th",
            "ph",
            "bh",
            "kh",
            "gh",
            "sh",
            "ng",
            "ny",
            "ck",
            "tt",
            "dd",
            "ll",
            "rr"
        )

        # Common Romanized Telugu endings
        roman_telugu_endings = (
            "am",
            "ani",
            "andi",
            "aku",
            "ki",
            "ku",
            "ni",
            "nu",
            "na",
            "ne",
            "no",
            "ra",
            "re",
            "ri",
            "ro",
            "ru",
            "ga",
            "ge",
            "gi",
            "go",
            "gu",
            "la",
            "le",
            "li",
            "lo",
            "lu",
            "va",
            "ve",
            "vi",
            "vo",
            "vu",
            "tha",
            "thi",
            "tho",
            "thu",
            "dha",
            "dhi",
            "dho",
            "dhu"
        )

        looks_roman_telugu = (
            any(
                pattern in lower
                for pattern in roman_telugu_patterns
            )
            or lower.endswith(roman_telugu_endings)
        )

        # If it does not look like Romanized Telugu,
        # preserve it as English.
        if not looks_roman_telugu:
            output.append(token)
            continue

        try:

            converted = sanscript.transliterate(
                token,
                sanscript.ITRANS,
                sanscript.TELUGU
            )

            output.append(converted)

        except Exception:
            output.append(token)

    return "".join(output)


# ============================================================
# MIXED SCRIPT
# ============================================================

def mixed_script_transform(text, probability=0.50):
    """
    Create a mixed-script version.

    Telugu tokens may be converted to Romanized form
    with the specified probability.

    Existing English words remain unchanged.
    """

    tokens = tokenize_text(text)

    output = []

    for token in tokens:

        if (
            token.strip()
            and contains_telugu(token)
            and random.random() < probability
        ):

            try:

                converted = sanscript.transliterate(
                    token,
                    sanscript.TELUGU,
                    sanscript.ITRANS
                )

                output.append(converted)

            except Exception:

                output.append(token)

        else:

            output.append(token)

    return "".join(output)


# ============================================================
# MILD SPELLING NOISE
# ============================================================

def add_mild_spelling_noise(text):
    """
    Controlled mild spelling variation.
    """

    words = str(text).split()

    new_words = []

    for word in words:

        if len(word) >= 4 and random.random() < 0.25:

            chars = list(word)

            positions = [
                i
                for i, c in enumerate(chars)
                if c.isalpha()
            ]

            if positions:

                i = random.choice(positions)

                chars.insert(
                    i,
                    chars[i]
                )

            word = "".join(chars)

        new_words.append(word)

    return " ".join(new_words)


# ============================================================
# SEVERE SPELLING NOISE
# ============================================================

def add_severe_spelling_noise(text):
    """
    Controlled severe spelling variation.
    """

    words = str(text).split()

    new_words = []

    for word in words:

        if len(word) >= 3 and random.random() < 0.50:

            chars = list(word)

            positions = [
                i
                for i, c in enumerate(chars)
                if c.isalpha()
            ]

            if positions:

                i = random.choice(positions)

                chars.insert(
                    i,
                    chars[i]
                )

                if (
                    random.random() < 0.40
                    and i < len(chars)
                ):

                    chars.insert(
                        i,
                        chars[i]
                    )

            word = "".join(chars)

        new_words.append(word)

    return " ".join(new_words)


# ============================================================
# GENERATE COMPLETE SCRIPT-SHIFT DATASET
# ============================================================

def generate_script_shift_dataset(
    input_csv,
    output_csv
):

    print("=" * 70)
    print("SCRIPT-CERT SCRIPT-SHIFT GENERATION")
    print("=" * 70)

    input_csv = Path(input_csv)
    output_csv = Path(output_csv)

    df = pd.read_csv(input_csv)

    print("\nInput shape:", df.shape)

    # --------------------------------------------------------
    # Check required columns
    # --------------------------------------------------------

    required_columns = [
        "original_text",
        "label"
    ]

    for column in required_columns:

        if column not in df.columns:

            raise ValueError(
                f"Missing required column: {column}"
            )

    # --------------------------------------------------------
    # Original script classification
    # --------------------------------------------------------

    df["original_script"] = (
        df["original_text"]
        .apply(classify_script)
    )

    # --------------------------------------------------------
    # Generate transformations
    # --------------------------------------------------------

    print("\nGenerating transformations...")

    df["original"] = (
        df["original_text"]
    )

    df["romanized"] = (
        df["original_text"]
        .apply(telugu_to_romanized)
    )

    df["telugu_script"] = (
        df["original_text"]
        .apply(romanized_to_telugu)
    )

    df["mixed_script"] = (
        df["original_text"]
        .apply(mixed_script_transform)
    )

    df["mild_noise"] = (
        df["original_text"]
        .apply(add_mild_spelling_noise)
    )

    df["severe_noise"] = (
        df["original_text"]
        .apply(add_severe_spelling_noise)
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output_csv.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        output_csv,
        index=False,
        encoding="utf-8-sig"
    )

    print("\nSaved:")
    print(output_csv)

    print("\nOutput shape:", df.shape)

    print("\nColumns:")

    for column in df.columns:
        print(" -", column)

    print("\nOriginal script distribution:")

    print(
        df["original_script"]
        .value_counts()
    )

    print("\nTransformation check:")

    transformations = [
        "original",
        "romanized",
        "telugu_script",
        "mixed_script",
        "mild_noise",
        "severe_noise"
    ]

    for column in transformations:

        changed = (
            df[column].astype(str)
            != df["original_text"].astype(str)
        ).mean()

        print(
            f"{column:20s}: "
            f"{changed * 100:.2f}% changed"
        )

    print("\n" + "=" * 70)
    print("SCRIPT-SHIFT GENERATION COMPLETE")
    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    project_root = (
        Path(__file__)
        .resolve()
        .parents[2]
    )

    input_file = (
        project_root
        / "data"
        / "script_shift"
        / "script_shift_base.csv"
    )

    output_file = (
        project_root
        / "data"
        / "script_shift"
        / "validation_script_shift_full.csv"
    )

    generate_script_shift_dataset(
        input_file,
        output_file
    )