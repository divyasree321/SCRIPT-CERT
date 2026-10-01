import pandas as pd
import os

INPUT_FILE = "results/predictions/semantic_audit_200.csv"
OUTPUT_FILE = "results/predictions/semantic_audit_200.csv"

# Columns to audit
AUDITS = [
    ("romanized_preserved", "romanized", "ROMANIZED"),
    ("telugu_script_preserved", "telugu_script", "TELUGU SCRIPT"),
    ("mixed_script_preserved", "mixed_script", "MIXED SCRIPT"),
    ("mild_noise_preserved", "mild_noise", "MILD NOISE"),
    ("severe_noise_preserved", "severe_noise", "SEVERE NOISE"),
]

# Load data
df = pd.read_csv(INPUT_FILE, encoding="utf-8-sig")

print("=" * 70)
print("SCRIPT-CERT SEMANTIC PRESERVATION AUDIT")
print("=" * 70)
print(f"Rows: {len(df)}")
print()
print("Instructions:")
print("  1 = Meaning is preserved")
print("  0 = Meaning is changed / corrupted")
print("  q = Quit and save progress")
print("=" * 70)

total_questions = len(df) * len(AUDITS)
completed = 0

# Count already completed answers
for col, _, _ in AUDITS:
    if col in df.columns:
        completed += df[col].notna().sum()

print(f"Already completed: {completed}/{total_questions}")
print()

for row_idx in range(len(df)):

    original = str(df.loc[row_idx, "original_text"])
    row_id = df.loc[row_idx, "id"]

    print("\n")
    print("#" * 80)
    print(f"ROW {row_idx + 1}/{len(df)}   |   ID: {row_id}")
    print("#" * 80)

    print("\nORIGINAL:")
    print(original)

    for answer_col, transformed_col, condition_name in AUDITS:

        # Skip if already answered
        existing = df.loc[row_idx, answer_col]

        if pd.notna(existing) and str(existing).strip() in ["0", "1"]:
            continue

        transformed = str(df.loc[row_idx, transformed_col])

        print("\n" + "-" * 80)
        print(f"CONDITION: {condition_name}")
        print("-" * 80)

        print("\nTRANSFORMED:")
        print(transformed)

        print("\nDoes the transformed sentence preserve the ORIGINAL meaning?")
        print("1 = YES, meaning preserved")
        print("0 = NO, meaning changed/corrupted")
        print("q = Quit and save progress")

        while True:
            answer = input("\nYour answer [1/0/q]: ").strip().lower()

            if answer == "1":
                df.loc[row_idx, answer_col] = 1
                completed += 1
                break

            elif answer == "0":
                df.loc[row_idx, answer_col] = 0
                completed += 1
                break

            elif answer == "q":
                df.to_csv(
                    OUTPUT_FILE,
                    index=False,
                    encoding="utf-8-sig"
                )

                print("\n" + "=" * 70)
                print("PROGRESS SAVED")
                print(f"Completed: {completed}/{total_questions}")
                print(f"Saved to: {OUTPUT_FILE}")
                print("=" * 70)

                raise SystemExit

            else:
                print("Please enter only 1, 0, or q.")

        # Save after EVERY answer
        df.to_csv(
            OUTPUT_FILE,
            index=False,
            encoding="utf-8-sig"
        )

print("\n" + "=" * 70)
print("SEMANTIC AUDIT COMPLETED")
print("=" * 70)

print(f"Total answers: {total_questions}")

print("\nResults:")

for answer_col, _, condition_name in AUDITS:

    values = pd.to_numeric(df[answer_col], errors="coerce")

    preserved = (values == 1).sum()
    changed = (values == 0).sum()
    total = preserved + changed

    if total > 0:
        percentage = preserved / total * 100
    else:
        percentage = 0

    print(
        f"{condition_name:20s}: "
        f"{preserved}/{total} preserved "
        f"({percentage:.2f}%), "
        f"{changed} changed"
    )

print(f"\nSaved: {OUTPUT_FILE}")