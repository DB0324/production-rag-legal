"""
Identifies "vague" auto-generated questions -- generic phrasing with NO
specific named case/party reference -- vs "well-formed" questions that
name a specific case, even if using similar phrasing.
"""
import json
import re

EVAL_PATH = "data/eval/indiclegalqa_filtered.json"

# A question counts as "specific" if it contains a case-name marker:
# "versus"/"vs"/"v." between two capitalized words, OR a specific date.
CASE_NAME_MARKER = re.compile(r'\b(versus|vs\.?|v\.)\b', re.IGNORECASE)
SPECIFIC_DATE_MARKER = re.compile(r'\b\d{1,2}(st|nd|rd|th)?\s+(January|February|March|April|May|June|July|August|September|October|November|December)\b', re.IGNORECASE)

GENERIC_PHRASES = [
    "broader issue",
    "broader legal principle",
    "broader principle",
]

def is_vague(question: str) -> bool:
    q_lower = question.lower()

    has_generic_phrase = any(p in q_lower for p in GENERIC_PHRASES)
    if not has_generic_phrase:
        return False

    # Even with generic phrasing, if it names a specific case/date, it's answerable
    has_case_name = bool(CASE_NAME_MARKER.search(question))
    has_specific_date = bool(SPECIFIC_DATE_MARKER.search(question))

    return not (has_case_name or has_specific_date)


def main():
    with open(EVAL_PATH) as f:
        data = json.load(f)

    vague = [item for item in data if is_vague(item["question"])]
    clean = [item for item in data if not is_vague(item["question"])]

    print(f"Total questions: {len(data)}")
    print(f"Flagged as vague: {len(vague)} ({100*len(vague)/len(data):.1f}%)")
    print(f"Clean/specific: {len(clean)} ({100*len(clean)/len(data):.1f}%)")

    with open("data/eval/eval_set_vague.json", "w") as f:
        json.dump(vague, f, indent=2)
    with open("data/eval/eval_set_clean.json", "w") as f:
        json.dump(clean, f, indent=2)

    print("\nAll vague questions (should be genuinely generic, no case name):")
    for item in vague:
        print(f"  - {item['question']}")

    print("\nSaved: data/eval/eval_set_vague.json, data/eval/eval_set_clean.json")


if __name__ == "__main__":
    main()
