BANNED_LETTERS = {"J", "Z"}

DIALOGUE_SCRIPT = [
    {"diana": "Hi! I am Diana.", "user": "HI"},
    {"diana": "Can you sign GOOD?", "user": "GOOD"},
    {"diana": "Well done! Bye!", "user": "BYE"},
]


def validate_dialogue(script=DIALOGUE_SCRIPT):
    for row, item in enumerate(script):
        for speaker in ("diana", "user"):
            text = item[speaker].upper()
            banned = sorted(BANNED_LETTERS.intersection(set(text)))
            if banned:
                raise ValueError(f"Row {row} [{speaker}] contains banned letters: {banned}")
        if not any(c.isalpha() for c in item["user"]):
            raise ValueError(f"Row {row} has no signable letters in user text")
    return True
