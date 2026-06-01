BANNED_LETTERS = {"J", "Z"}

DIALOGUE_SCRIPT = [
    {"diana": "Hi X. I am Diana. Let us learn ASL.", "user": "HI DIANA"},
    {"diana": "Great. Can you tell me your mood?", "user": "I AM FINE"},
    {"diana": "Nice. Can you say you like ASL?", "user": "I LIKE ASL"},
    {"diana": "Good work. Can you say this is fun?", "user": "THIS IS FUN"},
    {"diana": "One more. Can you thank me?", "user": "THANK DIANA"},
]


def validate_dialogue(script=DIALOGUE_SCRIPT):
    for row, item in enumerate(script):
        for speaker in ("diana", "user"):
            text = item[speaker].upper()
            banned = sorted(BANNED_LETTERS.intersection(text))
            if banned:
                raise ValueError(f"Dialogue row {row} speaker {speaker} contains banned letters: {banned}")
        if not any(char.isalpha() for char in item["user"]):
            raise ValueError(f"Dialogue row {row} has no signable user letters")
    return True
