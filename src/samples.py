"""
Curated sample comments across different toxicity categories for testing and demonstrations.
"""

SAMPLE_COMMENTS = [
    {
        "category": "Clean / Positive",
        "comment": "Thank you so much for this helpful explanation! You did a fantastic job on this project.",
        "expected_flag": "Clean"
    },
    {
        "category": "Clean / Disagreement (Non-Toxic)",
        "comment": "I respectfully disagree with your conclusion here. In my experience, the data points in a different direction.",
        "expected_flag": "Clean"
    },
    {
        "category": "Insult",
        "comment": "You are completely pathetic and braindead. Nobody asked for your stupid opinion.",
        "expected_flag": "Insult"
    },
    {
        "category": "Threat / Violent Cyberbullying",
        "comment": "I will find where you live and beat you to a pulp, watch your back.",
        "expected_flag": "Threat"
    },
    {
        "category": "Identity Attack / Hate Speech",
        "comment": "All people from your background are disgusting parasites and should be wiped out.",
        "expected_flag": "Identity Hate"
    },
    {
        "category": "Obscene / Profanity",
        "comment": "Shut the fuck up you miserable piece of shit, get the hell out of here.",
        "expected_flag": "Obscene / Severe"
    },
    {
        "category": "Obfuscated / Leetspeak Cyberbullying",
        "comment": "Ur a useless b!tch, go kill urself u f*cking l0ser.",
        "expected_flag": "Severe Toxic / Threat"
    }
]
