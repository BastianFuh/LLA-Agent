import difflib

import pycountry


def get_language_code(language_name: str) -> str | None:
    language_name = language_name.lower()

    # Build a list of language names and their objects
    all_languages = {
        lang.name.lower(): lang
        for lang in pycountry.languages
        if hasattr(lang, "alpha_2")
    }

    # Try direct match first
    if language_name in all_languages:
        return all_languages[language_name].alpha_2

    # Fuzzy match
    closest_matches = difflib.get_close_matches(
        language_name, all_languages.keys(), n=1, cutoff=0.8
    )
    if closest_matches:
        match = closest_matches[0]
        return all_languages[match].alpha_2

    return None
