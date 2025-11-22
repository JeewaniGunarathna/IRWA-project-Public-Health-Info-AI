# Very small helper to normalize country names -> canonical forms.
def canonical_country(name: str) -> str:
    if not name:
        return ""
    s = name.strip().lower()
    # Add more aliases as needed
    aliases = {
        "usa": "united states",
        "us": "united states",
        "u.s.": "united states",
        "u.s.a.": "united states",
        "sri lanka": "sri lanka",
        "india": "india",
        "united kingdom": "united kingdom",
        "uk": "united kingdom",
    }
    return aliases.get(s, s)
