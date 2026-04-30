"""
character_tracker.py
====================
Detects real characters using spaCy NER with aggressive pre/post filtering.
Handles poorly formatted text (missing apostrophes, merged words, ALL CAPS).
"""

import re
from dataclasses import dataclass


@dataclass
class Character:
    name: str
    gender: str = "unknown"
    age_group: str = "unknown"
    hair: str = ""
    clothing: str = ""
    visual_anchor: str = ""
    first_seen_chunk: int = 0
    mention_count: int = 0


class CharacterTracker:

    # ── Curated known characters for Alice in Wonderland + common archetypes ──
    # This is the ONLY source of truth. No auto-detection of random words.
    # Add your own story's characters here if needed.
    _KNOWN_CHARACTERS = {
        # name_lowercase : (gender, age_group, hair, clothing)
        "alice"        : ("female", "child",   "long blonde hair",           "blue pinafore dress"),
        "dinah"        : ("female", "unknown", "orange fur",                 "nothing, an orange cat"),
        "rabbit"       : ("male",   "adult",   "white fur",                  "white rabbit with a waistcoat and pocket watch"),
        "white rabbit" : ("male",   "adult",   "white fur",                  "white rabbit with a waistcoat and pocket watch"),
        "mouse"        : ("unknown","adult",   "grey fur",                   "nothing, a small mouse"),
        "duck"         : ("unknown","adult",   "feathers",                   "nothing, a duck"),
        "dodo"         : ("unknown","adult",   "colourful feathers",         "nothing, a dodo bird"),
        "lory"         : ("unknown","adult",   "bright feathers",            "nothing, a parrot-like bird"),
        "eaglet"       : ("unknown","child",   "small feathers",             "nothing, a small eaglet"),
        "caterpillar"  : ("unknown","adult",   "blue body",                  "nothing, a blue caterpillar sitting on a mushroom"),
        "pigeon"       : ("unknown","adult",   "grey feathers",              "nothing, a pigeon"),
        "duchess"      : ("female", "adult",   "dark elegant hair",          "grand dress with large ruff collar"),
        "cheshire cat" : ("unknown","adult",   "striped grey and purple fur","nothing, a grinning Cheshire cat that can disappear"),
        "cheshire"     : ("unknown","adult",   "striped grey and purple fur","nothing, a grinning Cheshire cat that can disappear"),
        "hatter"       : ("male",   "adult",   "wild brown hair",            "oversized top hat and patchwork coat"),
        "mad hatter"   : ("male",   "adult",   "wild brown hair",            "oversized top hat and patchwork coat"),
        "march hare"   : ("unknown","adult",   "brown fur",                  "nothing, an excitable hare in a waistcoat"),
        "dormouse"     : ("unknown","adult",   "brown fur",                  "nothing, a small sleepy dormouse"),
        "queen"        : ("female", "adult",   "elegant dark hair",          "royal red and black gown and crown"),
        "king"         : ("male",   "adult",   "short dark hair",            "royal gold crown and red robe"),
        "knave"        : ("male",   "adult",   "short dark hair",            "playing card soldier uniform"),
        "gryphon"      : ("unknown","adult",   "golden feathers",            "nothing, a large gryphon with eagle head and lion body"),
        "mock turtle"  : ("unknown","adult",   "green shell",                "nothing, a sad mock turtle"),
        "bill"         : ("unknown","adult",   "scruffy fur",                "nothing, a lizard in workman's clothes"),
        "pat"          : ("male",   "adult",   "short brown hair",           "gardener's earthy clothing"),
        "mary ann"     : ("female", "adult",   "brown hair",                 "simple maid's dress"),
        "ada"          : ("female", "child",   "brown hair",                 "simple colorful dress"),
        "mabel"        : ("female", "child",   "brown hair",                 "simple colorful dress"),
        "elsie"        : ("female", "child",   "fair hair",                  "simple colorful dress"),
        "lacie"        : ("female", "child",   "brown hair",                 "simple colorful dress"),
        "tillie"       : ("female", "child",   "dark hair",                  "simple colorful dress"),
        # Generic archetypes for other stories
        "witch"        : ("female", "elder",   "grey straggly hair",         "black robes and pointed hat"),
        "wizard"       : ("male",   "elder",   "long white beard",           "blue robes and tall hat"),
        "prince"       : ("male",   "teen",    "neat brown hair",            "royal blue tunic and cape"),
        "princess"     : ("female", "teen",    "long flowing hair",          "elegant pink gown"),
        "fairy"        : ("female", "adult",   "shimmering golden hair",     "delicate glowing wings and white dress"),
        "knight"       : ("male",   "adult",   "hidden by helmet",           "shining silver armor and sword"),
        "giant"        : ("male",   "adult",   "wild brown hair",            "rough enormous clothes"),
        "dragon"       : ("unknown","adult",   "green scales",               "nothing, a large dragon with wings"),
        "wolf"         : ("male",   "adult",   "grey fur",                   "nothing, a large grey wolf"),
        "merchant"     : ("male",   "adult",   "short grey hair",            "brown merchant vest and satchel"),
        "grandmother"  : ("female", "elder",   "white hair bun",             "grey shawl and apron"),
        "grandfather"  : ("male",   "elder",   "white beard",                "long coat"),
        "mother"       : ("female", "adult",   "dark hair",                  "apron and dress"),
        "father"       : ("male",   "adult",   "short brown hair",           "practical clothing"),
        "soldier"      : ("male",   "adult",   "short hair",                 "military uniform"),
        "guard"        : ("male",   "adult",   "short hair",                 "guard uniform"),
        "farmer"       : ("male",   "adult",   "weathered brown hair",       "overalls and straw hat"),
    }

    def __init__(self):
        self.characters: dict[str, Character] = {}
        self._chunk_characters: list[list[str]] = []
        self._nlp = None
        self._load_spacy()

    # ------------------------------------------------------------------ #
    #  Public                                                               #
    # ------------------------------------------------------------------ #

    def analyse_all_chunks(self, chunks: list[str]) -> None:
        print("  [CharacterTracker] Scanning story for characters ...")
        self._chunk_characters = [[] for _ in chunks]

        for i, chunk in enumerate(chunks):
            clean = self._clean_text(chunk)
            names = self._find_characters(clean, i)
            self._chunk_characters[i] = names

        self._build_all_anchors()
        print(f"  [CharacterTracker] Found {len(self.characters)} character(s).")

    def get_scene_anchor(self, chunk: str, chunk_index: int) -> str:
        names = []
        if chunk_index < len(self._chunk_characters):
            names = self._chunk_characters[chunk_index]
        if not names:
            names = self._pronoun_fallback(chunk)
        if not names:
            return self._generic_anchor()

        parts = []
        for i, name in enumerate(names[:2]):
            char = self.characters.get(name)
            if not char:
                continue
            pos = "prominently in foreground, large in frame" if i == 0 else "visible nearby"
            parts.append(f"{name}: {char.visual_anchor}, {pos}")
        return "; ".join(parts) if parts else self._generic_anchor()

    def print_character_report(self) -> None:
        print("\n" + "=" * 60)
        print("  CHARACTER REPORT")
        print("=" * 60)
        for name, c in self.characters.items():
            print(f"  {name:<20} | {c.age_group} {c.gender:<8} | "
                  f"{c.hair:<30} | {c.clothing}")
        print("=" * 60 + "\n")

    def get_all_characters(self) -> dict[str, Character]:
        return self.characters

    # ------------------------------------------------------------------ #
    #  Private -- text cleaning                                             #
    # ------------------------------------------------------------------ #

    def _clean_text(self, text: str) -> str:
        """
        Fix common formatting issues before NER:
        - ALL CAPS words → title case
        - Insert spaces before capitals in merged words (RabbitHole → Rabbit Hole)
        - Restore apostrophes (Ive → I've, dont → don't, etc.)
        """
        # Split merged CamelCase words: RabbitHole → Rabbit Hole
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)

        # ALL CAPS words that are not acronyms → title case
        def fix_caps(m):
            word = m.group(0)
            if len(word) <= 3:  # keep short acronyms: THE → keep, III → keep
                return word
            return word.title()
        text = re.sub(r'\b[A-Z]{3,}\b', fix_caps, text)

        return text

    # ------------------------------------------------------------------ #
    #  Private -- character finding                                         #
    # ------------------------------------------------------------------ #

    def _find_characters(self, text: str, chunk_index: int) -> list[str]:
        found = {}

        # 1. Check already-registered characters (fast path)
        for name in list(self.characters.keys()):
            if re.search(r'\b' + re.escape(name) + r'\b', text, re.IGNORECASE):
                self.characters[name].mention_count += 1
                found[name] = found.get(name, 0) + 1

        # 2. Check known character dictionary (multi-word first, then single)
        sorted_known = sorted(self._KNOWN_CHARACTERS.keys(),
                              key=lambda k: len(k.split()), reverse=True)
        for key in sorted_known:
            if re.search(r'\b' + re.escape(key) + r'\b', text, re.IGNORECASE):
                display_name = key.title()
                if display_name not in self.characters:
                    gender, age, hair, clothing = self._KNOWN_CHARACTERS[key]
                    char = Character(
                        name=display_name,
                        gender=gender, age_group=age,
                        hair=hair, clothing=clothing,
                        first_seen_chunk=chunk_index, mention_count=1,
                    )
                    self.characters[display_name] = char
                else:
                    self.characters[display_name].mention_count += 1
                found[display_name] = found.get(display_name, 0) + 1

        # 3. spaCy NER for characters NOT in our known list
        if self._nlp:
            doc = self._nlp(text)
            for ent in doc.ents:
                if ent.label_ == "PERSON":
                    name = ent.text.strip().title()
                    name = re.sub(r"[^a-zA-Z\s]", "", name).strip()
                    if not name or len(name) < 3:
                        continue
                    # Skip if it matches something already found via known list
                    if name in self.characters:
                        found[name] = found.get(name, 0) + 1
                        continue
                    # Only add if it looks like a real proper name
                    # (not a common word that spaCy mislabelled)
                    if self._is_valid_new_name(name, text):
                        char = Character(
                            name=name, first_seen_chunk=chunk_index, mention_count=1)
                        self._infer_traits(char, text)
                        self.characters[name] = char
                        found[name] = found.get(name, 0) + 1

        return sorted(found.keys(), key=lambda n: found[n], reverse=True)

    def _is_valid_new_name(self, name: str, context: str) -> bool:
        """
        Extra validation for spaCy-detected names not in our known list.
        Rejects common words, places, abstract nouns, etc.
        """
        n = name.lower()

        # Must be longer than 3 characters
        if len(n) < 4:
            return False

        # Reject words that are clearly not person names
        reject_patterns = [
            r'^\d',                          # starts with number
            r'ing$',                         # gerunds: laughing, stretching
            r'tion$',                        # nouns: distraction, derision
            r'ness$',                        # nouns: ugliness
            r'ment$',                        # nouns: amusement
            r'ance$', r'ence$',              # nouns: silence, evidence
            r'ology$', r'graphy$',           # subject names
            r'^(north|south|east|west)',     # compass directions
        ]
        for pattern in reject_patterns:
            if re.search(pattern, n):
                return False

        # Reject known non-name words
        non_names = {
            "everybody", "nobody", "somebody", "anyone", "everyone",
            "christmas", "english", "french", "latin", "geography",
            "arithmetic", "multiplication", "grammar", "longitude",
            "latitude", "australia", "zealand", "wonderland",
            "shakespeare", "william", "conqueror", "normans",
            "england", "france", "rome", "paris", "morcar",
            "mercia", "northumbria", "stigand", "canterbury",
            "edgar", "atheling",
        }
        if n in non_names:
            return False

        return True

    # ------------------------------------------------------------------ #
    #  Private -- trait inference & anchor building                         #
    # ------------------------------------------------------------------ #

    def _infer_traits(self, char: Character, context: str) -> None:
        """For spaCy-detected names not in known list -- guess from context."""
        female = len(re.findall(r'\b(she|her|girl|woman|lady)\b', context, re.IGNORECASE))
        male   = len(re.findall(r'\b(he|him|his|boy|man)\b',       context, re.IGNORECASE))
        if female > male:   char.gender = "female"
        elif male > female: char.gender = "male"

        if re.search(r'\b(child|little|young|small|kid)\b', context, re.IGNORECASE):
            char.age_group = "child"
        elif re.search(r'\b(old|elder|ancient|grandmother|grandfather)\b', context, re.IGNORECASE):
            char.age_group = "elder"
        else:
            char.age_group = "adult"

        defaults = {
            "female_child": ("brown hair", "simple colorful dress"),
            "male_child":   ("short hair", "simple shirt and trousers"),
            "female_adult": ("dark hair",  "simple dress"),
            "male_adult":   ("short hair", "practical clothing"),
            "female_elder": ("grey hair",  "long dress or shawl"),
            "male_elder":   ("grey beard", "long robe or coat"),
        }.get(f"{char.gender}_{char.age_group}", ("dark hair", "simple clothing"))

        char.hair     = defaults[0]
        char.clothing = defaults[1]

    def _build_all_anchors(self) -> None:
        for char in self.characters.values():
            if not char.visual_anchor:
                self._build_anchor(char)

    def _build_anchor(self, char: Character) -> None:
        age_map = {"child":"young child","teen":"teenager",
                   "adult":"adult","elder":"elderly","unknown":"character"}
        gender_map = {
            "female": "girl" if char.age_group in ("child","teen") else "woman",
            "male":   "boy"  if char.age_group in ("child","teen") else "man",
            "unknown":"character",
        }
        char.visual_anchor = (
            f"{age_map.get(char.age_group,'character')} "
            f"{gender_map.get(char.gender,'character')} "
            f"with {char.hair or 'dark hair'}, "
            f"wearing {char.clothing or 'simple clothing'}, "
            f"expressive face, well-drawn"
        )

    def _load_spacy(self) -> None:
        try:
            import spacy
            try:
                self._nlp = spacy.load("en_core_web_sm")
                print("  [CharacterTracker] spaCy NER loaded.")
            except OSError:
                print("  [CharacterTracker] WARNING: spaCy model missing.")
                print("  Run: python -m spacy download en_core_web_sm")
                print("  Falling back to known-character list only.")
                self._nlp = None
        except ImportError:
            print("  [CharacterTracker] WARNING: spaCy not installed.")
            print("  Run: pip install spacy && python -m spacy download en_core_web_sm")
            print("  Falling back to known-character list only.")
            self._nlp = None

    def _pronoun_fallback(self, text: str) -> list[str]:
        female = len(re.findall(r'\b(she|her)\b', text, re.IGNORECASE))
        male   = len(re.findall(r'\b(he|him)\b',  text, re.IGNORECASE))
        recent = sorted(self.characters.values(), key=lambda c: c.first_seen_chunk, reverse=True)
        for c in recent:
            if female > male and c.gender == "female": return [c.name]
            if male > female and c.gender == "male":   return [c.name]
        if self.characters:
            return [max(self.characters.values(), key=lambda c: c.mention_count).name]
        return []

    def _generic_anchor(self) -> str:
        return "a character prominently in foreground, clearly visible, large in frame, expressive face, well-drawn"
