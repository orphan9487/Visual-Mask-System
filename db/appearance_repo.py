"""User-selected appearance hints used to stabilize SD1.5 generation."""

from __future__ import annotations

from db.connection import get_connection


FIELDS = (
    "gender_presentation", "age_group", "hair_length", "hair_color",
    "hair_style", "glasses", "facial_hair", "framing",
)
DEFAULTS = {
    "gender_presentation": "unspecified", "age_group": "adult",
    "hair_length": "unspecified", "hair_color": "unspecified",
    "hair_style": "unspecified", "glasses": False,
    "facial_hair": "unspecified", "framing": "face_only",
}
ALLOWED = {
    "gender_presentation": {"masculine", "feminine", "neutral", "unspecified"},
    "age_group": {"young_adult", "adult", "middle_aged", "unspecified"},
    "hair_length": {"short", "medium", "long", "unspecified"},
    "hair_color": {"black", "brown", "blonde", "gray", "unspecified"},
    "hair_style": {"straight", "wavy", "curly", "side_part", "center_part", "unspecified"},
    "facial_hair": {"none", "stubble", "beard", "unspecified"},
    "framing": {"face_only", "head_and_shoulders", "portrait", "half_body"},
}


def _normalise(values: dict | None) -> dict:
    values = values or {}
    result = dict(DEFAULTS)
    for field in FIELDS:
        value = values.get(field, DEFAULTS[field])
        if field == "glasses":
            value = bool(value)
        elif value not in ALLOWED[field]:
            raise ValueError(f"appearance field has invalid value: {field}")
        result[field] = value
    return result


def get_appearance(user_id: int) -> dict:
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT gender_presentation, age_group, hair_length, hair_color, "
                "hair_style, glasses, facial_hair, framing "
                "FROM user_appearance_profiles WHERE user_id = %s", (user_id,)
            )
            row = cur.fetchone()
        if not row:
            return dict(DEFAULTS)
        return dict(zip(FIELDS, (row[0], row[1], row[2], row[3], row[4], bool(row[5]), row[6], row[7])))
    finally:
        conn.close()


def save_appearance(user_id: int, values: dict) -> dict:
    appearance = _normalise(values)
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO user_appearance_profiles "
                "(user_id, gender_presentation, age_group, hair_length, hair_color, hair_style, glasses, facial_hair, framing) "
                "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s) "
                "ON DUPLICATE KEY UPDATE gender_presentation=VALUES(gender_presentation), age_group=VALUES(age_group), "
                "hair_length=VALUES(hair_length), hair_color=VALUES(hair_color), hair_style=VALUES(hair_style), "
                "glasses=VALUES(glasses), facial_hair=VALUES(facial_hair), framing=VALUES(framing)",
                (user_id, appearance["gender_presentation"], appearance["age_group"], appearance["hair_length"],
                 appearance["hair_color"], appearance["hair_style"], int(appearance["glasses"]),
                 appearance["facial_hair"], appearance["framing"]),
            )
        conn.commit()
        return appearance
    finally:
        conn.close()


def prompt_fragment(user_id: int) -> str:
    """Convert selected options into a conservative English SD prompt fragment."""
    a = get_appearance(user_id)
    parts = []
    labels = {
        "masculine": "masculine presentation", "feminine": "feminine presentation", "neutral": "androgynous presentation",
        "young_adult": "young adult", "adult": "adult", "middle_aged": "middle-aged",
        "short": "short hair", "medium": "medium-length hair", "long": "long hair",
        "black": "black hair", "brown": "brown hair", "blonde": "blonde hair", "gray": "gray hair",
        "straight": "straight hair", "wavy": "wavy hair", "curly": "curly hair", "side_part": "side-parted hair", "center_part": "center-parted hair",
        "none": "clean-shaven", "stubble": "light stubble", "beard": "beard",
        "face_only": "face-only portrait, no shoulders", "head_and_shoulders": "head-and-shoulders portrait", "portrait": "portrait", "half_body": "half-body portrait",
    }
    # FaceID currently uses a fixed face-only composition. Keep the stored
    # framing option for future layouts, but do not let an older
    # head-and-shoulders selection reintroduce shoulders into this mode.
    for field in ("gender_presentation", "age_group", "hair_length", "hair_color", "hair_style", "facial_hair"):
        value = a[field]
        if value != "unspecified":
            parts.append(labels[value])
    if a["glasses"]:
        parts.append("wearing glasses")
    return ", ".join(parts)
