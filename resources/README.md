# Local LivePortrait assets

Personal source photos, rendered caches and expression templates are not
published in this repository. Do not upload participant photos or face data.

The current backend reads `configs/henry_liveportrait_v1.json` (legacy filename)
and expects these locally prepared motion templates:

```text
resources/henry_liveportrait_v1/templates/
  neutral.pkl
  joy.pkl
  sadness.pkl
  anger.pkl
  surprise.pkl
  fear.pkl
  disgust.pkl
```

Obtain the team's approved templates separately, or prepare replacements using
the official LivePortrait template-making workflow and verify each expression.
The legacy `source.png` and `cache/*.jpg` fields are not needed for uploaded
portrait masks; the backend must not fall back to another person's image when
a user's render fails.

Only use trusted pickle templates. Python pickle is not a safe interchange
format for untrusted files. Redistribute third-party assets only with permission
and in accordance with their licenses.
