This directory contains the external Flask app used for human ranking studies.

Contents:

- `app.py`: the primary ranking-study Flask app.
- `app_expanded.py`: a related expanded app variant kept with the study code.
- `legacy_variant_2/`: a second older app variant preserved for reference.
- `make_ims.py`, `make_transparent_ims.py`, `mi.py`: study-specific image
  preparation helpers.
- `templates/`: HTML templates for the main study app.

This is not part of the core reward-model training runtime, but it is kept as
the user-study application layer behind the paper's ranking data.
