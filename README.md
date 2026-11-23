# House Door Checker

Takes one image and prints **ACCEPT** if it shows a house with its door or gate visible,
otherwise **REJECT**. A house's entrance doorway photographed from outside (for example a
survey photo of the main door with a garland and geotag text) is accepted. Room interiors,
door-detail close-ups (handles, locks), houses with no visible door, and unrelated photos
are rejected.

## Setup on Windows (one time)

1. Install **Python 3.10, 3.11 or 3.12** from https://www.python.org/downloads/
   (tick "Add python.exe to PATH" during install).
2. Unzip this folder anywhere, e.g. `C:\house-checker`.
3. Double-click **`setup_windows.bat`**. It installs the packages and downloads the
   image model (about 290 MB, from GitHub). Takes a few minutes.

## Use it

- **Web page:** double-click **`run_web.bat`**. Your browser opens http://127.0.0.1:5000.
  Choose a photo and click *Check image*. Close the black window to stop it.
- **Command line:** open a Command Prompt in this folder and run
  `check.bat C:\path\to\photo.jpg` (add `--verbose` to see the score).

Everything runs on your own computer; images are not uploaded anywhere.

## How it decides

1. A pretrained CLIP image model (DataComp CLIP ViT-B/32) describes the photo as 512 numbers.
2. A small classifier (`model/probe.json`) turns those numbers into a score from 0 to 1. It was
   trained on 4,055 labelled survey photos: 3,000 accepted property images, and as rejects the
   "Property image not proper", long-distance, house-entrance and washroom folders plus plate close-ups.
3. That score is blended 50/50 with a vote of the 15 most similar photos in
   `model/reference_bank.npz` (the same labelled photos plus any corrections you add).
4. Score 0.68 or more is ACCEPT. Use `--threshold 0.8` (command line) for a stricter check.

Measured accuracy (version 5), 5-fold cross-validation grouped by house ID:
**94.8%** of good property photos accepted and **90.1%** of bad ones rejected
(version 3 managed 90.9% and 70.4% on the same kind of photos).

## Teach it from its mistakes

1. Copy photos that should be accepted into `corrections\accept\` and photos that should be
   rejected into `corrections\reject\`.
2. Run `.venv\Scripts\python add_examples.py`. It prints each photo's verdict before and after.
3. Close and reopen `run_web.bat` so the web page loads them.

Similar-looking photos then get the same verdict. Older models are kept as
`model/probe_v2_backup.json` ... `probe_v4_backup.json`.

## Files

| File | What it is |
|---|---|
| `checker.py` | Core logic |
| `check_house.py` / `check.bat` | Command line tool |
| `app.py` / `run_web.bat` | Web page |
| `setup_windows.bat` | One-time setup |
| `model/probe.json` | Trained classifier weights and settings |
| `model/reference_bank.npz` | Labelled reference photos (features only, not images) |
| `add_examples.py` / `corrections/` | Add your own accept/reject examples |
