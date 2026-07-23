"""
Convert iPhone HEIC captures to JPEG for the field-data pipeline.

The pipeline reads images with cv2.imread, which cannot decode HEIC, and
intake_check.IMAGE_EXTENSIONS does not list .heic - a HEIC-only child folder
is silently reported as having zero photos. Converting to JPEG is what makes
field captures visible to intake.

EXIF is copied across explicitly: DateTimeOriginal is what pins each child's
age at assessment, and ImageMagick's HEIC path drops it silently. Orientation
is baked into the pixels and the tag normalised to 1, so cv2 and the pose
scorer agree on which way is up regardless of how either reads the tag.

Originals are never modified. Existing .jpg outputs are skipped unless --force.

Usage:
    PYTHONPATH=. .venv/bin/python scripts/heic_to_jpeg.py <dir> [--quality 90]
"""
import argparse
import sys
from pathlib import Path

import pillow_heif
from PIL import Image, ImageOps

pillow_heif.register_heif_opener()

ORIENTATION_TAG = 0x0112
EXIF_IFD_TAG = 0x8769
DATETIME_ORIGINAL_TAG = 36867


def convert(src: Path, quality: int, force: bool) -> tuple[bool, str]:
    """Convert one HEIC to JPEG beside it. Returns (converted, reason)."""
    dst = src.with_suffix(".jpg")
    if dst.exists() and not force:
        return False, "exists"

    try:
        im = Image.open(src)
    except Exception as e:
        return False, f"unreadable: {e}"

    exif = im.getexif()
    original_dto = exif.get_ifd(EXIF_IFD_TAG).get(DATETIME_ORIGINAL_TAG)

    # Bake rotation into pixels, then declare the image upright, so nothing
    # downstream can rotate it a second time.
    im = ImageOps.exif_transpose(im)
    exif[ORIENTATION_TAG] = 1

    im.convert("RGB").save(dst, "JPEG", quality=quality, exif=exif.tobytes())

    # Verify the timestamp actually landed rather than trusting the write.
    written = Image.open(dst).getexif().get_ifd(EXIF_IFD_TAG).get(
        DATETIME_ORIGINAL_TAG
    )
    if original_dto and written != original_dto:
        return False, f"EXIF LOST (src={original_dto} dst={written})"
    return True, "ok"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--quality", type=int, default=90)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    files = sorted(args.root.rglob("*.HEIC")) + sorted(args.root.rglob("*.heic"))
    if not files:
        print(f"no HEIC files under {args.root}", file=sys.stderr)
        return 1

    done = skipped = 0
    failures: list[str] = []
    for f in files:
        ok, reason = convert(f, args.quality, args.force)
        if ok:
            done += 1
        elif reason == "exists":
            skipped += 1
        else:
            failures.append(f"{f.name}: {reason}")

    print(f"converted={done} skipped={skipped} failed={len(failures)}")
    for x in failures:
        print(f"  FAIL {x}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
