#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Set, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


COCO_IMAGE_BASE_URL = "http://images.cocodataset.org/unlabeled2017"

DEFAULT_SPLIT = "val"
DEFAULT_NUM_RANDOM = 100
DEFAULT_SEED = 42
DEFAULT_WORKERS = 16
DEFAULT_RETRIES = 3
DEFAULT_TIMEOUT = 30


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download a small subset of COCO 2017 Unlabeled images required by CIRCO val set.")

    parser.add_argument("--dataset-root", type=Path, required=True,
        help="Path to the CIRCO dataset root. Example: /data/CIRCO")

    parser.add_argument("--split", choices=("val", "test"), default=DEFAULT_SPLIT,
        help=f"CIRCO split to use (default: {DEFAULT_SPLIT})")

    parser.add_argument("--num-random", type=int, default=DEFAULT_NUM_RANDOM, metavar="N",
        help=f"Number of additional random COCO images to download (default: {DEFAULT_NUM_RANDOM})")

    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
        help=f"Random seed for reproducible image selection (default: {DEFAULT_SEED})")

    parser.add_argument("--output-dir", type=Path, default=None, metavar="PATH",
        help="Directory where COCO images will be stored. Default: <dataset-root>/COCO2017_unlabeled/unlabeled2017")

    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, metavar="N",
        help=f"Number of concurrent download workers (default: {DEFAULT_WORKERS})")

    parser.add_argument("--retries", type=int, default=DEFAULT_RETRIES, metavar="N",
        help=f"Number of retries after an initial failed download (default: {DEFAULT_RETRIES})")

    parser.add_argument("--dry-run", action="store_true",
        help="Show what would be downloaded without downloading anything.")

    args = parser.parse_args()

    if args.num_random < 0:
        parser.error("--num-random must be >= 0")

    if args.workers < 1:
        parser.error("--workers must be >= 1")

    if args.retries < 0:
        parser.error("--retries must be >= 0")

    return args


# ---------------------------------------------------------------------------
# Path handling
# ---------------------------------------------------------------------------

def resolve_paths(args: argparse.Namespace) -> Tuple[Path, Path, Path]:
    """
    Resolve all dataset paths from --dataset-root.

    Returns:
        val/test annotation path
        COCO metadata path
        output image directory
    """

    dataset_root = args.dataset_root.expanduser().resolve()

    split_json = dataset_root / "annotations" / f"{args.split}.json"

    coco_metadata = (
        dataset_root
        / "COCO2017_unlabeled"
        / "annotations"
        / "image_info_unlabeled2017.json"
    )

    if args.output_dir is not None:
        output_dir = args.output_dir.expanduser().resolve()
    else:
        output_dir = (
            dataset_root
            / "COCO2017_unlabeled"
            / "unlabeled2017"
        )

    return split_json, coco_metadata, output_dir


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_dataset_structure(
    dataset_root: Path,
    split_json: Path,
    coco_metadata: Path,
    output_dir: Path,
) -> None:
    """Validate required files and create output directory."""

    if not dataset_root.exists():
        raise FileNotFoundError(
            f"Dataset root does not exist:\n  {dataset_root}"
        )

    if not dataset_root.is_dir():
        raise NotADirectoryError(
            f"Dataset root is not a directory:\n  {dataset_root}"
        )

    if not split_json.is_file():
        raise FileNotFoundError(
            f"CIRCO annotation file not found:\n  {split_json}\n\n"
            f"Please check --dataset-root and --split."
        )

    if not coco_metadata.is_file():
        raise FileNotFoundError(
            "COCO metadata file not found:\n"
            f"  {coco_metadata}\n\n"
            "Expected:\n"
            "  COCO2017_unlabeled/annotations/"
            "image_info_unlabeled2017.json"
        )

    if not output_dir.exists():
        print(f"Creating output directory:\n  {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)

    if not output_dir.is_dir():
        raise NotADirectoryError(
            f"Output path is not a directory:\n  {output_dir}"
        )


# ---------------------------------------------------------------------------
# JSON loading
# ---------------------------------------------------------------------------

def load_json(path: Path):
    """Load a JSON file."""

    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid JSON file:\n  {path}\n\n"
            f"JSON error: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# CIRCO image IDs
# ---------------------------------------------------------------------------

def load_circo_required_ids(split_json: Path) -> Set[int]:
    """
    Extract all image IDs required by CIRCO.

    Required IDs consist of:
        - reference_img_id
        - every ID in gt_img_ids
    """

    data = load_json(split_json)

    if not isinstance(data, list):
        raise ValueError(
            f"Expected CIRCO annotation file to contain a list:\n  {split_json}"
        )

    required_ids: Set[int] = set()

    for index, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(
                f"Invalid annotation at index {index} in {split_json}"
            )

        if "reference_img_id" not in item:
            raise KeyError(
                f"Missing 'reference_img_id' at index {index} in {split_json}"
            )

        if "gt_img_ids" not in item:
            raise KeyError(
                f"Missing 'gt_img_ids' at index {index} in {split_json}"
            )

        required_ids.add(int(item["reference_img_id"]))

        gt_ids = item["gt_img_ids"]

        if not isinstance(gt_ids, list):
            raise ValueError(
                f"'gt_img_ids' must be a list at index {index}"
            )

        required_ids.update(int(image_id) for image_id in gt_ids)

    return required_ids


# ---------------------------------------------------------------------------
# COCO metadata
# ---------------------------------------------------------------------------

def load_coco_metadata(
    coco_metadata_path: Path,
) -> Dict[int, str]:
    """
    Load COCO metadata and create:

        image_id -> filename

    mapping.
    """

    data = load_json(coco_metadata_path)

    if "images" not in data:
        raise KeyError(
            f"COCO metadata does not contain an 'images' field:\n"
            f"  {coco_metadata_path}"
        )

    images = data["images"]

    if not isinstance(images, list):
        raise ValueError(
            f"COCO metadata 'images' field is not a list:\n"
            f"  {coco_metadata_path}"
        )

    id_to_filename: Dict[int, str] = {}

    for image in images:
        if "id" not in image or "file_name" not in image:
            continue

        image_id = int(image["id"])
        filename = image["file_name"]

        id_to_filename[image_id] = filename

    if not id_to_filename:
        raise ValueError(
            f"No valid images found in COCO metadata:\n"
            f"  {coco_metadata_path}"
        )

    return id_to_filename


# ---------------------------------------------------------------------------
# Random image selection
# ---------------------------------------------------------------------------

def select_random_ids(
    all_coco_ids: Set[int],
    required_ids: Set[int],
    num_random: int,
    seed: int,
) -> Set[int]:
    """
    Select random COCO IDs while excluding required CIRCO IDs.
    """

    available_ids = list(all_coco_ids - required_ids)

    if num_random > len(available_ids):
        raise ValueError(
            f"Requested {num_random} random images, but only "
            f"{len(available_ids)} are available after excluding "
            "CIRCO required images."
        )

    rng = random.Random(seed)

    return set(rng.sample(available_ids, num_random))


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def download_one(
    image_id: int,
    filename: str,
    output_dir: Path,
    retries: int,
    timeout: int = DEFAULT_TIMEOUT,
) -> Tuple[int, str, bool, str]:
    """
    Download one COCO image.

    Returns:
        (image_id, filename, success, message)
    """

    output_path = output_dir / filename
    url = f"{COCO_IMAGE_BASE_URL}/{filename}"

    # Extra protection against duplicate work.
    if output_path.is_file():
        return image_id, filename, True, "exists"

    last_error = ""

    for attempt in range(retries + 1):
        try:
            request = Request(
                url,
                headers={
                    "User-Agent": "CIRCO-COCO-downloader/1.0"
                },
            )

            with urlopen(request, timeout=timeout) as response:
                if response.status != 200:
                    raise RuntimeError(
                        f"HTTP status {response.status}"
                    )

                content = response.read()

            if not content:
                raise RuntimeError("Downloaded file is empty")

            # Write only after the complete response has been received.
            # This prevents partially downloaded files from being treated
            # as valid files after an interrupted download.
            output_path.write_bytes(content)

            return image_id, filename, True, "downloaded"

        except HTTPError as exc:
            last_error = f"HTTP {exc.code}: {exc.reason}"

        except URLError as exc:
            last_error = f"URL error: {exc.reason}"

        except TimeoutError:
            last_error = "timeout"

        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"

        # No retry needed after the final attempt.
        if attempt < retries:
            # Exponential backoff:
            # 1s, 2s, 4s, ...
            delay = 2 ** attempt
            time.sleep(delay)

    return image_id, filename, False, last_error


# ---------------------------------------------------------------------------
# Dry-run
# ---------------------------------------------------------------------------

def print_dry_run(
    split: str,
    required_ids: Set[int],
    random_ids: Set[int],
    download_items: List[Tuple[int, str]],
    existing_count: int,
    output_dir: Path,
) -> None:
    """Print planned operations without downloading."""

    print()
    print("=" * 60)
    print("Dry run")
    print("=" * 60)
    print(f"Split        : {split}")
    print(f"Required     : {len(required_ids)}")
    print(f"Random       : {len(random_ids)}")
    print(f"Total        : {len(required_ids | random_ids)}")
    print(f"Existing     : {existing_count}")
    print(f"To download  : {len(download_items)}")
    print(f"Output       : {output_dir}")
    print()
    print("Dry run enabled. No files will be downloaded.")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(
    dataset_root: Path,
    split: str,
    required_ids: Set[int],
    random_ids: Set[int],
    existing_count: int,
    downloaded_count: int,
    failed_items: List[Tuple[int, str, str]],
    output_dir: Path,
) -> None:
    """Print final download summary."""

    total_count = len(required_ids | random_ids)

    print()
    print("=" * 60)
    print("Download summary")
    print("=" * 60)

    print(f"Dataset root : {dataset_root}")
    print(f"Split        : {split}")
    print()

    print(f"Required     : {len(required_ids)}")
    print(f"Random       : {len(random_ids)}")
    print(f"Total        : {total_count}")
    print()

    print(f"Existing     : {existing_count}")
    print(f"Downloaded   : {downloaded_count}")
    print(f"Failed       : {len(failed_items)}")
    print()

    print(f"Output       : {output_dir}")

    if failed_items:
        print()
        print("Failed downloads:")
        for image_id, filename, error in failed_items:
            print(f"  {image_id}  {filename}")
            print(f"      {error}")

    print("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()

    dataset_root = args.dataset_root.expanduser().resolve()

    split_json, coco_metadata_path, output_dir = resolve_paths(args)

    print("=" * 60)
    print("CIRCO COCO 2017 Unlabeled downloader")
    print("=" * 60)
    print(f"Dataset root : {dataset_root}")
    print(f"Split        : {args.split}")
    print(f"Random       : {args.num_random}")
    print(f"Seed         : {args.seed}")
    print(f"Workers      : {args.workers}")
    print(f"Retries      : {args.retries}")
    print(f"Output       : {output_dir}")
    print()

    try:
        validate_dataset_structure(
            dataset_root=dataset_root,
            split_json=split_json,
            coco_metadata=coco_metadata_path,
            output_dir=output_dir,
        )

        print("Loading CIRCO annotations...")
        required_ids = load_circo_required_ids(split_json)

        print(f"Found {len(required_ids)} required image IDs.")

        print("Loading COCO metadata...")
        id_to_filename = load_coco_metadata(coco_metadata_path)

        all_coco_ids = set(id_to_filename.keys())

        # Make sure every CIRCO-required image exists in COCO metadata.
        missing_ids = required_ids - all_coco_ids

        if missing_ids:
            sample = sorted(missing_ids)[:20]

            raise ValueError(
                f"{len(missing_ids)} CIRCO image IDs were not found "
                "in COCO metadata.\n\n"
                f"Example missing IDs: {sample}"
            )

        random_ids = select_random_ids(
            all_coco_ids=all_coco_ids,
            required_ids=required_ids,
            num_random=args.num_random,
            seed=args.seed,
        )

        download_ids = required_ids | random_ids

        # Map IDs to filenames.
        download_items: List[Tuple[int, str]] = []

        existing_count = 0

        for image_id in sorted(download_ids):
            filename = id_to_filename[image_id]
            output_path = output_dir / filename

            if output_path.is_file():
                existing_count += 1
            else:
                download_items.append((image_id, filename))

        print()
        print(f"Required images : {len(required_ids)}")
        print(f"Random images   : {len(random_ids)}")
        print(f"Total images    : {len(download_ids)}")
        print(f"Already exist   : {existing_count}")
        print(f"To download     : {len(download_items)}")

        if args.dry_run:
            print_dry_run(
                split=args.split,
                required_ids=required_ids,
                random_ids=random_ids,
                download_items=download_items,
                existing_count=existing_count,
                output_dir=output_dir,
            )
            return 0

        if not download_items:
            print()
            print("Nothing to download. All requested images already exist.")
            print_summary(
                dataset_root=dataset_root,
                split=args.split,
                required_ids=required_ids,
                random_ids=random_ids,
                existing_count=existing_count,
                downloaded_count=0,
                failed_items=[],
                output_dir=output_dir,
            )
            return 0

        print()
        print(f"Starting download with {args.workers} workers...")
        print()

        downloaded_count = 0
        failed_items: List[Tuple[int, str, str]] = []

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_to_item = {
                executor.submit(
                    download_one,
                    image_id,
                    filename,
                    output_dir,
                    args.retries,
                ): (image_id, filename)
                for image_id, filename in download_items
            }

            try:
                from tqdm import tqdm
            except ImportError:
                tqdm = None

            if tqdm is not None:
                progress = tqdm(
                    total=len(download_items),
                    desc="Downloading COCO images",
                    unit="img",
                )
            else:
                progress = None
                print(
                    "WARNING: tqdm is not installed. "
                    "Progress bar is disabled."
                )

            try:
                for future in as_completed(future_to_item):
                    image_id, filename = future_to_item[future]

                    try:
                        (
                            result_id,
                            result_filename,
                            success,
                            message,
                        ) = future.result()

                        if success:
                            if message == "downloaded":
                                downloaded_count += 1
                        else:
                            failed_items.append(
                                (
                                    result_id,
                                    result_filename,
                                    message,
                                )
                            )

                    except Exception as exc:
                        failed_items.append(
                            (
                                image_id,
                                filename,
                                f"{type(exc).__name__}: {exc}",
                            )
                        )

                    finally:
                        if progress is not None:
                            progress.update(1)

            finally:
                if progress is not None:
                    progress.close()

        print_summary(
            dataset_root=dataset_root,
            split=args.split,
            required_ids=required_ids,
            random_ids=random_ids,
            existing_count=existing_count,
            downloaded_count=downloaded_count,
            failed_items=failed_items,
            output_dir=output_dir,
        )

        # Non-zero exit code if anything failed.
        return 1 if failed_items else 0

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        print(
            "Run the same command again to resume. "
            "Existing files will be skipped."
        )
        return 130

    except Exception as exc:
        print(
            f"\nERROR: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())