#!/usr/bin/env python3

import argparse
import hashlib
import logging
import os
import signal
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from PIL import Image
from tqdm import tqdm


DATA_TYPES = ["dress", "shirt", "toptee"]
SPLITS = ["train", "val", "test"]

STOP = False


def signal_handler(sig, frame):
    global STOP
    STOP = True
    print("\nStopping after current downloads...")


signal.signal(signal.SIGINT, signal_handler)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(8192)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def load_placeholder_hashes(metadata_root):
    broken_dir = os.path.join(
        metadata_root,
        "image_url",
        "broken_links",
    )

    hashes = set()

    for f in os.listdir(broken_dir):
        if f.lower().endswith(".jpg"):
            hashes.add(
                sha256_file(os.path.join(broken_dir, f))
            )

    return hashes


def load_url_table(metadata_root):
    mapping = {}

    for cloth in DATA_TYPES:

        txt = os.path.join(
            metadata_root,
            "image_url",
            f"asin2url.{cloth}.txt",
        )

        with open(txt, "r", encoding="utf8") as f:

            for line in f:

                line = line.strip()

                if not line:
                    continue

                parts = line.split()

                if len(parts) < 2:
                    continue

                asin = parts[0]
                url = parts[-1]

                mapping[asin] = url

    return mapping


def collect_asins(dataset_root, selected):

    ids = set()

    split_root = os.path.join(dataset_root, "image_splits")
    caption_root = os.path.join(dataset_root, "captions")

    import json

    for cloth, split in selected:

        split_file = os.path.join(
            split_root,
            f"split.{cloth}.{split}.json",
        )

        with open(split_file, "r") as f:
            arr = json.load(f)

        ids.update(arr)

        cap_file = os.path.join(
            caption_root,
            f"cap.{cloth}.{split}.json",
        )

        with open(cap_file, "r") as f:
            caps = json.load(f)

        for x in caps:

            if "candidate" in x:
                ids.add(x["candidate"])

            if "target" in x:
                ids.add(x["target"])

    return ids


def verify_image(path, placeholder_hashes):

    try:

        Image.open(path).verify()

    except Exception:
        return False

    h = sha256_file(path)

    if h in placeholder_hashes:
        return False

    return True


def download_one(
    asin,
    url,
    out_dir,
    placeholder_hashes,
    retries=3,
):

    outfile = os.path.join(out_dir, asin + ".jpg")

    if os.path.exists(outfile):
        return "exists"

    tmp = outfile + ".tmp"

    for _ in range(retries):

        try:

            r = requests.get(
                url,
                timeout=20,
                stream=True,
            )

            if r.status_code != 200:
                raise RuntimeError("status")

            with open(tmp, "wb") as f:

                for chunk in r.iter_content(8192):
                    if chunk:
                        f.write(chunk)

            if not verify_image(tmp, placeholder_hashes):

                logging.warning(
                    "[%s] Placeholder image detected\nURL: %s\n",
                    asin,
                    url,
                )

                os.remove(tmp)

                return "broken"

            os.rename(tmp, outfile)

            return "downloaded"

        except Exception as e:

            if os.path.exists(tmp):
                os.remove(tmp)

            logging.error(
                "[%s] %s\nURL: %s\nReason: %s\n",
                asin,
                time.strftime("%Y-%m-%d %H:%M:%S"),
                url,
                repr(e),
            )

            time.sleep(1)

    return "failed"


def parse_selected(arg):

    if arg is None:

        res = []

        for t in DATA_TYPES:
            for s in SPLITS:
                res.append((t, s))

        return res

    out = []

    for x in arg:

        cloth, split = x.split(":")

        if cloth not in DATA_TYPES:
            raise ValueError(cloth)

        if split not in SPLITS:
            raise ValueError(split)

        out.append((cloth, split))

    return out


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset-root",
        required=True,
    )

    parser.add_argument(
        "--metadata-root",
        required=True,
    )

    parser.add_argument(
        "--splits",
        nargs="+",
        default=None,
        help="dress:val shirt:test ...",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=16,
    )

    args = parser.parse_args()

    os.makedirs(
        os.path.join(args.dataset_root, "images"),
        exist_ok=True,
    )

    logging.basicConfig(
        filename="download.log",
        level=logging.INFO,
    )

    selected = parse_selected(args.splits)

    print("\nSelected:")

    for x in selected:
        print(x)

    print()

    ids = collect_asins(
        args.dataset_root,
        selected,
    )

    print(f"Need {len(ids)} images")

    urls = load_url_table(args.metadata_root)

    placeholders = load_placeholder_hashes(
        args.metadata_root
    )

    out_dir = os.path.join(
        args.dataset_root,
        "images",
    )

    failed = []

    stats = {
        "downloaded": 0,
        "exists": 0,
        "broken": 0,
        "failed": 0,
    }

    with ThreadPoolExecutor(
        max_workers=args.workers
    ) as pool:

        futures = {}

        for asin in ids:

            if STOP:
                break

            if asin not in urls:
                logging.error(
                    "[%s] URL not found in metadata",
                    asin,
                )

                stats["failed"] += 1
                failed.append(asin)
                continue

            futures[
                pool.submit(
                    download_one,
                    asin,
                    urls[asin],
                    out_dir,
                    placeholders,
                )
            ] = asin

        for f in tqdm(
            as_completed(futures),
            total=len(futures),
        ):

            r = f.result()

            stats[r] += 1

            if r == "failed":
                failed.append(futures[f])

    with open(
        "failed_urls.txt",
        "w",
    ) as f:

        for x in failed:
            f.write(x + "\n")

    print("\n========== SUMMARY ==========")

    for k, v in stats.items():
        print(f"{k:12s}: {v}")

    print("=============================")


if __name__ == "__main__":
    main()