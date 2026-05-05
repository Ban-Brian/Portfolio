#!/usr/bin/env python3
import argparse
import hashlib
import json
import pathlib
import sys
import time
import zipfile
import tarfile

CITIES = ["Philadelphia", "Tampa", "Indianapolis", "Nashville", "Tucson"]

CUISINES = [
    "American (Traditional)", "American (New)", "Italian", "Mexican",
    "Chinese", "Japanese", "Thai", "Indian", "Mediterranean", "Pizza",
]

MIN_REVIEWS = 20

YELP_FILES = {
    "business":  "yelp_academic_dataset_business.json",
    "review":    "yelp_academic_dataset_review.json",
    "tip":       "yelp_academic_dataset_tip.json",
    "checkin":   "yelp_academic_dataset_checkin.json",
    "user":      "yelp_academic_dataset_user.json",
}

def is_restaurant(categories):
    return isinstance(categories, str) and "Restaurants" in categories

def primary_cuisine(categories):
    if not isinstance(categories, str):
        return None
    tags = [t.strip() for t in categories.split(",")]
    for c in CUISINES:
        if c in tags:
            return c
    return None

def keep_business(rec):
    if rec.get("city") not in CITIES:
        return False
    if not is_restaurant(rec.get("categories")):
        return False
    if primary_cuisine(rec.get("categories")) is None:
        return False
    if (rec.get("review_count") or 0) < MIN_REVIEWS:
        return False
    return True

def filter_businesses(tar, src_name, out_lines):
    kept_ids = set()
    n_in = n_out = 0
    f = tar.extractfile(src_name)
    for line in f:
        line = line.decode('utf-8')
        n_in += 1
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if keep_business(rec):
            kept_ids.add(rec["business_id"])
            out_lines.append(line if line.endswith("\n") else line + "\n")
            n_out += 1
    return kept_ids, n_in, n_out

def filter_by_business_id(tar, src_name, kept_ids, out_lines):
    n_in = n_out = 0
    f = tar.extractfile(src_name)
    for line in f:
        line = line.decode('utf-8')
        n_in += 1
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("business_id") in kept_ids:
            out_lines.append(line if line.endswith("\n") else line + "\n")
            n_out += 1
    return n_in, n_out

class PartitionedZipWriter:
    CHUNK_SIZE = 4 * 1024 * 1024

    def __init__(self, out_dir, basename, part_size_bytes):
        self.out_dir = pathlib.Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.basename = basename
        self.part_size = part_size_bytes
        self.part_index = 0
        self.current_zip = None
        self.current_path = None
        self.parts = []
        self.entries = []
        self._open_next_part()

    def _open_next_part(self):
        if self.current_zip is not None:
            self.current_zip.close()
            self._finalize_part()
        self.part_index += 1
        self.current_path = (
            self.out_dir / f"{self.basename}.part{self.part_index:02d}.zip"
        )
        self.current_zip = zipfile.ZipFile(
            self.current_path, "w",
            compression=zipfile.ZIP_DEFLATED, compresslevel=6,
        )

    def _finalize_part(self):
        path = self.current_path
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        self.parts.append({
            "path": path.name,
            "sha256": h.hexdigest(),
            "bytes": path.stat().st_size,
        })

    def _current_part_bytes(self):
        self.current_zip.fp.flush()
        return self.current_path.stat().st_size

    def add_bytes(self, member_name, data):
        chunks = []
        offset = 0
        chunk_idx = 0
        n = len(data)

        while offset < n:
            if self._current_part_bytes() >= self.part_size:
                self._open_next_part()

            take = min(self.CHUNK_SIZE, n - offset)
            piece = data[offset:offset + take]

            if offset == 0 and take == n:
                arcname = member_name
            else:
                chunk_idx += 1
                arcname = f"{member_name}.chunk{chunk_idx:04d}"

            self.current_zip.writestr(arcname, piece)
            chunks.append({
                "part": self.current_path.name,
                "member": arcname,
                "bytes": take,
            })
            offset += take

        self.entries.append({
            "name": member_name,
            "total_bytes": n,
            "chunks": chunks,
        })

    def close(self):
        if self.current_zip is not None:
            self.current_zip.close()
            self.current_zip = None
            self._finalize_part()

def human_bytes(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yelp-tar", required=True)
    ap.add_argument("--out-dir", default="./yelp_subset_zips")
    ap.add_argument("--part-size-mb", type=int, default=95)
    ap.add_argument("--include", nargs="+", default=["business", "review", "tip", "checkin"])
    args = ap.parse_args()

    yelp_tar_path = pathlib.Path(args.yelp_tar)
    tar = tarfile.open(yelp_tar_path, "r")

    filtered = {}
    counts = {}

    bus_lines = []
    kept_ids, b_in, b_out = filter_businesses(tar, YELP_FILES["business"], bus_lines)
    filtered["business"] = bus_lines
    
    if "review" in args.include:
        rev_lines = []
        r_in, r_out = filter_by_business_id(tar, YELP_FILES["review"], kept_ids, rev_lines)
        filtered["review"] = rev_lines

    if "tip" in args.include:
        tip_lines = []
        t_in, t_out = filter_by_business_id(tar, YELP_FILES["tip"], kept_ids, tip_lines)
        filtered["tip"] = tip_lines

    if "checkin" in args.include:
        chk_lines = []
        c_in, c_out = filter_by_business_id(tar, YELP_FILES["checkin"], kept_ids, chk_lines)
        filtered["checkin"] = chk_lines

    writer = PartitionedZipWriter(args.out_dir, "yelp_subset", args.part_size_mb * 1024 * 1024)

    for key, lines in filtered.items():
        member = YELP_FILES[key]
        payload = "".join(lines).encode("utf-8")
        writer.add_bytes(member, payload)

    writer.close()

if __name__ == "__main__":
    main()
