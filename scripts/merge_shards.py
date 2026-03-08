"""
Merge shard output files into a single deduplicated v4.2 upgraded JSONL.

Usage:
    python3 scripts/merge_shards.py
    python3 scripts/merge_shards.py --include-v41  # also merge v4.1 data
"""

import json
import argparse
from pathlib import Path
from collections import Counter

OUTPUT_DIR = Path("docs/datasets/upgraded")


def merge():
    parser = argparse.ArgumentParser()
    parser.add_argument("--include-v41", action="store_true",
                        help="Also include v4.1 upgraded examples")
    args = parser.parse_args()

    seen = set()
    merged = []
    dupes = 0
    sources = Counter()

    # Collect all shard files
    shard_files = sorted(OUTPUT_DIR.glob("v42_upgraded_shard*.jsonl"))
    main_file = OUTPUT_DIR / "v42_upgraded.jsonl"

    files_to_merge = []
    if main_file.exists():
        files_to_merge.append(("main", main_file))
    for sf in shard_files:
        files_to_merge.append((sf.stem, sf))

    if args.include_v41:
        v41_file = OUTPUT_DIR / "v41_upgraded.jsonl"
        if v41_file.exists():
            files_to_merge.append(("v41", v41_file))

    for label, fpath in files_to_merge:
        count = 0
        with open(fpath) as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    meta = entry.get("metadata", {})
                    key = (meta.get("source", ""), str(meta.get("source_id", "")))
                    if key in seen:
                        dupes += 1
                        continue
                    seen.add(key)
                    merged.append(entry)
                    sources[meta.get("source", "unknown")] += 1
                    count += 1
                except (json.JSONDecodeError, KeyError):
                    continue
        print(f"  {label:30s}: {count} unique examples")

    # Write merged output
    output = OUTPUT_DIR / "v42_merged.jsonl"
    with open(output, "w") as f:
        for entry in merged:
            f.write(json.dumps(entry) + "\n")

    print(f"\n{'=' * 50}")
    print(f"Merged: {len(merged)} unique examples")
    print(f"Duplicates removed: {dupes}")
    print(f"\nBy dataset:")
    for ds, count in sources.most_common():
        print(f"  {ds:25s}: {count}")
    print(f"\nOutput: {output}")

    # Also merge logs
    log_files = sorted(OUTPUT_DIR.glob("upgrade_log_shard*.jsonl"))
    main_log = OUTPUT_DIR / "upgrade_log.jsonl"
    merged_log = OUTPUT_DIR / "upgrade_log_merged.jsonl"

    log_seen = set()
    log_count = 0
    with open(merged_log, "w") as out:
        all_logs = [main_log] + list(log_files)
        for lf in all_logs:
            if not lf.exists():
                continue
            with open(lf) as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                        key = entry.get("id", "")
                        ts = entry.get("timestamp", "")
                        dedup_key = f"{key}:{ts}"
                        if dedup_key in log_seen:
                            continue
                        log_seen.add(dedup_key)
                        out.write(line)
                        log_count += 1
                    except json.JSONDecodeError:
                        continue
    print(f"\nMerged log: {log_count} entries → {merged_log}")


if __name__ == "__main__":
    merge()
