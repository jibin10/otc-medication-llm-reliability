#!/usr/bin/env python3
"""Download and verify bulk openFDA NDC + drug-label snapshots for the HRP.

Uses only the Python standard library. RxNorm full release is downloaded
manually after the researcher accepts the NLM/UMLS terms.

Usage:
  python download_hrp_datasets.py --out "D:/HRP/data" --plan
  python download_hrp_datasets.py --out "D:/HRP/data" --download
  python download_hrp_datasets.py --out "D:/HRP/data" --verify

The first use saves the live FDA download manifest. Later runs use that
same manifest, unless --refresh-manifest is explicitly requested.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen
import zipfile

FDA_MANIFEST = "https://api.fda.gov/download.json"
FDA_DOWNLOAD_HOST = "download.open.fda.gov"
UA = "HRP-academic-data-acquisition/1.0 (openFDA public bulk data)"
CHUNK = 8 * 1024 * 1024


def request(url, start=0):
    headers = {"User-Agent": UA}
    if start:
        headers["Range"] = f"bytes={start}-"
    return Request(url, headers=headers)


def load_manifest(path, refresh=False):
    if path.exists() and not refresh:
        return json.loads(path.read_text(encoding="utf-8"))
    print(f"Fetching official manifest: {FDA_MANIFEST}")
    with urlopen(request(FDA_MANIFEST), timeout=60) as response:
        raw = response.read()
    value = json.loads(raw)
    # Check the required endpoints BEFORE replacing an existing manifest.
    extract_endpoints(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".json.part")
    temp.write_bytes(raw)
    os.replace(temp, path)
    print(f"Saved frozen snapshot: {path}")
    return value


def extract_endpoints(manifest):
    drug = manifest["results"]["drug"]
    found = {}
    for endpoint in ("ndc", "label"):
        entry = drug[endpoint]
        parts = entry.get("partitions", [])
        if not parts:
            raise ValueError(f"No partitions found for drug/{endpoint}")
        urls = set()
        for item in parts:
            parsed = urlparse(item["file"])
            if parsed.scheme != "https" or parsed.hostname != FDA_DOWNLOAD_HOST:
                raise ValueError(f"Unexpected download URL: {item['file']}")
            if item["file"] in urls:
                raise ValueError(f"Repeated URL in manifest: {item['file']}")
            urls.add(item["file"])
        found[endpoint] = entry
    return found


def zip_ok(path):
    try:
        if not path.exists() or path.stat().st_size == 0:
            return False
        with zipfile.ZipFile(path) as archive:
            if not archive.namelist():
                return False
            # Reads the contents to verify the ZIP CRCs.
            return archive.testzip() is None
    except (OSError, zipfile.BadZipFile, RuntimeError):
        return False


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_download(url, dest, tries=5):
    temp = dest.with_suffix(dest.suffix + ".part")
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and zip_ok(dest):
        print(f"  Existing verified: {dest.name}")
        return
    if dest.exists():
        print(f"  Invalid ZIP; redownloading: {dest.name}")
        dest.unlink()
    for attempt in range(1, tries + 1):
        offset = temp.stat().st_size if temp.exists() else 0
        try:
            with urlopen(request(url, offset), timeout=120) as response:
                status = getattr(response, "status", response.getcode())
                content_range = response.headers.get("Content-Range", "")
                if offset and status == 206:
                    match = re.match(r"bytes\s+(\d+)-\d+/", content_range)
                    if match is None or int(match.group(1)) != offset:
                        raise OSError("Server returned an unexpected byte range")
                    mode = "ab"
                    print(f"  Resuming {dest.name} from {offset:,} bytes")
                elif status == 200:
                    mode = "wb"  # Range not supported: restart safely.
                    print(f"  Downloading {dest.name}")
                else:
                    raise OSError(f"Unexpected HTTP status: {status}")
                with open(temp, mode) as output:
                    while True:
                        buf = response.read(CHUNK)
                        if not buf:
                            break
                        output.write(buf)
            if not zip_ok(temp):
                raise OSError("Downloaded file failed ZIP CRC verification")
            os.replace(temp, dest)
            print(f"  Verified ZIP: {dest.name} ({dest.stat().st_size / 1024**2:.1f} MiB)")
            return
        except HTTPError as exc:
            if exc.code == 416 and temp.exists():
                if zip_ok(temp):
                    os.replace(temp, dest)
                    return
                temp.unlink()
            print(f"  Attempt {attempt}/{tries}: HTTP {exc.code}")
        except (URLError, OSError, zipfile.BadZipFile) as exc:
            print(f"  Attempt {attempt}/{tries}: {exc}")
            # If a completed download is corrupt, retry it from zero.
            if isinstance(exc, OSError) and "failed ZIP CRC" in str(exc) and temp.exists():
                temp.unlink()
        if attempt < tries:
            time.sleep(min(2 ** attempt, 15))
    raise RuntimeError(f"Failed to download a valid copy of {dest.name}")


def rxnorm_status(folder, validate=False):
    """Verify licensed RxNorm ZIP or the core files of an extracted full release.

    NLM distributions can contain two complete RRF directories: the full
    terminology and a smaller prescribable subset. Select the full RRF set,
    not an arbitrary match returned by a recursive filename search.
    """
    records = []
    if not folder.exists():
        return records
    required = ("RXNCONSO.RRF", "RXNREL.RRF", "RXNSAT.RRF")
    for path in sorted(folder.iterdir()):
        name = path.name.lower()
        if not name.startswith("rxnorm_full_"):
            continue
        is_full = not name.startswith("rxnorm_full_prescribe_")
        kind = "full" if is_full else "prescribable_subset"
        if path.is_file() and path.suffix.lower() == ".zip":
            info = {"file": str(path), "format": "zip", "release_type": kind,
                    "bytes": path.stat().st_size}
            if validate:
                info["zip_crc_ok"] = zip_ok(path)
                if info["zip_crc_ok"]:
                    info["sha256"] = sha256(path)
            records.append(info)
        elif path.is_dir():
            # Collect complete sets of files by their common parent folder.
            grouped = {}
            for filename in required:
                for match in path.rglob(filename):
                    if match.is_file() and match.stat().st_size > 0:
                        grouped.setdefault(match.parent, {})[filename] = match
            complete = [(parent, files) for parent, files in grouped.items()
                        if all(filename in files for filename in required)]

            def is_prescribable_dir(parent):
                parts = parent.relative_to(path).parts
                return any("prescrib" in component.lower() for component in parts)

            candidate_sets = []
            for parent, files in sorted(complete, key=lambda item: str(item[0])):
                candidate_sets.append({
                    "directory": str(parent),
                    "is_prescribable_subset": is_prescribable_dir(parent),
                    "total_core_bytes": sum(files[n].stat().st_size for n in required),
                })

            full_candidates = [(parent, files) for parent, files in complete
                               if not is_prescribable_dir(parent)]
            # If the archive contains a canonical /rrf folder, prefer it.
            canonical = [(parent, files) for parent, files in full_candidates
                         if parent == path / "rrf" or parent == path / path.name / "rrf"]
            if len(canonical) == 1:
                chosen_parent, chosen_files = canonical[0]
                selection = "canonical full-release rrf directory"
            elif len(full_candidates) == 1:
                chosen_parent, chosen_files = full_candidates[0]
                selection = "unique non-prescribable complete RRF directory"
            else:
                chosen_parent, chosen_files, selection = None, {}, None

            selected = {}
            for filename, match in chosen_files.items():
                selected[filename] = {"path": str(match), "bytes": match.stat().st_size}
                if validate:
                    selected[filename]["sha256"] = sha256(match)
            info = {
                "file": str(path), "format": "extracted_directory",
                "release_type": kind, "required_core_files": list(required),
                "candidate_rrf_sets": candidate_sets,
                "selected_rrf_directory": str(chosen_parent) if chosen_parent else None,
                "selection_reason": selection,
                "core_files": selected,
                "core_files_present": len(selected) == len(required),
                "validation_note": "Extracted core files verified by presence, nonzero size and SHA-256; "
                                   "original ZIP CRC/provenance not verified."
            }
            records.append(info)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=Path("hrp_data"), help="Root output directory")
    flags = parser.add_mutually_exclusive_group()
    flags.add_argument("--plan", action="store_true", help="Show required partitions without downloading data (default)")
    flags.add_argument("--download", action="store_true", help="Download all openFDA NDC and drug-label parts")
    flags.add_argument("--verify", action="store_true", help="Verify FDA archives and either a RxNorm ZIP or extracted release")
    parser.add_argument("--refresh-manifest", action="store_true", help="Replace the frozen FDA manifest with the currently offered snapshot")
    args = parser.parse_args()
    root = args.out.expanduser().resolve()
    manifest_path = root / "raw" / "openfda" / "download_manifest.json"
    if args.verify and not manifest_path.exists():
        parser.error(f"Missing frozen manifest: {manifest_path}. Run --plan or --download first.")
    manifest = load_manifest(manifest_path, refresh=args.refresh_manifest)
    endpoints = extract_endpoints(manifest)
    print(f"FDA manifest last updated: {manifest.get('meta', {}).get('last_updated', 'unavailable')}")
    total_parts = 0
    estimated_mb = 0.0
    for endpoint, entry in endpoints.items():
        parts = entry["partitions"]
        total_parts += len(parts)
        size_mb = sum(float(p.get("size_mb", 0) or 0) for p in parts)
        estimated_mb += size_mb
        print(f"  drug/{endpoint}: {len(parts)} archive(s), ~{size_mb:,.1f} MB compressed, export {entry.get('export_date')}, {entry.get('total_records')} records")
    print(f"Total: {total_parts} FDA archives, ~{estimated_mb / 1024:.2f} GB compressed (manifest estimate)")
    if not (args.download or args.verify):
        print("Plan only. Use --download to fetch all FDA archives, then manually add the licensed RxNorm ZIP.")
        return 0

    if args.download:
        free = shutil.disk_usage(root).free / (1024**3)
        print(f"Free disk space: {free:.1f} GiB")
        if free < max(5.0, estimated_mb / 1024 * 1.5):
            print("WARNING: Low free space. Consider a larger disk before downloading.")
    files = []
    failed = []
    for endpoint, entry in endpoints.items():
        for part in entry["partitions"]:
            url = part["file"]
            dest = root / "raw" / "openfda" / endpoint / Path(urlparse(url).path).name
            try:
                if args.download:
                    save_download(url, dest)
                if not zip_ok(dest):
                    raise OSError("Missing archive or failed ZIP CRC check")
                files.append({"endpoint": f"drug/{endpoint}", "file": str(dest), "source": url,
                              "bytes": dest.stat().st_size, "sha256": sha256(dest), "zip_crc_ok": True})
                if args.verify:
                    print(f"  OK: {dest.name}")
            except (OSError, RuntimeError) as exc:
                failed.append({"file": str(dest), "error": str(exc)})
                print(f"ERROR: {dest.name}: {exc}", file=sys.stderr)
    rxnorm_dir = root / "raw" / "rxnorm"
    rxnorm_dir.mkdir(parents=True, exist_ok=True)
    rxnorm = rxnorm_status(rxnorm_dir, validate=True)
    full_rxnorm = any(
        x["release_type"] == "full" and
        (x.get("zip_crc_ok") is True or x.get("core_files_present") is True)
        for x in rxnorm
    )
    rxnorm_original_verified = any(x["release_type"] == "full" and x.get("zip_crc_ok") is True for x in rxnorm)
    if not rxnorm:
        print(f"RxNorm is MISSING. Put its ZIP or extracted directory in: {rxnorm_dir}")
    elif not full_rxnorm:
        print("RxNorm full release not verified; required extracted RRF files may be missing/empty.")
    elif rxnorm_original_verified:
        print("RxNorm full release ZIP verified (CRC + SHA-256).")
    else:
        print("RxNorm extracted release core files verified structurally + SHA-256 (original ZIP unavailable).")
    report = {"created_utc": datetime.now(timezone.utc).isoformat(),
              "fda_manifest": str(manifest_path), "fda_manifest_sha256": sha256(manifest_path),
              "fda_last_updated": manifest.get("meta", {}).get("last_updated"),
              "fda_files_expected": total_parts, "fda_files_verified": len(files),
              "fda_files": files, "fda_errors": failed, "rxnorm_files": rxnorm,
              "complete": len(files) == total_parts and len(failed) == 0 and full_rxnorm}
    out = root / "reports" / "download_verification.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved audit: {out}")
    if report["complete"]:
        if rxnorm_original_verified:
            print("ALL THREE DATASETS VERIFIED (FDA ZIPs + original RxNorm ZIP)")
        else:
            print("FDA ZIPs VERIFIED; RxNorm EXTRACTED CORE FILES VERIFIED (original RxNorm ZIP not checked)")
    else:
        print("NOT YET COMPLETE: review missing/failed files")
    return 0 if report["complete"] else 2


if __name__ == "__main__":
    sys.exit(main())
