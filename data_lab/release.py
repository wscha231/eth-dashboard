"""Authoritative DATA research releases compatible with PR53's immutable consumer.

This builder certifies byte provenance and formulas, NOT commercial rights, original
historical receipt availability, predictive quality or production readiness.
"""
from __future__ import annotations
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import os
import re
import shutil
import sys
import numpy as np
import tempfile
from typing import Any

import pandas as pd

from .core import canonical, check_hash, read_bars, relative, safe_file, sha256, utc, validate_handoff
from .features import (INCUMBENT_BUILDER_SHA256, EXTENSION_VERSION, coverage, extension_features,
                       incumbent_features, incumbent_registry)

REQUIRED = {"release_id", "schema_version", "dataset_manifest_id", "dataset_manifest_hash", "feature_set_id",
    "feature_set_version", "code_sha", "data_cutoff_utc", "generated_at_utc", "source_versions",
    "coverage_by_feature", "quality_status", "vintage_mode", "license_status", "partition_locations",
    "checksums", "exclusions", "breaking_changes"}


def write_json(path: Path, data: Any) -> None:
    with Path(path).open("xb") as f:
        f.write(canonical(data))


def write_frame(path: Path, frame: pd.DataFrame) -> None:
    # Stable gzip header and float round-trip; no mandatory parquet dependency.
    raw = frame.to_csv(index_label="decision_at", float_format="%.17g", lineterminator="\n").encode()
    with Path(path).open("xb") as f:
        f.write(gzip.compress(raw, compresslevel=6, mtime=0))


def validate_release(root: Path, expected_sha256: str) -> dict:
    check_hash(expected_sha256)
    p = safe_file(root,"release.json")
    if sha256(p) != expected_sha256:
        raise ValueError("release hash mismatch")
    d = json.loads(p.read_bytes())
    if not REQUIRED <= d.keys() or d["schema_version"] != 1:
        raise ValueError("unsupported or incomplete DATA release")
    if d["quality_status"] not in {"pass", "partial"} or d["vintage_mode"] not in {"reconstructed", "observed_vintages"}:
        raise ValueError("unsupported release status")
    if not isinstance(d["license_status"],str) or not d["license_status"]:
        raise ValueError("explicit license status required")
    if not re.fullmatch(r"[0-9a-f]{40}",d["code_sha"]):
        raise ValueError("exact code SHA required")
    if utc(d["data_cutoff_utc"]) > utc(d["generated_at_utc"]):
        raise ValueError("invalid release time order")
    if not d["partition_locations"] or set(d["partition_locations"]) != set(d["checksums"]):
        raise ValueError("partition/checksum keys differ")
    for key,path in d["partition_locations"].items():
        relative(key);check_hash(d["checksums"][key])
        if sha256(safe_file(root,path)) != d["checksums"][key]:
            raise ValueError("partition hash mismatch: "+key)
    manifest = safe_file(root,"dataset_manifest.json")
    if sha256(manifest) != d["dataset_manifest_hash"]:
        raise ValueError("dataset manifest hash mismatch")
    m = json.loads(manifest.read_bytes())
    if m["dataset_manifest_id"] != d["dataset_manifest_id"]:
        raise ValueError("dataset identity mismatch")
    expected = set(d["checksums"]) - {"dataset_manifest.json"}
    if set(m["files"]) != expected:
        raise ValueError("manifest member set mismatch")
    for key,spec in m["files"].items():
        f = safe_file(root,d["partition_locations"][key])
        if spec["sha256"] != d["checksums"][key] or spec["bytes"] != f.stat().st_size:
            raise ValueError("manifest partition metadata mismatch")
    if sha256(p) != expected_sha256:
        raise ValueError("release changed during validation")
    return d


def load_feature_snapshot(root: Path, expected_sha256: str, *, family: str = "extensions",
                          mode: str = "retrospective") -> pd.DataFrame:
    d = validate_release(root,expected_sha256)
    if mode != "retrospective":
        raise ValueError("precomputed reconstructed snapshot is not a live PIT consumer")
    if family == "incumbent":
        bars, _ = read_bars(safe_file(root,"observations.db"),d["checksums"]["observations.db"],
                           cutoff=d["data_cutoff_utc"],generated=d["generated_at_utc"])
        return incumbent_features(bars)
    names = {"extensions":"extension_features.csv.gz"}
    if family not in names:
        raise ValueError("unknown feature family")
    p = safe_file(root,names[family])
    frame = pd.read_csv(p,float_precision="round_trip")
    frame.index = pd.to_datetime(frame.pop("decision_at"),utc=True,format="mixed")
    for c in frame:
        if c == "available_at" or c.endswith("__available_at"):
            frame[c] = pd.to_datetime(frame[c],utc=True,format="mixed")
    if frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
        raise ValueError("invalid feature snapshot index")
    if sha256(p) != d["checksums"][names[family]]:
        raise ValueError("feature snapshot changed during consumption")
    return frame


def build_release(source_root: Path, source_release_sha256: str, handoff_zip: Path,
                  handoff_sha256: str, destination: Path, *, generated_at: Any,
                  builder_commit: str) -> dict:
    """No network. Refuse nonempty destination; original inputs and receipts unchanged."""
    if not re.fullmatch(r"[0-9a-f]{40}",builder_commit):
        raise ValueError("builder_commit must be a full Git SHA")
    generated = utc(generated_at)
    check_hash(source_release_sha256)
    original_path = safe_file(source_root,"release.json")
    if sha256(original_path) != source_release_sha256:
        raise ValueError("source release checksum mismatch")
    src = json.loads(original_path.read_bytes())
    if not REQUIRED <= src.keys() or src["schema_version"] != 1:
        raise ValueError("incomplete source release")
    if utc(src["generated_at_utc"]) > generated:
        raise ValueError("cannot generate a release before its source receipt")
    if src["source_versions"].get("feature_builder_sha256") != INCUMBENT_BUILDER_SHA256:
        raise ValueError("unsupported input feature builder")
    if set(src["checksums"]) != set(src["partition_locations"]):
        raise ValueError("source partition contract mismatch")
    inputs = {}
    for key, rel in src["partition_locations"].items():
        relative(key);check_hash(src["checksums"][key])
        f = safe_file(source_root,rel)
        if sha256(f) != src["checksums"][key]:
            raise ValueError("source partition checksum mismatch")
        inputs[key] = f
    for key in ("observations.db","research.zip"):
        if key not in inputs:
            raise ValueError("missing required original input partition")
    handoff_manifest, handoff_members = validate_handoff(handoff_zip,handoff_sha256)
    bars, audit = read_bars(inputs["observations.db"],src["checksums"]["observations.db"],
                            cutoff=src["data_cutoff_utc"],generated=src["generated_at_utc"])
    incumbent = incumbent_features(bars)
    extensions, extension_specs = extension_features(bars)
    if not incumbent.index.equals(extensions.index):
        raise ValueError("feature families have different time grids")
    inc_cov, ext_cov = coverage(incumbent), coverage(extensions)
    inc_specs = incumbent_registry(list(inc_cov),INCUMBENT_BUILDER_SHA256)
    for spec in inc_specs + extension_specs:
        c = (inc_cov if spec["feature_id"] in inc_cov else ext_cov)[spec["feature_id"]]
        spec["coverage"] = c
        spec["first_valid_time"] = c["first_valid_time"]
        spec["last_valid_time"] = c["last_valid_time"]
    legacy_candidates = json.loads(handoff_members["research_20260915/feature_registry.json"])
    exact_incumbent_time = extensions["x_rel_720__available_at"].rename("available_at_exact").to_frame()
    paired = incumbent.available_at.notna() & exact_incumbent_time.available_at_exact.notna()
    ns_delta = (incumbent.loc[paired,"available_at"].astype("int64") - exact_incumbent_time.loc[paired,"available_at_exact"].astype("int64"))
    registry = {"schema_version":1,"incumbent_count":len(inc_specs),"extension_count":len(extension_specs),
        "features":inc_specs+extension_specs,"candidate_catalog_count":legacy_candidates["count"],
        "candidate_catalog_partition":"research_feature_candidates.json",
        "warning":"candidate catalog entries are not additional implemented columns or performance approvals"}
    audit.update({"authoritative_for":"this DATA research delivery only", "production_approved":False,
        "original_drive_sqlite_full_restore_verified":False,
        "known_source_lineage":src["source_versions"], "source_release_sha256":source_release_sha256,
        "source_research_package_missing_resolved":True,
        "handoff_validated_members":len(handoff_members),
        "incumbent_builder_sha256":INCUMBENT_BUILDER_SHA256,
        "extensions_version":EXTENSION_VERSION, "coverage_by_feature":{**inc_cov,**ext_cov},
        "incumbent_timestamp_precision_audit":{"comparable_rows":int(paired.sum()),
            "rounded_earlier_rows":int((ns_delta<0).sum()),"rounded_later_rows":int((ns_delta>0).sum()),
            "min_delta_ns":int(ns_delta.min()) if len(ns_delta) else None,
            "max_delta_ns":int(ns_delta.max()) if len(ns_delta) else None,
            "remedy":"incumbent_availability_exact.csv.gz; no change to historical/operating formulas"},
        "unresolved":["commercial/public/live usage rights", "historical publication and contemporaneous receipts",
                      "measured live latency limits", "P1 derivatives/macro native feeds and vintages",
                      "MODEL acceptance and same-origin predictive ablation"]})
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError("immutable release destination already exists")
    destination.parent.mkdir(parents=True,exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".data-release-",dir=destination.parent))
    try:
        for key in ("observations.db","research.zip"):
            shutil.copyfile(inputs[key],stage/key)
        shutil.copyfile(original_path,stage/"input_receipt.json")
        shutil.copyfile(handoff_zip,stage/"research_handoff.zip")
        write_json(stage/"feature_registry.json",registry)
        write_json(stage/"research_feature_candidates.json",legacy_candidates)
        write_json(stage/"quality_report.json",audit)
        write_frame(stage/"incumbent_availability_exact.csv.gz",exact_incumbent_time)
        write_frame(stage/"extension_features.csv.gz",extensions)
        for key in ("observations.db","research.zip"):
            if sha256(inputs[key]) != src["checksums"][key] or sha256(stage/key) != src["checksums"][key]:
                raise ValueError("source changed while copying")
        if (sha256(original_path) != source_release_sha256 or sha256(stage/"input_receipt.json") != source_release_sha256
                or sha256(handoff_zip) != handoff_sha256 or sha256(stage/"research_handoff.zip") != handoff_sha256):
            raise ValueError("input receipts changed while packaging")
        files = {p.name:{"bytes":p.stat().st_size,"sha256":sha256(p)} for p in sorted(stage.iterdir())}
        recipe = {"files":files,"builder_commit":builder_commit,"generated_at":generated.isoformat(),
                  "extension_version":EXTENSION_VERSION}
        identity = hashlib.sha256(canonical(recipe)).hexdigest()
        manifest = {"dataset_manifest_id":"eth-data-research-"+identity[:24],**recipe}
        write_json(stage/"dataset_manifest.json",manifest)
        allfiles = {p.name:sha256(p) for p in sorted(stage.iterdir())}
        module_hashes = {p.name:sha256(p) for p in sorted(Path(__file__).parent.glob("*.py"))}
        release = {"release_id":"eth-data-"+identity[:24], "schema_version":1,
            "dataset_manifest_id":manifest["dataset_manifest_id"],"dataset_manifest_hash":allfiles["dataset_manifest.json"],
            "feature_set_id":"signal_pipeline.build_features","feature_set_version":"sha256:"+INCUMBENT_BUILDER_SHA256,
            "code_sha":builder_commit,"data_cutoff_utc":src["data_cutoff_utc"],"generated_at_utc":generated.isoformat(),
            "source_versions":{**src["source_versions"],"input_release_sha256":source_release_sha256,
                "upstream_formula_code_sha":src["code_sha"],"research_handoff_sha256":handoff_sha256,
                "data_builder_module_sha256":module_hashes,"extension_version":EXTENSION_VERSION,
                "runtime":{"python":sys.version.split()[0],"pandas":pd.__version__,"numpy":np.__version__}},
            "coverage_by_feature":{**inc_cov,**ext_cov},"quality_status":"partial","vintage_mode":"reconstructed",
            "license_status":"unreviewed_no_public_redistribution", "partition_locations":{k:k for k in allfiles},
            "checksums":allfiles,"exclusions":audit["unresolved"],"breaking_changes":[],
            "authority_scope":"DATA-approved bytes, formulas and metadata for reconstructed research; not production",
            "optional_feature_sets":{"ohlcv_extensions_v1":{"partition":"extension_features.csv.gz",
                "count":len(extension_specs),"status":"implemented_not_performance_validated"}}}
        write_json(stage/"release.json",release)
        release_hash = sha256(stage/"release.json")
        validate_release(stage,release_hash)
        # mkdir above is only the parent. A lock prevents concurrent writers claiming this destination.
        lock = destination.with_name(destination.name+".lock")
        fd = os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
        try:
            if destination.exists():
                raise FileExistsError("release appeared during build")
            os.rename(stage,destination)
        finally:
            os.close(fd);lock.unlink()
        return {"release_id":release["release_id"],"release_sha256":release_hash,"root":str(destination),
                "incumbent_features":len(inc_specs),"extension_features":len(extension_specs),
                "candidate_catalog_count":legacy_candidates["count"],"rows":len(incumbent),
                "quality_status":"partial","vintage_mode":"reconstructed","production_approved":False}
    finally:
        if stage.exists():
            shutil.rmtree(stage)  # temporary output only; never touches source or published releases


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command",required=True)
    b = sub.add_parser("build")
    for name in ("source-root","source-release-sha256","handoff","handoff-sha256","out","generated-at","builder-commit"):
        b.add_argument("--"+name,required=True)
    v = sub.add_parser("verify");v.add_argument("--root",required=True);v.add_argument("--sha256",required=True)
    args = p.parse_args()
    if args.command == "verify":
        d = validate_release(Path(args.root),args.sha256)
        result = {"release_id":d["release_id"],"files_verified":len(d["checksums"]),"status":"verified_research_not_live"}
    else:
        result = build_release(Path(args.source_root),args.source_release_sha256,Path(args.handoff),
            args.handoff_sha256,Path(args.out),generated_at=args.generated_at,builder_commit=args.builder_commit)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
