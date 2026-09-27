"""Export a local handoff snapshot. Standard library only; no training or network."""
from __future__ import annotations

import hashlib
import json
import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "HANDOFF_MANIFEST.json"
OUT = ROOT / "outputs" / "handoff"
PUBLIC_FEEDBACK = "outputs/kaggle_feedback/public_submissions_v19_v24_20260927.json"
DECK = "outputs/presentation_rehearsal_v22_20260927"
DOCUMENTS = [
    "AGENTS.md", "HANDOFF.md", "HANDOFF_LOG.md", "README.md",
    "CURRENT_STATUS.md", "PRIVATE_SELECTION.md", "DELIVERY_REQUIREMENTS.md",
    "requirements.txt", "requirements-round2.txt",
    "reports/v22_method_and_code.md",
    "reports/round3_preprocessing_record.md",
    "reports/round79_report.md", "reports/round79_delivery.md",
    "reports/round80_delivery.md",
    "reports/round83_report.md", "reports/round83_delivery.md",
    "reports/round84_report.md", "reports/continuation_plan_after_r84.md",
    "reports/validation_design_audit_r84.md",
    "reports/reference_blending_audit_r84.md",
    "reports/github_reproducibility_audit_r83.md",
    "reports/presentation_rehearsal_v22_20260927.md",
    "outputs/round84/decision.json", "outputs/round84/results_review.json",
    "outputs/round79/selection/decision.json",
    "outputs/current_private_candidate.json",
    "outputs/round79/final_export/release.json",
    "outputs/round83/final_export/release.json",
    f"{DECK}/README.md",
    f"{DECK}/deliverables/臺灣房價預測_模擬報告_v22.pptx",
    f"{DECK}/deliverables/講稿與問答.md",
    f"{DECK}/build/build_deck.mjs",
    f"{DECK}/build/finalize_deck.mjs",
    f"{DECK}/build/source_data.json",
    f"{DECK}/build/final.validation.json",
    f"{DECK}/build/assets/taipei_markus_winkler.jpg",
]
# Identity checks only: these potentially large files are NOT copied into the ZIP.
KEY_ARTIFACTS = [
    "outputs/round79/final_model/model/model.cbm",
    "outputs/round79/final_model/model/feature_state.json",
    "outputs/round79/final_model/model/metadata.json",
    "outputs/round79/final_model/model/predictions_twd.npy",
    "outputs/round79/final/predictions.npy",
    "outputs/round71/final/nn_spatial.npy",
    "outputs/round46/trial/recency_log_spatial.npy",
    "outputs/round39/trial/artifacts/spatial_state.pkl",
]
READ_FIRST = """先讀 AGENTS.md、HANDOFF.md、HANDOFF_LOG.md。
本 ZIP 是交接閱讀與程式快照，不是完整訓練或推論環境。
包含文件、原 src 程式、Public 證據及模擬簡報。
不包含原始 Excel、提交 CSV、大型模型、data_cache、data_external、虛擬環境或完整歷史輸出。
本機檔案身分與是否隨包提供見 HANDOFF_MANIFEST.json。
連結驗證針對來源工作區，不保證每份歷史報告的全部連結都隨本包提供。
要接續訓練／推論，須另取必要資料、模型與狀態並建立合適環境，詳見 HANDOFF.md。
"""

def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def local_path(relative: str) -> Path:
    path = ROOT / relative
    if not path.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError(f"Path outside project: {relative}")
    return path

def record(relative: str, included: bool) -> dict:
    path = local_path(relative)
    if not path.is_file():
        raise FileNotFoundError(relative)
    return {"path": relative, "size_bytes": path.stat().st_size,
            "sha256": digest(path), "included_in_zip": included}

def validate_links() -> int:
    count = 0
    for name in ("HANDOFF.md", "AGENTS.md", "HANDOFF_LOG.md"):
        content = (ROOT / name).read_text(encoding="utf-8-sig")
        for target in re.findall(r"\[[^\]]+\]\(([^)]+)\)", content):
            target = target.split("#", 1)[0].strip("<>")
            if not target or re.match(r"^[a-zA-Z]+://", target):
                continue
            if target == "HANDOFF_MANIFEST.json":
                continue  # This command generates it after preflight succeeds.
            if not local_path(target).exists():
                raise FileNotFoundError(f"{name}: {target}")
            count += 1
    return count

def main() -> None:
    now = datetime.now(timezone.utc)
    links_checked = validate_links()
    feedback = json.loads((ROOT / PUBLIC_FEEDBACK).read_text(encoding="utf-8-sig"))
    identity_paths = list(KEY_ARTIFACTS)
    csv_checks = []
    handoff = (ROOT / "HANDOFF.md").read_text(encoding="utf-8-sig")
    for filename, expected in feedback["local_csv_sha256"].items():
        relative = f"outputs/{filename}"
        actual = digest(local_path(relative))
        if actual != expected:
            raise ValueError(f"CSV differs from recorded identity: {relative}")
        score = feedback["screenshotted_scores"][filename]
        if f"{score:,.4f}" not in handoff:
            raise ValueError(f"HANDOFF public score missing: {filename}")
        csv_checks.append({"path": relative, "sha256": actual, "matches": True})
        identity_paths.append(relative)

    pptx = f"{DECK}/deliverables/臺灣房價預測_模擬報告_v22.pptx"
    receipt = json.loads((ROOT / f"{DECK}/build/final.validation.json").read_text(encoding="utf-8"))
    expected_deck_hash = receipt["finalSha256"]
    if digest(local_path(pptx)) != expected_deck_hash:
        raise ValueError("Presentation differs from finalized receipt")

    package_paths = set(DOCUMENTS)
    # Preserve all top-level Python research sources without importing/executing them.
    package_paths.update(p.relative_to(ROOT).as_posix() for p in (ROOT / "src").glob("*.py"))
    package_paths.update(p.relative_to(ROOT).as_posix()
                         for p in (ROOT / "outputs/kaggle_feedback").iterdir()
                         if p.is_file() and p.suffix.lower() in {".json", ".png", ".sha256"})
    entries = [record(name, True) for name in sorted(package_paths)]
    entries += [record(name, False) for name in sorted(set(identity_paths) - package_paths)]
    manifest = {
        "schema_version": "handoff-snapshot-v1",
        "created_at_utc": now.isoformat(),
        "source_project_path": str(ROOT),
        "scope": "Documentation/source snapshot; selected artifact identities only, not a complete model dependency manifest.",
        "not_included": ["raw Excel", "submission CSV", "model weights", "data_cache",
                         "data_external", "virtual environments", "complete historical outputs"],
        "validation": {
            "handoff_relative_links_checked_in_source_workspace": links_checked,
            "public_feedback_source": PUBLIC_FEEDBACK,
            "local_csv_identity_checks": csv_checks,
            "uploaded_kaggle_bytes_verified": False,
            "presentation_receipt_sha256_matches": True,
            "training_or_prediction_executed": False,
            "semantic_status_requires_manual_maintenance": True,
        },
        "files": entries,
    }
    manifest_bytes = (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    # Resolve and bound the output before creating it. No deletes or overwrites of old ZIPs.
    local_path("outputs/handoff").mkdir(parents=True, exist_ok=True)
    stamp = now.strftime("%Y%m%dT%H%M%S%fZ")
    archive = OUT / f"handoff_{stamp}.zip"
    with zipfile.ZipFile(archive, mode="x", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("READ_FIRST.txt", READ_FIRST.encode("utf-8"))
        z.writestr("HANDOFF_MANIFEST.json", manifest_bytes)
        for entry in entries:
            if entry["included_in_zip"]:
                payload = local_path(entry["path"]).read_bytes()
                if hashlib.sha256(payload).hexdigest() != entry["sha256"]:
                    raise RuntimeError(f"File changed during export: {entry['path']}")
                z.writestr(entry["path"], payload)

    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None:
            raise RuntimeError("ZIP CRC verification failed")
        for entry in entries:
            if entry["included_in_zip"]:
                if hashlib.sha256(z.read(entry["path"])).hexdigest() != entry["sha256"]:
                    raise RuntimeError(f"ZIP member hash mismatch: {entry['path']}")
        if z.read("HANDOFF_MANIFEST.json") != manifest_bytes:
            raise RuntimeError("Packaged manifest mismatch")

    # Update pointers only after the new snapshot is verified. Older ZIPs remain.
    MANIFEST.write_bytes(manifest_bytes)
    latest = {
        "created_at_utc": now.isoformat(),
        "archive_path": archive.relative_to(ROOT).as_posix(),
        "archive_sha256": digest(archive),
        "archive_size_bytes": archive.stat().st_size,
        "manifest_path": "HANDOFF_MANIFEST.json",
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "included_files": len(package_paths) + 2,
        "all_packaged_file_hashes_verified": True,
        "workspace_link_count_checked": links_checked,
        "complete_retraining_backup": False,
    }
    (OUT / "latest.json").write_text(json.dumps(latest, ensure_ascii=False, indent=2) + "\n",
                                    encoding="utf-8")
    print(json.dumps(latest, ensure_ascii=False))

if __name__ == "__main__":
    main()
