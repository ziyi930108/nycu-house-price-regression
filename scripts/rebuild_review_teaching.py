"""Rebuild the frozen 4,065-Han review notes in three existing presentation styles.

This entry only rebuilds PPTX files. It never trains or runs a pricing model.
The underlying notes builder remains unchanged. Run the presentation skill's
edit marker separately before authoring; --check-only performs no writes.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import runpy
import shutil
import subprocess
import time
from zipfile import ZipFile


REVIEW = Path("teaching/v22_review_20261002")
EVIDENCE = REVIEW / "04_程式與證據"
SOURCE = Path("teaching/v22_styles_20261001/fifteen_minute")
NAMES = {"A": "白底學術", "B": "深色科技", "C": "暖色雜誌"}
PINS = {
    EVIDENCE / "本次內容/narrative.json": "b907f4e71a1f4f912f951f7c7780366e3b2aee9268aca44586611a3144f944b6",
    EVIDENCE / "本次內容/notes_plan.json": "2b9f4b830d75621a716c432e6d6944dcfc6ae68539a5c992480554946699eba8",
    EVIDENCE / "本次製作程式/run_build.py": "dd11527bead0ef2bd8b13eae240f461640fcef7416a196b9efb5ecc0de118e0c",
    EVIDENCE / "本次製作程式/preserve_package.py": "fba374079356a71937f1ce235aac0f44913e754fa3c1a34bb127803c986adbd5",
    EVIDENCE / "本次製作程式/edit_notes.mjs": "e71d12ef762aeb4b7ee40654e327191762a99ca1e466b3c396a79b395ad546db",
    EVIDENCE / "本次製作程式/finalize_notes.mjs": "bfc57510792e1645700ddfa6222ae1741af08597f2dfe81754d9504ece59b886",
    EVIDENCE / "本次簡報製作收據.json": "7f391c02c39559fec58a76f6a80e64d64665761004907be81a3a9d1eecf5d45f",
    SOURCE / "delivery_receipt.json": "2bf59705a8c5aeb86c039e006e188c760297aacf1fac02ad46b7f8af806de199",
}
SOURCE_SHA = {
    "A": "1d7fa4cd7aae76c4620357d5b3f2e1839ba43cadf3fee0669c5f82fc87aaee00",
    "B": "08d716d6317fc837295f0fa9573d906e7ce71b063eb1fb539de5ce824d9c88a6",
    "C": "4c8ac06efd61c1c4395b0bee939dec78363a3094d6761f10697b7961e66f6261",
}
REVIEW_SHA = {
    "A": "30526e504021249ee2a5fadf41c655ecf533c7dec703bad1f5a34d564c531512",
    "B": "5e286f34aecc4cad8183694a07fc009c7352c4cd60298ebf509b64236e1d4216",
    "C": "feb42d8605b41ed858a4194f6ecc96e475f1b61d3c0ca941768a6cd6ee4a09f8",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def overlap(first: Path, second: Path) -> bool:
    return first.is_relative_to(second) or second.is_relative_to(first)


def unchanged(identities: dict[str, str]) -> None:
    for path, expected in identities.items():
        if sha(Path(path)) != expected:
            raise ValueError("Protected input changed: " + path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True, help="New directory outside the repository/runtime")
    for name in ("node", "python", "node-modules", "presentation-skill"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--check-only", action="store_true", help="Read and validate inputs; create no directory or PPTX")
    args = parser.parse_args()
    if os.path.lexists(args.output):
        parser.error("Output already exists; choose a new directory")
    repo, output = args.repository.resolve(), args.output.resolve()
    node, python, modules, skill = (
        path.resolve() for path in (args.node, args.python, args.node_modules, args.presentation_skill)
    )
    for protected in (repo, modules, skill, node.parent, python.parent):
        if overlap(output, protected):
            raise ValueError("Output overlaps protected repository/runtime: " + str(protected))
    runtime_files = [node, python, modules / "@oai/artifact-tool/package.json"] + [
        skill / "container_tools" / name for name in (
            "artifact_tool_utils.mjs", "inspect_presentation_package_integrity.py",
            "inspect_presentation_layout_geometry.py",
        )
    ]
    if any(not path.is_file() for path in runtime_files):
        raise ValueError("A required bundled runtime file is missing")

    pins = dict(PINS)
    for key, label in NAMES.items():
        pins[SOURCE / "deliverables" / f"臺灣房價預測_v22_{label}_15分鐘版.pptx"] = SOURCE_SHA[key]
        pins[REVIEW / "01_上台簡報" / f"臺灣房價預測_v22_{label}_審查版.pptx"] = REVIEW_SHA[key]
    identities = {}
    for relative, expected in pins.items():
        path = repo / relative
        if not path.resolve().is_relative_to(repo):
            raise ValueError("Repository input escapes repository: " + str(relative))
        if sha(path) != expected:
            raise ValueError("Frozen source SHA mismatch: " + relative.as_posix())
        identities[str(path)] = expected
    identities[str(Path(__file__).resolve())] = sha(Path(__file__))
    runtime_identities = {str(path): sha(path) for path in runtime_files}

    content = repo / EVIDENCE / "本次內容"
    builder = repo / EVIDENCE / "本次製作程式"
    plan, narrative = read_json(content / "notes_plan.json"), read_json(content / "narrative.json")
    original_reference = read_json(repo / SOURCE / "delivery_receipt.json")
    review_reference = read_json(repo / EVIDENCE / "本次簡報製作收據.json")
    original = {entry["style"]: entry for entry in original_reference["styles"]}
    target = {entry["style"]: entry for entry in review_reference["outputs"]}
    if list(original) != list(NAMES) or list(target) != list(NAMES):
        raise ValueError("Reference styles or order changed")
    slides = narrative["slides"]
    if [slide["number"] for slide in slides] != list(range(1, 24)):
        raise ValueError("Expected 23 slides in order")
    if [slide["is_appendix"] for slide in slides] != [False] * 20 + [True] * 3:
        raise ValueError("Expected twenty main slides and three appendices")
    han = sum(len(re.findall(r"[\u4e00-\u9fff]", slide["script"])) for slide in slides[:20])
    seconds = sum(slide["seconds"] for slide in slides[:20])
    if han != 4065 or seconds != 885 or plan["narrative_sha256"] != sha(content / "narrative.json"):
        raise ValueError("Frozen main narrative identity/count/timing changed")
    if [entry["style"] for entry in plan["styles"]] != list(NAMES):
        raise ValueError("Notes plan styles or order changed")
    # Only imports pinned, definition-only helpers. run_path does not write pyc.
    helper = runpy.run_path(str(builder / "preserve_package.py"))
    for entry in plan["styles"]:
        key, notes = entry["style"], entry["notes"]
        source_name = f"臺灣房價預測_v22_{NAMES[key]}_15分鐘版.pptx"
        if entry["source_file"] != source_name or entry["source_sha256"] != SOURCE_SHA[key]:
            raise ValueError("Notes plan names another source deck")
        if entry["changed_slide_numbers"] != [12, 13] or len(notes) != 23:
            raise ValueError("Frozen notes change scope changed")
        if any(not isinstance(note, str) or not slide["script"] or note.count(slide["script"]) != 1
               for slide, note in zip(slides, notes)):
            raise ValueError("Expected each complete script exactly once in its note")
        source_deck = repo / SOURCE / "deliverables" / source_name
        review_deck = repo / REVIEW / "01_上台簡報" / source_name.replace("_15分鐘版", "_審查版")
        with ZipFile(source_deck) as package:
            order = helper["note_order"](package)
            changes = [index + 1 for index, part in enumerate(order)
                       if helper["text"](package.read(part)) != notes[index]]
        if changes != [12, 13]:
            raise ValueError("Source notes differ beyond slides 12 and 13")
        helper["verify"](source_deck, review_deck, entry)
        if original[key]["sha256"] != SOURCE_SHA[key] or target[key]["sha256"] != REVIEW_SHA[key]:
            raise ValueError("Reference PPTX SHA mismatch")
        if original[key]["render_sha256"] != target[key]["render_sha256"]:
            raise ValueError("Original/review rendered-slide references disagree")
    preflight = {
        "status": "preflight_passed", "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository": str(repo), "output": str(output), "input_sha256": identities,
        "runtime_file_sha256": runtime_identities,
        "runtime_paths": {"node": str(node), "python": str(python), "node_modules": str(modules), "skill": str(skill)},
        "runtime_versions": {
            "node": subprocess.check_output([str(node), "--version"], text=True).strip(),
            "python": subprocess.check_output([str(python), "--version"], text=True).strip(),
        },
        "main_han": han, "planned_main_seconds": seconds,
        "changed_slide_numbers": [12, 13], "new_ML_fits": 0,
        "outputs_in_scope": "three PPTX decks and validation renders only",
        "marker_must_have_been_run_separately_before_authoring": True,
    }
    unchanged(identities)
    unchanged(runtime_identities)
    if args.check_only:
        print(json.dumps(preflight, ensure_ascii=False, indent=2))
        return

    output.mkdir(parents=True, exist_ok=False)
    save(output / "entry_preflight.json", preflight)
    started = time.perf_counter()
    try:
        (output / "content").mkdir()
        for name in ("narrative.json", "notes_plan.json"):
            shutil.copy2(content / name, output / "content" / name)
        command = [str(python), "-B", "-X", "utf8", str(builder / "run_build.py"),
                   "--source-dir", str(repo / SOURCE / "deliverables"), "--work-dir", str(output),
                   "--runtime-node", str(node), "--runtime-python", str(python),
                   "--runtime-node-modules", str(modules), "--skill-dir", str(skill),
                   "--reference-receipt", str(repo / SOURCE / "delivery_receipt.json")]
        environment = os.environ.copy()
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        with (output / "builder.stdout.log").open("xb") as stdout, (output / "builder.stderr.log").open("xb") as stderr:
            result = subprocess.run(command, cwd=output, env=environment, stdout=stdout, stderr=stderr)
        if result.returncode:
            raise RuntimeError("Notes builder failed; keep logs and use a new output directory")
        build = read_json(output / "build/execution.json")
        if build["status"] != "completed" or build["new_ML_fits"] != 0 or len(build["outputs"]) != 3:
            raise ValueError("Unexpected notes-builder result")
        for entry in build["outputs"]:
            key = entry["style"]
            if entry["sha256"] != REVIEW_SHA[key] or sha(output / entry["path"]) != REVIEW_SHA[key]:
                raise ValueError("Rebuilt PPTX bytes differ from frozen review edition: " + key)
            if entry["render_sha256"] != target[key]["render_sha256"]:
                raise ValueError("Rebuilt PNG bytes differ from frozen review edition: " + key)
            if not entry["all_23_notes_match"] or not entry["original_embedded_workbooks_preserved"]:
                raise ValueError("Notes or embedded-workbook verification failed")
        unchanged(identities)
        unchanged(runtime_identities)
        receipt = {
            "status": "completed", "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": time.perf_counter() - started, "builder_command": command,
            "builder_receipt": "build/execution.json", "builder_receipt_sha256": sha(output / "build/execution.json"),
            "outputs": build["outputs"], "all_project_inputs_from_repository": True,
            "all_three_PPTX_bytes_identical_to_review": True, "all_69_render_bytes_identical_to_references": True,
            "frozen_sources_and_runtime_files_unchanged": True, "new_ML_fits": 0,
            "html_pdf_video_rebuilt": False, "clean_runtime_installation_tested": False,
            "PowerPoint_desktop_tested": False, "human_rehearsal_measured": False,
        }
        save(output / "rebuild_receipt.json", receipt)
        print(json.dumps({key: value for key, value in receipt.items() if key != "outputs"}, ensure_ascii=False, indent=2))
    except Exception as error:
        save(output / "entry_failure.json", {
            "status": "failed", "error": str(error), "elapsed_seconds": time.perf_counter() - started,
            "instruction": "Preserve this directory and use a new output path for another attempt.",
        })
        raise


if __name__ == "__main__":
    main()
