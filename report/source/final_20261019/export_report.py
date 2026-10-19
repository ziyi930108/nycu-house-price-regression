"""Copy the reviewed report and export its ordered Word body to Markdown.

This is document export only; it never trains or runs a price model, and it
does not regenerate the reviewed PDF. The older section-removal builder in
the parent directory is preserved as a historical source.
"""
from pathlib import Path
from hashlib import sha256
from zipfile import ZipFile
from xml.etree import ElementTree as ET
import argparse
import html
import json
import posixpath
import shutil

SOURCE = Path(__file__).resolve().parent
NAME = "台灣房屋銷售價格預測挑戰賽_正式報告.docx"
EXPECTED_DOCX_SHA256 = "f3a6b3402eaada071c6bb0ac1a93bef3a19ba1b7655da2bcfc6329a456b450f5"
NS = {
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
}
ASSETS = {
    "3943c11c41ae343ee8f20964862513e9dd8dba921fe6d96dc06ee9df5c0c2348": "city_price_distribution.png",
    "c7a681b36d54a926d870029b3659679b034f893f6622d33ed32467ba911779d0": "program_flow.png",
    "7269b22f74029de2505316622ce8bd4c5faabc7712c3f371a7301ebfae86af56": "training_curve.png",
    "c1b6aa472fb937f4aff70b948de22f4a2fa4da6acc4d39d312c7e886573dab3a": "fusion_validation.png",
    "9820be5c41918e3f94a8b2b2103f8acdcbfc17229db317d7b828830453b81a74": "error_by_price_band.png",
}


def plain_text(node):
    parts = []
    for child in node.iter():
        if child.tag == f"{{{NS['w']}}}t":
            parts.append(child.text or "")
        elif child.tag == f"{{{NS['w']}}}tab":
            parts.append("\t")
        elif child.tag in {f"{{{NS['w']}}}br", f"{{{NS['w']}}}cr"}:
            parts.append("\n")
    return "".join(parts)


def markdown_text(value, table=False):
    value = html.escape(value, quote=False).replace("\r\n", "\n")
    if table:
        value = value.replace("|", "\\|")
    return value.replace("\n", "<br>")


def export_markdown(docx):
    chunks, blocks, images = [], [], []
    with ZipFile(docx) as archive:
        document = ET.fromstring(archive.read("word/document.xml"))
        relationships = ET.fromstring(archive.read("word/_rels/document.xml.rels"))
        rels = {item.attrib["Id"]: item.attrib["Target"] for item in relationships}
        body = document.find("w:body", NS)
        nodes = list(body)
        for index, node in enumerate(nodes):
            kind = node.tag.rsplit("}", 1)[-1]
            if kind == "p":
                text = plain_text(node)
                style_node = node.find("w:pPr/w:pStyle", NS)
                style = style_node.get(f"{{{NS['w']}}}val", "") if style_node is not None else ""
                for image in node.findall(".//a:blip", NS):
                    rel_id = image.attrib[f"{{{NS['r']}}}embed"]
                    image_part = posixpath.normpath(posixpath.join("word", rels[rel_id]))
                    image_hash = sha256(archive.read(image_part)).hexdigest()
                    asset = ASSETS[image_hash]
                    asset_path = SOURCE.parent / "assets" / asset
                    assert sha256(asset_path.read_bytes()).hexdigest() == image_hash, asset
                    caption = plain_text(nodes[index + 1]) if index + 1 < len(nodes) else asset
                    chunks.append(f"![{markdown_text(caption)}](source/assets/{asset})")
                    images.append({"part": image_part, "asset": asset, "sha256": image_hash, "caption": caption})
                if not text.strip():
                    continue
                prefix = {"Title": "# ", "AbstractHeading": "## ", "Heading1": "## ",
                          "Heading2": "### ", "Heading3": "#### ", "AppendixHeading": "## "}.get(style, "")
                chunks.append(prefix + markdown_text(text))
                blocks.append({"type": "paragraph", "style": style, "text": text})
            elif kind == "tbl":
                rows = []
                for row in node.findall("w:tr", NS):
                    cells = []
                    for cell in row.findall("w:tc", NS):
                        cells.append("\n".join(plain_text(p) for p in cell.findall("w:p", NS)))
                    rows.append(cells)
                width = len(rows[0])
                assert all(len(row) == width for row in rows), "Non-rectangular table needs explicit handling"
                lines = ["| " + " | ".join(markdown_text(cell, table=True) for cell in row) + " |" for row in rows]
                lines.insert(1, "| " + " | ".join(["---"] * width) + " |")
                chunks.append("\n".join(lines))
                blocks.append({"type": "table", "rows": rows})
        assert len(images) == 5
        assert sum(block["type"] == "table" for block in blocks) == 13
        assert sum(block.get("style") == "Heading1" for block in blocks) == 8
    return "\n\n".join(chunks) + "\n", {"ordered_body_blocks": blocks, "figures": images}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path, help="A new, non-existing output directory")
    args = parser.parse_args()
    reference = SOURCE / "reference_report.docx"
    assert sha256(reference.read_bytes()).hexdigest() == EXPECTED_DOCX_SHA256, "Reviewed Word SHA mismatch"
    output = args.output_dir.resolve()
    assert not output.exists(), "Refuse to overwrite an existing output directory"
    markdown, extraction = export_markdown(reference)
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(reference, output / NAME)
    (output / "REPORT.md").write_text(markdown, encoding="utf-8", newline="\n")
    # Match the relative figure links used by the repository's report/REPORT.md.
    asset_output = output / "source" / "assets"
    asset_output.mkdir(parents=True)
    for figure in extraction["figures"]:
        shutil.copyfile(SOURCE.parent / "assets" / figure["asset"], asset_output / figure["asset"])
    receipt = {
        "status": "exported_reviewed_document",
        "docx_sha256": EXPECTED_DOCX_SHA256,
        "markdown_sha256": sha256((output / "REPORT.md").read_bytes()).hexdigest(),
        "body_blocks": len(extraction["ordered_body_blocks"]),
        "tables": 13, "images": 5, "chapters": 8,
        "body_blocks_sha256": sha256(json.dumps(extraction["ordered_body_blocks"], ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest(),
        "model_training": False, "price_inference": False,
        "PDF_regenerated": False, "new_visual_review_performed": False,
    }
    (output / "export.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, ensure_ascii=False))


if __name__ == "__main__":
    main()
