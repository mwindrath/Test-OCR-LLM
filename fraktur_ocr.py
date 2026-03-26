#!/usr/bin/env python3
"""
Fraktur OCR & TEI Encoder
=========================
Analyses historical German Fraktur newspaper scans via the Anthropic Vision API.
Accepts individual files OR entire folder trees (recursive).

Produces per input file inside --output-dir (default: ./ocr_output):
  <rel/path/to/basename>_tei.xml            – TEI-P5 XML with layout zones
  <rel/path/to/basename>_transcription.txt  – plain text extracted locally from the TEI
                                              (no extra API call, ~0 extra tokens)

The output directory mirrors the relative sub-folder structure of the input.

Usage
-----
  # Single files
  python fraktur_ocr.py seite1.jpg seite2.png

  # Entire folder (recursive)
  python fraktur_ocr.py --folder /path/to/scans

  # Custom output directory
  python fraktur_ocr.py --folder /path/to/scans --output-dir /path/to/results

  # Skip already processed files
  python fraktur_ocr.py --folder /path/to/scans --skip-existing

Supported input formats: JPEG, PNG, GIF, WEBP, PDF (first page rendered to PNG)

Requirements
------------
  pip install anthropic pillow pdf2image

Environment
-----------
  ANTHROPIC_API_KEY  – your Anthropic key (or pass via --api-key)
"""

import argparse
import base64
import os
import re
import sys
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Dependency imports (with friendly error messages)
# ---------------------------------------------------------------------------
try:
    import anthropic
except ImportError:
    sys.exit("Missing dependency: pip install anthropic")

try:
    from PIL import Image
except ImportError:
    sys.exit("Missing dependency: pip install pillow")


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are a digital humanities OCR and TEI encoding assistant.
Your task is to process a scanned historical newspaper page written in German Fraktur.
Goals:
1. Perform high-quality Fraktur OCR.
2. Preserve the original newspaper layout (columns and article structure).
3. Identify sections such as articles, announcements, obituaries, and memorial tables.
4. Produce ONE output only: a complete TEI-P5 XML document with the full transcription
   embedded inside it. Do NOT produce a separate plain-text block.
Instructions:
1. OCR
   - Transcribe the page exactly inside TEI <p> elements.
   - Preserve historical spelling. Do not modernize words.
   - Mark uncertain readings with [?].
   - Add a confidence annotation after each uncertain token:
     [?:token:LOW]  = confidence < 50 %
     [?:token:MED]  = confidence 50-80 %
     [?:token:HIGH] = confidence > 80 % but still uncertain
2. Layout detection
   - Detect all columns (left to right order).
   - Detect article titles and section headings.
   - For every detected zone, estimate its bounding box in image pixels:
     ulx (upper-left x), uly (upper-left y), lrx (lower-right x), lry (lower-right y).
   - These coordinates are relative to the full page image (origin = top-left corner).
3. TEI layout encoding — use TEI P5 with the following structure.
   Every <zone> MUST carry pixel coordinates AND an @xml:id.
   The <graphic> MUST carry the full image URL in @url and the image dimensions
   in @width/@height (pixels) so a IIIF viewer can compute the region parameter.
   <TEI>
     <facsimile>
       <surface ulx="0" uly="0" lrx="PAGE_WIDTH_PX" lry="PAGE_HEIGHT_PX">
         <graphic url="IMAGE_URL" width="PAGE_WIDTH_PX" height="PAGE_HEIGHT_PX"/>
         <!-- one zone per detected layout region -->
         <zone xml:id="col_1"        type="column"  ulx="X1" uly="Y1" lrx="X2" lry="Y2"/>
         <zone xml:id="zone_article_1" type="article" ulx="X1" uly="Y1" lrx="X2" lry="Y2"/>
         <!-- additional zone types: heading | announcement | obituary | table -->
       </surface>
     </facsimile>
     <text>
       <body>
         <div type="article" facs="#zone_article_1">
           <head>Article title</head>
           <p>Full article text here.</p>
         </div>
       </body>
     </text>
   </TEI>
4. IIIF coordinate mapping
   - For each <zone>, derive the IIIF Image API region parameter:
       region = ulx,uly,width,height   (width = lrx-ulx, height = lry-uly)
   - Embed a <note type="iiif-region"> inside each <zone> with this value so
     a viewer can construct: {base_url}/{region}/full/0/default.jpg
   - Example:
       <zone xml:id="zone_article_1" type="article" ulx="120" uly="340" lrx="580" lry="820">
         <note type="iiif-region">120,340,460,480</note>
       </zone>
5. Text encoding — link text sections to layout zones using @facs.
Important constraints:
- Do NOT summarize text. Preserve the full article text inside <p> elements.
- Preserve line breaks when meaningful using <lb/>.
- Maintain column order left to right.
- Language of transcription: German.
- If the input image URL is not a IIIF endpoint, use it as-is in <graphic @url>;
  leave the IIIF region note in place for when the image is served via IIIF later.
- Coordinates are ESTIMATES based on visual analysis of the image. Mark the
  <surface> with @type="estimated-coordinates" if you cannot verify pixel dimensions.
CRITICAL OUTPUT RULES:
- Output ONLY the raw XML. No section markers, no prose, no markdown fences.
- The very first character of your response must be: 
- The response MUST start with: <?xml version="1.0" encoding="UTF-8"?>
- The response MUST end with: </TEI>
- Every opened XML tag must be properly closed. No truncation, no omissions.
- Do NOT use placeholder comments like <!-- content omitted --> to skip content.
- If you are close to your output limit, shorten <p> text rather than leaving any tag unclosed.
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def encode_image(path: Path, dpi: int = 150) -> tuple[str, str]:
    """Return (base64_data, media_type) for an image or first-page of a PDF.

    PDFs are rendered at `dpi` resolution.
    Raster images are downscaled so their longest side equals what `dpi` would
    produce for an A4 page (≈ dpi * 11.7 inches), preserving aspect ratio.
    This directly reduces image tokens ≈ (width × height) / 750.
    """
    import io
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        try:
            from pdf2image import convert_from_path  # type: ignore
        except ImportError:
            sys.exit("PDF support requires: pip install pdf2image")
        images = convert_from_path(str(path), first_page=1, last_page=1, dpi=dpi)
        if not images:
            sys.exit(f"Could not render PDF: {path}")
        buf = io.BytesIO()
        images[0].save(buf, format="PNG")
        data = base64.standard_b64encode(buf.getvalue()).decode("utf-8")
        return data, "image/png"

    media_map = {
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }
    media_type = media_map.get(suffix)
    if not media_type:
        sys.exit(f"Unsupported file type '{suffix}'. Use JPEG, PNG, GIF, WEBP or PDF.")

    img = Image.open(path)

    # Downscale raster images to match target DPI for an A4 page.
    # A4 long side at target dpi: ~dpi * 11.7 px
    max_px = int(dpi * 11.7)
    w, h = img.size
    longest = max(w, h)
    if longest > max_px:
        scale = max_px / longest
        new_size = (int(w * scale), int(h * scale))
        img = img.resize(new_size, Image.LANCZOS)
        print(f"  ↓ Resized {w}×{h} → {new_size[0]}×{new_size[1]} px ({dpi} dpi target)")

    buf = io.BytesIO()
    # Save as JPEG for photos/greys to reduce payload; keep PNG for palettised images
    save_fmt = "JPEG" if img.mode in ("RGB", "RGBA", "L") else "PNG"
    if img.mode == "RGBA" and save_fmt == "JPEG":
        img = img.convert("RGB")
    img.save(buf, format=save_fmt, quality=85)
    data = base64.standard_b64encode(buf.getvalue()).decode("utf-8")
    out_type = "image/jpeg" if save_fmt == "JPEG" else "image/png"
    return data, out_type


def parse_tei(raw: str) -> str:
    """Extract and validate the TEI-XML from the raw API response.

    The new prompt asks for raw XML only (no section markers), so we just
    clean up any accidental markdown fences and trim to </TEI>.
    """
    tei = raw.strip()

    # Strip accidental markdown fences
    tei = re.sub(r"^```[a-zA-Z]*\n?", "", tei)
    tei = re.sub(r"\n?```\s*$", "", tei).strip()

    # Trim to last </TEI>
    close_tag = "</TEI>"
    idx = tei.rfind(close_tag)
    if idx != -1:
        tei = tei[: idx + len(close_tag)]
    else:
        tei += "\n<!-- WARNING: </TEI> closing tag missing — response may have been truncated -->"
        print("  ⚠ </TEI> not found in output — XML may be incomplete.")

    return tei


def tei_to_plaintext(tei: str) -> str:
    """Extract readable plain text from TEI-XML without any extra API call.

    Strategy:
      - Pull text from <head>, <p>, <lb/> elements in document order.
      - Prefix <head> lines with their @type if available (e.g. [article]).
      - Preserve <lb/> as newlines.
      - Collect uncertainty markers from the text as a summary header.
      - Strip all remaining XML tags.
    """
    # Replace <lb/> variants with newlines before stripping tags
    tei_work = re.sub(r"<lb\s*/?>", "\n", tei)

    # Extract content of <head ...>…</head> with optional type attribute
    def replace_head(m: re.Match) -> str:
        attrs = m.group(1)
        content = m.group(2).strip()
        type_match = re.search(r'type=["\']([^"\']+)["\']', attrs)
        label = f"[{type_match.group(1)}] " if type_match else ""
        return f"\n\n{'='*40}\n{label}{content}\n{'='*40}\n"

    tei_work = re.sub(r"<head([^>]*)>(.*?)</head>", replace_head, tei_work, flags=re.DOTALL)

    # Separate <p> blocks with a blank line
    tei_work = re.sub(r"<p[^>]*>", "\n", tei_work)
    tei_work = re.sub(r"</p>", "\n", tei_work)

    # Strip all remaining tags
    tei_work = re.sub(r"<[^>]+>", "", tei_work)

    # Decode common XML entities
    tei_work = (tei_work
                .replace("&amp;",  "&")
                .replace("&lt;",   "<")
                .replace("&gt;",   ">")
                .replace("&quot;", '"')
                .replace("&apos;", "'"))

    # Collapse excessive blank lines
    tei_work = re.sub(r"\n{3,}", "\n\n", tei_work).strip()

    # Build uncertainty summary from the extracted text
    low  = re.findall(r"\[\?:([^:]+):LOW\]",  tei_work)
    med  = re.findall(r"\[\?:([^:]+):MED\]",  tei_work)
    high = re.findall(r"\[\?:([^:]+):HIGH\]", tei_work)
    lines = ["=== UNCERTAINTY SUMMARY ==="]
    if low:
        lines.append(f"LOW  (<50%):  {', '.join(low)}")
    if med:
        lines.append(f"MED  (50-80%): {', '.join(med)}")
    if high:
        lines.append(f"HIGH (>80%):  {', '.join(high)}")
    if not (low or med or high):
        lines.append("No uncertain tokens detected.")
    lines.append("=" * 28)
    header = "\n".join(lines) + "\n\n"

    return header + tei_work


MAX_TOKENS = 10000   # output token cap per request (reduced from 16k — sufficient for most pages)
MAX_CONTINUATIONS = 3  # safety cap for continuation loop


def call_api(client: anthropic.Anthropic, image_data: str, media_type: str,
             model: str = "claude-sonnet-4-5") -> str:
    """Send image to Claude and return the complete raw text response."""
    user_content = [
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": media_type,
                "data": image_data,
            },
        },
        {
            "type": "text",
            "text": (
                    "Process this historical German Fraktur newspaper scan. "
                    "Return ONLY the complete TEI-P5 XML document. "
                    "Start your response with <?xml version=\"1.0\" encoding=\"UTF-8\"?> "
                    "and end with </TEI>. No prose, no markdown, no section markers."
                ),
        },
    ]

    messages: list[dict] = [{"role": "user", "content": user_content}]
    accumulated = ""

    for attempt in range(1 + MAX_CONTINUATIONS):
        response = client.messages.create(
            model=model,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            messages=messages,
        )

        chunk = "".join(
            block.text for block in response.content if hasattr(block, "text")
        )
        accumulated += chunk
        stop_reason = response.stop_reason  # "end_turn" | "max_tokens" | …

        # Check whether the XML looks complete
        tei_done = "</TEI>" in accumulated
        finished = stop_reason == "end_turn" or tei_done

        if finished:
            if attempt > 0:
                print(f"  ✓ XML completed after {attempt} continuation(s).")
            break

        if attempt < MAX_CONTINUATIONS:
            print(f"  ⚠ Response truncated (stop_reason={stop_reason!r}), continuing …")
            # Feed back the full conversation so the model knows where it left off
            messages.append({"role": "assistant", "content": chunk})
            messages.append({
                "role": "user",
                "content": (
                    "Your response was cut off. Continue EXACTLY where you left off. "
                    "Do not repeat any text already written. "
                    "Close all open XML tags and finish with </TEI> on its own line."
                ),
            })
        else:
            print(
                "  ⚠ Reached continuation limit; TEI may still be incomplete. "
                "Try increasing MAX_TOKENS or splitting the image."
            )

    return accumulated


def save_outputs(
    file_path: Path,
    tei: str,
    txt: str,
    output_root: Path,
    input_root: Optional[Path] = None,
) -> None:
    """Write TEI-XML and locally-derived plain-text, mirroring subfolder structure."""
    if input_root is not None:
        try:
            rel = file_path.parent.relative_to(input_root)
        except ValueError:
            rel = Path()
    else:
        rel = Path()

    out_dir = output_root / rel
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = file_path.stem
    tei_path = out_dir / f"{stem}_tei.xml"
    txt_path = out_dir / f"{stem}_transcription.txt"

    tei_path.write_text(tei, encoding="utf-8")
    print(f"  ✓ XML  → {tei_path}")

    txt_path.write_text(txt, encoding="utf-8")
    print(f"  ✓ TXT  → {txt_path}  (extracted from TEI, 0 extra tokens)")


def validate_tei(tei: str) -> None:
    """Print a quick structural check on the TEI output."""
    checks = {
        "<?xml":    "XML declaration",
        "<TEI":     "<TEI> root",
        "<facsimile": "<facsimile>",
        "<text":    "<text>",
        "</TEI>":   "closing </TEI>",
    }
    print("\n  ── TEI structure check ──────────────────────────")
    all_ok = True
    for token, label in checks.items():
        found = token in tei
        status = "✓" if found else "✗ MISSING"
        print(f"  {status}  {label}")
        if not found:
            all_ok = False
    if all_ok:
        print("  All structural markers present.")
    print("  ─────────────────────────────────────────────────\n")


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".pdf"}


def collect_files(sources: list[str], folder: Optional[str]) -> list[Path]:
    """Return a sorted, deduplicated list of supported input files."""
    found: list[Path] = []

    # Explicit file arguments
    for src in sources:
        p = Path(src)
        if not p.exists():
            print(f"[SKIP] Not found: {p}", file=sys.stderr)
            continue
        if p.is_dir():
            # If user passes a directory as a positional arg, treat it like --folder
            for f in sorted(p.rglob("*")):
                if f.is_file() and f.suffix.lower() in SUPPORTED_SUFFIXES:
                    found.append(f)
        elif p.suffix.lower() in SUPPORTED_SUFFIXES:
            found.append(p)
        else:
            print(f"[SKIP] Unsupported type: {p}", file=sys.stderr)

    # --folder recursive scan
    if folder:
        root = Path(folder)
        if not root.is_dir():
            sys.exit(f"--folder path is not a directory: {root}")
        for f in sorted(root.rglob("*")):
            if f.is_file() and f.suffix.lower() in SUPPORTED_SUFFIXES:
                found.append(f)

    # Deduplicate while preserving order
    seen: set[Path] = set()
    unique: list[Path] = []
    for f in found:
        key = f.resolve()
        if key not in seen:
            seen.add(key)
            unique.append(f)
    return unique


def already_processed(file_path: Path, output_root: Path, input_root: Optional[Path]) -> bool:
    """Return True if all three output files already exist for this input."""
    if input_root is not None:
        try:
            rel = file_path.parent.relative_to(input_root)
        except ValueError:
            rel = Path()
    else:
        rel = Path()
    out_dir = output_root / rel
    stem = file_path.stem
    return all(
        (out_dir / name).exists()
        for name in (
            f"{stem}_transcription.txt",
            f"{stem}_tei.xml",
        )
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fraktur OCR & TEI encoder via Anthropic Vision API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python fraktur_ocr.py seite1.jpg seite2.png\n"
            "  python fraktur_ocr.py --folder /scans\n"
            "  python fraktur_ocr.py --folder /scans --output-dir /results --skip-existing\n"
            "  python fraktur_ocr.py --folder /scans --dpi 200 --model claude-opus-4-5\n"
            "  python fraktur_ocr.py --folder /scans --dpi 100 --model claude-haiku-4-5-20251001\n"
        ),
    )
    parser.add_argument(
        "files",
        nargs="*",
        metavar="FILE",
        help="Individual image(s) or PDF(s) to process",
    )
    parser.add_argument(
        "--folder", "-f",
        metavar="DIR",
        help="Folder to scan recursively for supported images/PDFs",
    )
    parser.add_argument(
        "--output-dir", "-o",
        metavar="DIR",
        default="./ocr_output",
        help="Root directory for output files (default: ./ocr_output)",
    )
    parser.add_argument(
        "--skip-existing", "-s",
        action="store_true",
        help="Skip files whose three output files already exist",
    )
    parser.add_argument(
        "--model", "-m",
        default="claude-sonnet-4-5",
        metavar="MODEL",
        help=(
            "Anthropic model to use (default: claude-sonnet-4-5). "
            "Use claude-opus-4-5 for maximum quality, "
            "claude-haiku-4-5-20251001 for fastest/cheapest."
        ),
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        metavar="N",
        help=(
            "Resolution for PDF rendering and raster-image downscaling (default: 150). "
            "150 dpi saves ~75%% of image tokens vs 300 dpi. "
            "Use 200–300 for damaged or very small print."
        ),
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("ANTHROPIC_API_KEY", ""),
        help="Anthropic API key (default: $ANTHROPIC_API_KEY)",
    )
    args = parser.parse_args()

    if not args.api_key:
        sys.exit("No API key found. Set ANTHROPIC_API_KEY or pass --api-key <key>.")

    if not args.files and not args.folder:
        parser.print_help()
        sys.exit("\nError: provide at least one FILE or --folder DIR.")

    # Determine roots for relative path mirroring
    input_root: Optional[Path] = Path(args.folder).resolve() if args.folder else None
    output_root = Path(args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    all_files = collect_files(args.files, args.folder)
    if not all_files:
        sys.exit("No supported files found.")

    total = len(all_files)
    print(f"\nFound {total} file(s) to process.")
    print(f"Output root : {output_root}")
    print(f"Model       : {args.model}")
    print(f"DPI         : {args.dpi}\n")

    client = anthropic.Anthropic(api_key=args.api_key)
    processed = skipped = errors = 0

    for idx, path in enumerate(all_files, 1):
        print(f"[{idx}/{total}] {'='*50}")
        print(f"  File: {path}")

        if args.skip_existing and already_processed(path, output_root, input_root):
            print("  ↷ Already processed — skipping (use without --skip-existing to rerun).")
            skipped += 1
            continue

        print("  → Encoding image …")
        try:
            image_data, media_type = encode_image(path, dpi=args.dpi)
        except SystemExit as exc:
            print(f"  ✗ Encode error: {exc}", file=sys.stderr)
            errors += 1
            continue

        print("  → Calling Anthropic Vision API …")
        try:
            raw_response = call_api(client, image_data, media_type, model=args.model)
        except anthropic.APIError as exc:
            print(f"  ✗ API error: {exc}", file=sys.stderr)
            errors += 1
            continue

        print("  → Parsing TEI …")
        tei = parse_tei(raw_response)

        if not tei:
            print("  ⚠ Empty response — saving raw output.")
            raw_dir = output_root / (
                path.parent.relative_to(input_root)
                if input_root else Path()
            )
            raw_dir.mkdir(parents=True, exist_ok=True)
            (raw_dir / f"{path.stem}_raw.txt").write_text(raw_response, encoding="utf-8")
            errors += 1
            continue

        print("  → Extracting plain text from TEI (local, no API call) …")
        txt = tei_to_plaintext(tei)

        print("  → Saving outputs …")
        save_outputs(path, tei, txt, output_root, input_root)
        validate_tei(tei)
        processed += 1

    # Final summary
    print(f"\n{'='*55}")
    print(f"  Done.  Processed: {processed}  |  Skipped: {skipped}  |  Errors: {errors}")
    print(f"  Output directory: {output_root}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
