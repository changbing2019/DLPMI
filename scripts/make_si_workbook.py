#!/usr/bin/env python3
"""Collect the 20 supporting CSVs into one Excel workbook for submission.

Twenty separate uploads is awkward in Paragon Plus. This builds
`EST_Supporting_Data.xlsx` with a Contents sheet and one sheet per file, in the
order Text S9 lists them.

The sheet descriptions are READ FROM Text S9 rather than retyped, so the
workbook cannot drift from the document that describes it. Sheet names are the
CSV stems: all 20 fit inside Excel's 31-character limit and none contains an
illegal character, checked before this was written. No cell is date-like, so
nothing gets coerced on the way in.

The CSVs stay on disk. They are what the repository ships and what
REPRODUCE.md refers to; the workbook is a submission convenience built from
them.

    python make_si_workbook.py              # build, and report
    python make_si_workbook.py --update-si  # also rewrite Text S9 and the
                                            # Associated Content format tag

A note on format. CSV is the more archival choice, and a reviewer who wants to
load one file into a script will prefer it. If ES&T raises no objection to a
workbook this is purely a convenience; if it does, submit the CSVs and revert
Text S9 from the backup this script writes.
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import sys

import pandas as pd

R = r"D:\projects\carbon isotopic\DLMPI_reanalysis"
FC = os.path.join(R, "From_Claude_chat")
SD = os.path.join(FC, "supplementary_data")
SI = os.path.join(FC, "manuscript", "WRR_Groundwater_Age_Identifiability_SI.docx")
OUT = os.path.join(FC, "manuscript", "EST_Supporting_Data.xlsx")


def text_s9():
    """[(csv name, description)] in the order Text S9 lists them."""
    import docx
    d = docx.Document(SI)
    on, items = False, []
    for p in d.paragraphs:
        t = p.text.strip()
        if t.startswith("Text S9.") and p.style.name.startswith("Heading"):
            on = True
            continue
        if on and t.startswith("Text S10."):
            break
        if on:
            # accept both the pre-workbook form 'name.csv - desc' and the
            # post-workbook form 'name - desc', so this stays re-runnable
            m = re.match(r"^([A-Za-z0-9_\-]+)(?:\.csv)?\s*\u2014\s*(.+)$", t)
            if m and not t.startswith("The supporting data"):
                items.append((m.group(1), m.group(2).strip()))
    return items


def build(items):
    from openpyxl.styles import Alignment, Font
    from openpyxl.utils import get_column_letter

    missing = [n for n, _ in items if not os.path.exists(
        os.path.join(SD, n + ".csv"))]
    if missing:
        sys.exit("ABORT missing CSVs: %s" % missing)

    rows = []
    with pd.ExcelWriter(OUT, engine="openpyxl") as xw:
        # placeholder so Contents lands first; filled in below
        pd.DataFrame().to_excel(xw, sheet_name="Contents", index=False)
        for name, desc in items:
            df = pd.read_csv(os.path.join(SD, name + ".csv"))
            if len(name) > 31:
                sys.exit("ABORT sheet name too long: %s" % name)
            df.to_excel(xw, sheet_name=name, index=False)
            rows.append((name, len(df), len(df.columns), desc))

        contents = pd.DataFrame(rows, columns=[
            "Sheet", "Rows", "Columns", "Description"])
        contents.to_excel(xw, sheet_name="Contents", index=False)

        wb = xw.book
        for ws in wb.worksheets:
            ws.freeze_panes = "A2"
            for c in ws[1]:
                c.font = Font(bold=True)
                c.alignment = Alignment(vertical="top")
            for j, col in enumerate(ws.iter_cols(min_row=1, max_row=1), 1):
                head = str(col[0].value or "")
                width = min(max(len(head) + 2, 10), 30)
                if ws.title == "Contents" and head == "Description":
                    width = 110
                elif ws.title == "Contents" and head == "Sheet":
                    width = 30
                ws.column_dimensions[get_column_letter(j)].width = width
        if "Contents" in wb.sheetnames:
            wb.move_sheet("Contents", offset=-wb.sheetnames.index("Contents"))
    return rows


def update_si(items):
    import docx
    shutil.copy(SI, SI + ".bak_pre_workbook")
    d = docx.Document(SI)

    # find the Text S9 entries and replace them with one workbook paragraph
    body, on, targets, head = d.element.body, False, [], None
    for p in d.paragraphs:
        t = p.text.strip()
        if t.startswith("Text S9.") and p.style.name.startswith("Heading"):
            on, head = True, p
            continue
        if on and t.startswith("Text S10."):
            break
        if on and t:
            targets.append(p)
    if head is None or not targets:
        sys.exit("ABORT could not locate the Text S9 entries")

    lead = ("The supporting data are provided as a single Excel workbook, "
            "EST_Supporting_Data.xlsx. Its Contents sheet lists every sheet "
            "with a description and its dimensions; the sheets are named as "
            "below and appear in this order.")
    # INSERT the lead sentence rather than consuming an entry paragraph --
    # there are exactly as many entry paragraphs as data files, so reusing one
    # leaves the last file with nowhere to go
    import copy as _copy
    lead_el = _copy.deepcopy(targets[0]._p)
    head._p.addnext(lead_el)
    import docx as _docx
    lead_p = _docx.text.paragraph.Paragraph(lead_el, targets[0]._parent)
    if not lead_p.runs:
        lead_p.add_run("")
    lead_p.runs[0].text = lead
    for r in lead_p.runs[1:]:
        r.text = ""
    # the originals now hold one sheet description each
    keep = targets
    for i, (name, desc) in enumerate(items):
        line = "%s \u2014 %s" % (name, desc)
        if i < len(keep):
            p = keep[i]
            if not p.runs:
                p.add_run("")
            p.runs[0].text = line
            for r in p.runs[1:]:
                r.text = ""
        else:
            sys.exit("ABORT not enough paragraphs to hold %d entries"
                     % len(items))
    for p in keep[len(items):]:
        el = p._p
        el.getparent().remove(el)
    d.save(SI)
    print("  Text S9 now describes the workbook (%d sheet entries)"
          % len(items))


def update_assoc():
    import docx
    M = os.path.join(FC, "manuscript", "WRR_Groundwater_Age_Identifiability.docx")
    lock = os.path.join(os.path.dirname(M), "~$T_Environmental Tracer.docx")
    if os.path.exists(lock):
        print("  SKIPPED Associated Content: the manuscript is open in Word.")
        print("  Re-run with --update-si once it is closed, or change")
        print("  '(CSV)' to '(XLSX)' in the Supporting Information sentence.")
        return False
    d = docx.Document(M)
    hits = 0
    for p in d.paragraphs:
        if p.text.strip().startswith("Supporting Information.") \
                and "(CSV)" in p.text:
            for r in p.runs:
                if "(CSV)" in r.text:
                    r.text = r.text.replace("(CSV)", "(XLSX)")
                    hits += 1
                    break
    if hits != 1:
        print("  WARNING Associated Content '(CSV)' matched %d times" % hits)
        return False
    d.save(M)
    print("  Associated Content: (CSV) -> (XLSX)")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--update-si", action="store_true")
    A = ap.parse_args()

    items = text_s9()
    print("Text S9 lists %d data files" % len(items))
    if len(items) != 20:
        sys.exit("ABORT expected 20 entries, parsed %d" % len(items))

    rows = build(items)
    print("wrote %s  (%.2f MB, %d sheets incl. Contents)"
          % (OUT, os.path.getsize(OUT) / 1e6, len(rows) + 1))
    for name, nr, nc, _ in rows:
        print("   %-28s %5d rows x %2d cols" % (name, nr, nc))

    if A.update_si:
        print("\nupdating the documents:")
        update_si(items)
        update_assoc()


if __name__ == "__main__":
    main()
