import os
import glob
import traceback
import cv2
import gc
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

def generate_pdf_report(serial_number, panel_folder, seq_times=None, total_time=0, panel_start=None, cam2_image=None):
    """Generates a professional 5-page inspection report for the industrial panel (Cover, SEQ 1, SEQ 2, SEQ 3, Serial Best Picture)."""
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report_name = f"{serial_number}_Inspection_Report.pdf"
    report_path = os.path.join(panel_folder, report_name)
    
    print(f"\n📄 GENERATING PDF: {report_name}")
    
    doc = SimpleDocTemplate(report_path, pagesize=A4,
                            rightMargin=40, leftMargin=40,
                            topMargin=40, bottomMargin=40)
    story = []
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle('TitleStyle', parent=styles['Heading1'],
                                 fontSize=22, spaceAfter=20, alignment=1, textColor=colors.HexColor("#1a237e"))
    h1_style    = ParagraphStyle('H1Style', parent=styles['Heading1'],
                                 fontSize=16, spaceAfter=12, textColor=colors.HexColor("#1a237e"))
    h2_style    = ParagraphStyle('H2Style', parent=styles['Heading2'],
                                 fontSize=14, spaceAfter=10, textColor=colors.HexColor("#1565c0"))
    sub_style   = ParagraphStyle('SubStyle', parent=styles['Normal'],
                                 fontSize=10, leading=14, spaceAfter=8)
    
    USABLE_W = A4[0] - 80

    def _find(folder, *patterns):
        """Try multiple glob patterns in order; return first non-empty match.
        FIX 3: Search both folder root AND one level of subdirectories for
        robustness when the serial-rename step produced a nested path."""
        import glob as _glob
        for pat in patterns:
            # Direct match in the given folder
            results = sorted(_glob.glob(os.path.join(folder, pat)))
            if results:
                return results
            # One-level deep (e.g. SSP-SEQ/2025-01-01/SSP-SEQ_123456/<file>)
            results = sorted(_glob.glob(os.path.join(folder, '**', pat)))
            if results:
                return results
        return []

    def _find_best(folder, *patterns):
        """Like _find but returns the sharpest image (highest Laplacian variance)
        from all matches — important when multiple frames were saved per sequence."""
        import glob as _glob
        import cv2 as _cv2
        seen = set()
        candidates = []
        for pat in patterns:
            for p in _glob.glob(os.path.join(folder, pat)):
                rp = os.path.realpath(p)
                if rp not in seen: seen.add(rp); candidates.append(p)
            for p in _glob.glob(os.path.join(folder, '**', pat)):
                rp = os.path.realpath(p)
                if rp not in seen: seen.add(rp); candidates.append(p)
        if not candidates:
            return []
        if len(candidates) == 1:
            return candidates
        # Pick sharpest
        best_path  = candidates[0]
        best_var   = -1.0
        for p in candidates:
            try:
                img = _cv2.imread(p)
                if img is None:
                    continue
                gray = _cv2.cvtColor(img, _cv2.COLOR_BGR2GRAY)
                var  = _cv2.Laplacian(gray, _cv2.CV_64F).var()
                if var > best_var:
                    best_var  = var
                    best_path = p
            except Exception:
                pass
        return [best_path]

    def _fit_image(img_path, max_w_pt, max_h_pt):
        try:
            img = Image(img_path)
            iw, ih = img.imageWidth, img.imageHeight
            aspect = ih / iw
            
            w = min(iw, max_w_pt)
            h = w * aspect
            
            if h > max_h_pt:
                h = max_h_pt
                w = h / aspect
                
            img.drawWidth = w
            img.drawHeight = h
            return img
        except Exception as e:
            return Paragraph(f"Error loading image: {e}", sub_style)

    # ── PAGE 1: COVER & SUMMARY ──────────────────────────────
    story.append(Paragraph("Industrial Panel Inspection Report", title_style))
    story.append(Spacer(1, 20))
    
    meta_data = [
        ["Serial Number:", serial_number],
        ["Inspection Date:", timestamp],
        ["Status:", "PASS" if serial_number != "UNKNOWN" else "MANUAL REVIEW REQUIRED"],
        ["Location:", "Line 01 - Final Assembly"]
    ]
    t_meta = Table(meta_data, colWidths=[1.5*inch, 4*inch])
    t_meta.setStyle(TableStyle([
        ('FONTNAME', (0,0), (-1,-1), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 10),
    ]))
    story.append(t_meta)
    story.append(Spacer(1, 30))
    
    # Lead time table
    if seq_times:
        story.append(Paragraph("Sequence Lead Times", h1_style))
        lt_rows = [["Sequence Step", "Time Taken", "Status"]]
        seq_names = {1: "SEQ 1: Top Surface", 2: "SEQ 2: Middle Region", 3: "SEQ 3: Bottom Region"}
        row_colors = []  # track per-row background for color coding
        for i in (1, 2, 3):
            t = seq_times.get(i)
            # t == None means the sequence was not reached (missed)
            # t == 0.0  means completed but start time unavailable
            # t  > 0    means completed with measured duration
            if t is None:
                t_str  = "—"
                status = "✕ Missed"
                row_colors.append(colors.HexColor("#ffebee"))  # light red
            else:
                if t > 0:
                    m, s = divmod(int(t), 60)
                    t_str = f"{m}m {s}s" if m else f"{s}s"
                else:
                    t_str = "< 1s"   # 0.0 = completed but timing not recorded
                status = "✓ Completed"
                row_colors.append(colors.HexColor("#e8f5e9"))  # light green
            lt_rows.append([seq_names[i], t_str, status])
        
        t_lt = Table(lt_rows, colWidths=[2.5*inch, 1.5*inch, 1.5*inch])
        tbl_style = [
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#1a237e")),
            ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0,0), (-1,-1), 8),
        ]
        # Apply per-row background colors (data rows start at index 1)
        for row_idx, bg in enumerate(row_colors):
            tbl_style.append(('BACKGROUND', (0, row_idx+1), (-1, row_idx+1), bg))
        t_lt.setStyle(TableStyle(tbl_style))
        story.append(t_lt)
        
    story.append(Spacer(1, 20))
    m, s = divmod(int(total_time), 60)
    tt_str = f"{m}m {s}s" if m else f"{s}s"
    story.append(Paragraph(f"<b>Total Inspection Lead Time:</b> {tt_str}", sub_style))
    story.append(PageBreak())

    # ── PAGE 2: SEQ 1 FULL ───────────────────────────────────
    story.append(Paragraph("Sequence 1 — Full Panel Image", h1_style))
    files = _find_best(panel_folder, "*_sequence_1*.jpg", "SSP-SEQ_sequence_1*.jpg", "*_Seq1_Full_*.jpg", "SSP-SEQ_Seq1_Full_*.jpg", "*Seq1*Full*.jpg")
    if files:
        story.append(_fit_image(files[0], USABLE_W, 7*inch))
        print(f"  [PDF] SEQ1: {os.path.basename(files[0])}")
    else:
        story.append(Paragraph("<i>SEQ1 full image not captured.</i>", sub_style))
        print("  [PDF] WARNING: SEQ1 full image MISSING")
    story.append(PageBreak())


    # ── PAGE 4: SEQ 2 FULL ───────────────────────────────────
    story.append(Paragraph("Sequence 2 — Full Panel Image", h1_style))
    files = _find_best(panel_folder, "*_sequence_2*.jpg", "SSP-SEQ_sequence_2*.jpg", "*_Seq2_Full_*.jpg", "SSP-SEQ_Seq2_Full_*.jpg", "*Seq2*Full*.jpg")
    if files:
        story.append(_fit_image(files[0], USABLE_W, 7*inch))
        print(f"  [PDF] SEQ2: {os.path.basename(files[0])}")
    else:
        story.append(Paragraph("<i>SEQ2 full image not captured.</i>", sub_style))
        print("  [PDF] WARNING: SEQ2 full image MISSING")
    story.append(PageBreak())

    # ── PAGE 5: SEQ 3 FULL ───────────────────────────────────
    story.append(Paragraph("Sequence 3 — Full Panel Image", h1_style))
    files = _find_best(panel_folder, "*_sequence_3*.jpg", "SSP-SEQ_sequence_3*.jpg", "*_Seq3_Full_*.jpg", "SSP-SEQ_Seq3_Full_*.jpg", "*Seq3*Full*.jpg")
    if files:
        story.append(_fit_image(files[0], USABLE_W, 7*inch))
        print(f"  [PDF] SEQ3: {os.path.basename(files[0])}")
    else:
        story.append(Paragraph("<i>SEQ3 full image not captured.</i>", sub_style))
        print("  [PDF] WARNING: SEQ3 full image MISSING")
    story.append(PageBreak())

    # ── PAGE 5: SERIAL NUMBER FULL BEST PICTURE ──────────────
    story.append(Paragraph("Serial Number — Full Best Picture", h1_style))
    story.append(Spacer(1, 10))
    story.append(Paragraph(
        f"<b>Confirmed Serial Number:</b>  {serial_number}",
        ParagraphStyle('SerialLabel', parent=sub_style,
                       fontSize=14, textColor=colors.HexColor("#1a237e"))))
    story.append(Spacer(1, 15))

    serial_cap_dir = os.path.join(panel_folder, "serial_captures")

    # Search for the single best full picture of serial
    full_serial_candidates = (
        _find_best(serial_cap_dir, "CAM2_Best_Full.jpg", "*Best_Full*.jpg", "CAM2_Full_Frame_*.jpg") or
        _find_best(panel_folder, "CAM2_Best_Full.jpg", "*Serial_Original*.jpg", "*Serial*Best_Full*.jpg", "CAM2_Serial_Appeared_Best*.jpg") or
        ([cam2_image] if cam2_image and os.path.exists(cam2_image) else []) or
        _find_best(serial_cap_dir, "CAM2_Crop_1.jpg", "CAM2_Crop_*.jpg")
    )

    if full_serial_candidates and os.path.exists(full_serial_candidates[0]):
        best_pic = full_serial_candidates[0]
        story.append(_fit_image(best_pic, USABLE_W, 6.8*inch))
        print(f"  [PDF] Serial best picture: {os.path.basename(best_pic)}")
    else:
        story.append(Paragraph("<i>Serial picture not available.</i>", sub_style))
        print("  [PDF] WARNING: Serial picture MISSING")

    # Final build
    try:
        doc.build(story)
        page_count = getattr(doc, 'page', 5)
        print(f"✅ PDF successfully generated: {report_path} (Pages: {page_count})")
        gc.collect()  # Free memory
        return (report_path, page_count)
    except Exception as e:
        print(f"❌ PDF generation failed: {e}")
        traceback.print_exc()
        return (None, 0)
