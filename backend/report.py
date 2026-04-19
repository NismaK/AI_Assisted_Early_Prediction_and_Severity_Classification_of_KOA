from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
import datetime
import base64
import tempfile
import os

def generate_report(patient_data: dict, result: dict) -> str:
    filename = f"report_{patient_data['name'].replace(' ', '_')}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    filepath = f"reports/{filename}"
    os.makedirs("reports", exist_ok=True)

    doc = SimpleDocTemplate(
        filepath, pagesize=A4,
        rightMargin=1.5*cm, leftMargin=1.5*cm,
        topMargin=1.2*cm, bottomMargin=1.2*cm
    )

    story = []

    # ── Styles ────────────────────────────────────────────────
    header_style = ParagraphStyle('header', fontSize=13,
        textColor=colors.HexColor('#185FA5'),
        spaceAfter=2, alignment=TA_CENTER, fontName='Helvetica-Bold')

    sub_style = ParagraphStyle('sub', fontSize=9,
        textColor=colors.HexColor('#6B7280'),
        spaceAfter=2, alignment=TA_CENTER)

    section_style = ParagraphStyle('section', fontSize=10,
        textColor=colors.HexColor('#185FA5'),
        spaceBefore=6, spaceAfter=4, fontName='Helvetica-Bold')

    grade_style = ParagraphStyle('grade', fontSize=14,
        textColor=colors.HexColor('#185FA5'),
        alignment=TA_CENTER, fontName='Helvetica-Bold', spaceAfter=2)

    label_style = ParagraphStyle('label', fontSize=10,
        textColor=colors.HexColor('#185FA5'),
        alignment=TA_CENTER, spaceAfter=2)

    conf_style = ParagraphStyle('conf', fontSize=9,
        textColor=colors.HexColor('#6B7280'),
        alignment=TA_CENTER, spaceAfter=6)

    finding_style = ParagraphStyle('finding', fontSize=9,
        textColor=colors.HexColor('#374151'),
        spaceAfter=3, leftIndent=10)

    disclaimer_style = ParagraphStyle('disclaimer', fontSize=8,
        textColor=colors.HexColor('#9CA3AF'), alignment=TA_CENTER)

    border = {'style': 'SINGLE', 'size': 0.5, 'color': 'CCCCCC'}

    # ── Header ────────────────────────────────────────────────
    story.append(Paragraph("Knee OA Analyzer", header_style))
    story.append(Paragraph("Osteoarthritis Severity Report", sub_style))
    story.append(Paragraph(
        f"Date: {datetime.datetime.now().strftime('%d %B %Y, %I:%M %p')}",
        sub_style))
    story.append(HRFlowable(width="100%", thickness=0.5,
        color=colors.HexColor('#185FA5'), spaceAfter=6))

    # ── Patient Info ──────────────────────────────────────────
    story.append(Paragraph("Patient Information", section_style))

    border_style = {'style': 'SINGLE', 'size': 0.5, 'color': 'CCCCCC'}
    b = colors.HexColor('#CCCCCC')
    single = ('GRID', (0,0), (-1,-1), 0.5, b)

    patient_table = Table([
        ['Patient Name:', patient_data['name']],
        ['Age:', f"{patient_data['age']} years"],
        ['Gender:', patient_data['gender']],
    ], colWidths=[4*cm, 13*cm])

    patient_table.setStyle(TableStyle([
        single,
        ('FONTNAME', (0,0), (0,-1), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 9),
        ('ROWBACKGROUNDS', (0,0), (-1,-1),
         [colors.HexColor('#F9FAFB'), colors.white]),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(patient_table)
    story.append(Spacer(1, 6))

    # ── Grade Result ──────────────────────────────────────────
    story.append(HRFlowable(width="100%", thickness=0.5,
        color=colors.HexColor('#E5E7EB'), spaceAfter=4))
    story.append(Paragraph("Diagnosis Result", section_style))

    grade_colors_map = {
        0: '#1D9E75', 1: '#639922', 2: '#BA7517',
        3: '#D85A30', 4: '#A32D2D'
    }
    grade = result['grade']
    gc = grade_colors_map.get(grade, '#185FA5')

    grade_style.textColor = colors.HexColor(gc)
    label_style.textColor = colors.HexColor(gc)

    story.append(Paragraph(f"KL Grade {grade}", grade_style))
    story.append(Paragraph(result['label'], label_style))
    story.append(Paragraph(
        f"Confidence: {result['confidence']}%", conf_style))

    # ── Findings ──────────────────────────────────────────────
    story.append(Paragraph("Radiological Findings", section_style))
    for finding in result['findings']:
        story.append(Paragraph(f"• {finding}", finding_style))
    story.append(Spacer(1, 6))

    # ── Grad-CAM + Probability side by side ───────────────────
    story.append(Paragraph("Analysis Details", section_style))

    # Probability table
    prob_data = [['Grade', 'Description', 'Probability']]
    descriptions = ['Normal', 'Doubtful', 'Mild OA', 'Moderate OA', 'Severe OA']
    for i, desc in enumerate(descriptions):
        prob = result['all_probabilities'].get(f'Grade {i}', 0)
        prob_data.append([f'Grade {i}', desc, f'{prob}%'])

    prob_table = Table(prob_data, colWidths=[2.5*cm, 5.5*cm, 3*cm],
                      hAlign='CENTER')
    prob_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#185FA5')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 8),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('ROWBACKGROUNDS', (0,1), (-1,-1),
         [colors.HexColor('#F9FAFB'), colors.white]),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#E5E7EB')),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))

    # Grad-CAM image
    if result.get('gradcam_image'):
        img_data = base64.b64decode(result['gradcam_image'])
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            tmp.write(img_data)
            tmp_path = tmp.name

        gradcam_img = RLImage(tmp_path, width=6*cm, height=6*cm)

        story.append(prob_table)
        story.append(Spacer(1, 6))
        story.append(Paragraph("Grad-CAM Visualization", section_style))
        story.append(Spacer(1, 4))
        gradcam_table = Table([[gradcam_img]], colWidths=[17*cm])
        gradcam_table.setStyle(TableStyle([
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('LEFTPADDING', (0,0), (-1,-1), 0),
            ('RIGHTPADDING', (0,0), (-1,-1), 0),
        ]))
        story.append(gradcam_table)
    else:
        story.append(prob_table)

    story.append(Spacer(1, 8))

    # ── Disclaimer ────────────────────────────────────────────
    story.append(HRFlowable(width="100%", thickness=0.5,
        color=colors.HexColor('#E5E7EB'), spaceAfter=4))
    story.append(Paragraph(
        "For educational and decision-support purposes only. "
        "Final diagnosis must be made by a qualified physician.",
        disclaimer_style))

    doc.build(story)
    return filepath