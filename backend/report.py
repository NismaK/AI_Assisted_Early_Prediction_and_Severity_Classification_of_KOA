from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
import datetime

def generate_report(patient_data: dict, result: dict) -> str:
    filename = f"report_{patient_data['name'].replace(' ', '_')}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    filepath = f"reports/{filename}"
    
    import os
    os.makedirs("reports", exist_ok=True)
    
    doc = SimpleDocTemplate(filepath, pagesize=A4,
                           rightMargin=2*cm, leftMargin=2*cm,
                           topMargin=2*cm, bottomMargin=2*cm)
    
    styles = getSampleStyleSheet()
    story = []
    
    # Header
    header_style = ParagraphStyle('header', fontSize=20, 
                                  textColor=colors.HexColor('#185FA5'),
                                  spaceAfter=6, alignment=TA_CENTER,
                                  fontName='Helvetica-Bold')
    sub_style = ParagraphStyle('sub', fontSize=11,
                               textColor=colors.HexColor('#6B7280'),
                               spaceAfter=4, alignment=TA_CENTER)
    
    story.append(Paragraph("Knee OA Analyzer", header_style))
    story.append(Paragraph("Osteoarthritis Severity Report", sub_style))
    story.append(Paragraph(
        f"Date: {datetime.datetime.now().strftime('%d %B %Y, %I:%M %p')}",
        sub_style))
    story.append(HRFlowable(width="100%", thickness=1,
                            color=colors.HexColor('#185FA5'), spaceAfter=16))

    # Patient Details
    section_style = ParagraphStyle('section', fontSize=13,
                                   textColor=colors.HexColor('#185FA5'),
                                   spaceBefore=10, spaceAfter=8,
                                   fontName='Helvetica-Bold')
    story.append(Paragraph("Patient Information", section_style))
    
    patient_data_table = [
        ['Patient Name:', patient_data['name']],
        ['Age:', f"{patient_data['age']} years"],
        ['Gender:', patient_data['gender']],
    ]
    
    t = Table(patient_data_table, colWidths=[5*cm, 12*cm])
    t.setStyle(TableStyle([
        ('FONTNAME', (0,0), (0,-1), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 11),
        ('TEXTCOLOR', (0,0), (0,-1), colors.HexColor('#374151')),
        ('TEXTCOLOR', (1,0), (1,-1), colors.HexColor('#111827')),
        ('ROWBACKGROUNDS', (0,0), (-1,-1),
         [colors.HexColor('#F9FAFB'), colors.white]),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(t)
    story.append(Spacer(1, 16))

    # Grade Result
    story.append(HRFlowable(width="100%", thickness=0.5,
                            color=colors.HexColor('#E5E7EB'), spaceAfter=12))
    story.append(Paragraph("Diagnosis Result", section_style))
    
    grade_colors_map = {
        0: '#1D9E75', 1: '#639922', 2: '#BA7517',
        3: '#D85A30', 4: '#A32D2D'
    }
    grade = result['grade']
    grade_color = grade_colors_map.get(grade, '#185FA5')
    
    grade_style = ParagraphStyle('grade', fontSize=28,
                                 textColor=colors.HexColor(grade_color),
                                 alignment=TA_CENTER, fontName='Helvetica-Bold',
                                 spaceAfter=4)
    label_style = ParagraphStyle('label', fontSize=16,
                                 textColor=colors.HexColor(grade_color),
                                 alignment=TA_CENTER, spaceAfter=4)
    conf_style = ParagraphStyle('conf', fontSize=13,
                                textColor=colors.HexColor('#6B7280'),
                                alignment=TA_CENTER, spaceAfter=16)
    
    story.append(Spacer(1, 8))
    story.append(Paragraph(f"KL Grade {grade}", grade_style))
    story.append(Spacer(1, 10))
    story.append(Paragraph(result['label'], label_style))
    story.append(Spacer(1, 8))
    story.append(Paragraph(f"Confidence: {result['confidence']}%", conf_style))
    story.append(Spacer(1, 8))
    
    # Findings
    story.append(Paragraph("Radiological Findings", section_style))
    for finding in result['findings']:
        finding_style = ParagraphStyle('finding', fontSize=11,
                                       textColor=colors.HexColor('#374151'),
                                       spaceAfter=6, leftIndent=12)
        story.append(Paragraph(f"• {finding}", finding_style))
    
    story.append(Spacer(1, 16))

    # Probability Table
    story.append(Paragraph("Grade-wise Probability", section_style))
    
    prob_data = [['Grade', 'Description', 'Probability']]
    descriptions = ['Normal', 'Doubtful', 'Mild OA', 'Moderate OA', 'Severe OA']
    for i, desc in enumerate(descriptions):
        prob = result['all_probabilities'].get(f'Grade {i}', 0)
        prob_data.append([f'Grade {i}', desc, f'{prob}%'])
    
    prob_table = Table(prob_data, colWidths=[4*cm, 8*cm, 5*cm])
    prob_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#185FA5')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 11),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('ROWBACKGROUNDS', (0,1), (-1,-1),
         [colors.HexColor('#F9FAFB'), colors.white]),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#E5E7EB')),
        ('TOPPADDING', (0,0), (-1,-1), 8),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(prob_table)
    story.append(Spacer(1, 24))

    # Disclaimer
    disclaimer_style = ParagraphStyle('disclaimer', fontSize=9,
                                      textColor=colors.HexColor('#9CA3AF'),
                                      alignment=TA_CENTER)
    story.append(HRFlowable(width="100%", thickness=0.5,
                            color=colors.HexColor('#E5E7EB'), spaceAfter=8))
    story.append(Paragraph(
        "⚕ Yeh report sirf educational aur decision-support purpose ke liye hai. "
        "Final diagnosis ek qualified doctor ya radiologist dwara ki jani chahiye.",
        disclaimer_style))
    
    doc.build(story)
    return filepath