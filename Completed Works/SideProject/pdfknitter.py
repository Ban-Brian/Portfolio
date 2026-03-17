from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, ListFlowable, ListItem, Table, TableStyle
import re
from resumeBrianv2 import brian_butler_data  # Your resume dictionary

def create_pdf(output_file):
    doc = SimpleDocTemplate(
        output_file,
        pagesize=letter,
        rightMargin=40, leftMargin=40,
        topMargin=30, bottomMargin=30
    )

    styles = getSampleStyleSheet()

    # =========================
    # Styles
    # =========================
    name_style = ParagraphStyle(
        'Name',
        parent=styles['Title'],
        fontSize=22,
        leading=26,
        alignment=1,
        spaceAfter=4,
        textColor=colors.darkblue
    )

    contact_style = ParagraphStyle(
        'Contact',
        parent=styles['Normal'],
        fontSize=9.5,
        leading=11,
        alignment=1,
        spaceAfter=12
    )

    section_style = ParagraphStyle(
        'Section',
        parent=styles['Heading2'],
        fontSize=11,
        fontName='Helvetica-Bold',
        spaceBefore=12,
        spaceAfter=6,
        textColor=colors.black
    )

    sub_section_style = ParagraphStyle(
        'SubSection',
        parent=styles['Normal'],
        fontSize=10,
        fontName='Helvetica-Bold',
        spaceAfter=2
    )

    content_style = ParagraphStyle(
        'Content',
        parent=styles['Normal'],
        fontSize=9.5,
        leading=12,
        spaceAfter=2
    )

    def bold_numbers(text):
        return re.sub(r'(\d+)', r'<b>\1</b>', text)

    story = []

    # =========================
    # Header
    # =========================
    story.append(Paragraph(brian_butler_data["name"], name_style))
    story.append(Paragraph(brian_butler_data["headline"], contact_style))

    contact = brian_butler_data["contact"]
    contact_info = f"{contact['location']} | {contact['email']} | {contact['phone']}"
    story.append(Paragraph(contact_info, contact_style))

    websites = f'<link href="{contact["linkedin"]}">LinkedIn</link>'
    story.append(Paragraph(websites, contact_style))

    # =========================
    # Education
    # =========================
    story.append(Paragraph("EDUCATION", section_style))
    edu = brian_butler_data["education"]
    education_text = (
        f"<b>{edu['university']}</b><br/>"
        f"<b>{edu['degree']}</b><br/>{edu['details']}"
    )
    story.append(Paragraph(education_text, content_style))

    # =========================
    # Optional: You can add Work, Projects, etc.
    # Only include if these sections exist in your data
    # =========================
    if "work_experience" in brian_butler_data:
        story.append(Paragraph("WORK EXPERIENCE", section_style))
        for job in brian_butler_data["work_experience"]:
            job_header = f"<b>{job['position']}</b>, <b>{job['company']}</b> | {job['location']} | <b>{job['dates']}</b>"
            story.append(Paragraph(job_header, sub_section_style))
            bullets = [ListItem(Paragraph(bold_numbers(resp), content_style), leftIndent=12)
                       for resp in job["responsibilities"]]
            story.append(ListFlowable(bullets, bulletType='bullet'))
            story.append(Spacer(1, 4))

    # =========================
    # Build PDF
    # =========================
    doc.build(story)
    print("PDF created in McCombs style!")

if __name__ == "__main__":
    create_pdf("resume_mccombs.pdf")