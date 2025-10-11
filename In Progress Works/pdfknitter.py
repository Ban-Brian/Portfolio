from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, ListFlowable, ListItem, Table, TableStyle
import re
from reesume import resume  # Your resume dictionary

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
    story.append(Paragraph(resume["Name"], name_style))
    contact_info = f"{resume['Location']} | {resume['Contact']['Email']} | {resume['Contact']['Phone']}"
    story.append(Paragraph(contact_info, contact_style))
    websites = (
        f'<link href="{resume["Websites"]["LinkedIn"]}">LinkedIn</link> | '
        f'<link href="{resume["Websites"]["GitHub"]}">GitHub</link> | '
        f'<link href="{resume["Websites"]["LeetCode"]}">LeetCode</link>'
    )
    story.append(Paragraph(websites, contact_style))

    # =========================
    # Education
    # =========================
    story.append(Paragraph("EDUCATION", section_style))
    education = (
        f"<b>{resume['Education']['Degree']}</b>, Minor: <b>{resume['Education']['Minor']}</b><br/>"
        f"<b>{resume['Education']['Institution']}</b> | Expected Graduation: <b>{resume['Education']['Expected Graduation']}</b><br/>"
        f"<b>Relevant Coursework:</b> {', '.join(resume['Education']['Relevant Coursework'])}"
    )
    story.append(Paragraph(education, content_style))

    # =========================
    # Work Experience
    # =========================
    story.append(Paragraph("WORK EXPERIENCE", section_style))
    for job in resume["Work Experience"]:
        # Bold job title and company, align dates right
        job_header = f"<b>{job['Position']}</b>, <b>{job['Company']}</b> | {job['Location']} | <b>{job['Dates']}</b>"
        story.append(Paragraph(job_header, sub_section_style))
        # Bullets with numbers bolded
        bullets = [ListItem(Paragraph(bold_numbers(resp), content_style), leftIndent=12) for resp in job["Responsibilities"]]
        story.append(ListFlowable(bullets, bulletType='bullet'))
        story.append(Spacer(1, 4))

    # =========================
    # Projects
    # =========================
    story.append(Paragraph("PROJECTS", section_style))
    for project in resume["Projects"]:
        proj_header = f"<b>{project['Title']}</b> | <i>{project['Dates']}</i>"
        story.append(Paragraph(proj_header, sub_section_style))
        story.append(Paragraph(bold_numbers(project['Description']), content_style))
        story.append(Spacer(1, 2))

    # =========================
    # Skills & Memberships (side-by-side columns)
    # =========================
    story.append(Spacer(1, 6))
    skills_content = Paragraph('<b>Technical Skills:</b><br/>' + ' | '.join(resume['Skills']), content_style)
    memberships_content = Paragraph('<b>Memberships:</b><br/>' + '<br/>'.join(resume['Memberships']), content_style)
    table = Table([[skills_content, memberships_content]], colWidths=[doc.width/2-5, doc.width/2-5])
    table.setStyle(TableStyle([('VALIGN', (0, 0), (-1, -1), 'TOP')]))
    story.append(table)

    # =========================
    # Build PDF
    # =========================
    doc.build(story)
    print("PDF created in McCombs style!")

if __name__ == "__main__":
    create_pdf("resume_mccombs.pdf")