from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable, Table, TableStyle, ListFlowable, ListItem
from reesume import resume

def create_pdf(output_file):
    doc = SimpleDocTemplate(output_file, pagesize=letter, rightMargin=72, leftMargin=72, topMargin=24, bottomMargin=12)
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        name='TitleStyle',
        parent=styles['Title'],
        fontName='Times-Roman',
        fontSize=16.5,
        spaceAfter=3,
        backColor=colors.lightgrey,
        wordWrap='CJK'
    )
    section_style = ParagraphStyle(
        name='SectionStyle',
        parent=styles['Heading2'],
        fontName='Times-Roman',
        fontSize=10.5,
        spaceAfter=1,
        textColor=colors.darkblue,
        wordWrap='CJK',
        alignment=1  # Center-align the section titles
    )
    content_style = ParagraphStyle(
        name='ContentStyle',
        parent=styles['Normal'],
        fontName='Times-Roman',
        fontSize=8.5,
        leading=9.5,
        spaceAfter=1,
        textColor=colors.black,
        wordWrap='CJK'
    )

    story = []

    title = Paragraph(resume["Name"], title_style)
    story.append(title)
    story.append(HRFlowable(width=doc.width, thickness=1, color=colors.black))

    def add_section(title, content):
        story.append(Paragraph(title, section_style))
        story.append(Spacer(1, 3))
        story.append(Paragraph(content, content_style))
        story.append(Spacer(1, 1))
        story.append(HRFlowable(width=doc.width, thickness=1, color=colors.black))

    add_section("Projects", resume["Projects"])

    contact_info = f"Location: {resume['Location']}<br/>Email: {resume['Contact']['Email']}<br/>Phone: {resume['Contact']['Phone']}"
    skills = "<br/>".join(resume["Skills"])

    data = [
        [Paragraph("Contact Information", section_style), Paragraph("Skills", section_style)],
        [Paragraph(contact_info, content_style), Paragraph(skills, content_style)]
    ]
    table = Table(data, colWidths=[doc.width / 2.0] * 2)
    table.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('LEFTPADDING', (0, 0), (-1, -1), 0),
        ('RIGHTPADDING', (0, 0), (-1, -1), 0),
    ]))
    story.append(table)
    story.append(Spacer(1, 1))
    story.append(HRFlowable(width=doc.width, thickness=1, color=colors.black))

    education = (
        f"<b>{resume['Education']['Degree']}</b><br/>"
        f"<b>Minor:</b> {resume['Education']['Minor']}<br/>"
        f"{resume['Education']['Institution']}<br/>"
        f"<b>Expected Graduation:</b> {resume['Education']['Expected Graduation']}<br/>"
        f"<b>GPA:</b> {resume['Education']['GPA']}<br/>"
        f"<b>Relevant Coursework:</b> {', '.join(resume['Education']['Relevant Coursework'])}"
    )
    websites = (
        f'<a href="{resume["Websites"]["LinkedIn"]}">LinkedIn</a><br/>'
        f'<a href="{resume["Websites"]["GitHub"]}">GitHub</a><br/>'
        f'<a href="{resume["Websites"]["LeetCode"]}">LeetCode</a>'
    )

    data = [
        [Paragraph("Education", section_style), Paragraph("Websites, Portfolios, Profiles", section_style)],
        [Paragraph(education, content_style), Paragraph(websites, content_style)]
    ]
    table = Table(data, colWidths=[doc.width / 2.0] * 2)
    table.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('LEFTPADDING', (0, 0), (-1, -1), 0),
        ('RIGHTPADDING', (0, 0), (-1, -1), 0),
    ]))
    story.append(table)
    story.append(Spacer(1, 1))
    story.append(HRFlowable(width=doc.width, thickness=1, color=colors.black))

    add_section("Work Experience", "")

    for job in resume["Work History"]:
        job_title = f"<b>{job['Position']}</b><br/>{job['Company']} - {job['Location']}<br/>{job['Dates']}"
        responsibilities = [
            ListItem(Paragraph(responsibility, content_style), leftIndent=10, bulletIndent=0)
            for responsibility in job["Responsibilities"]
        ]
        job_data = [
            [Paragraph(job_title, content_style)],
            [ListFlowable(responsibilities, bulletType='bullet')]
        ]
        job_table = Table(job_data, colWidths=[doc.width])
        job_table.setStyle(TableStyle([
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('LEFTPADDING', (0, 0), (-1, -1), 0),
            ('RIGHTPADDING', (0, 0), (-1, -1), 0),
        ]))
        story.append(job_table)
        story.append(Spacer(1, 3))

    story.append(HRFlowable(width=doc.width, thickness=1, color=colors.black))
    memberships = "<br/>".join(resume["Memberships"])
    add_section("Memberships", memberships)

    doc.build(story)

output_file = "resume.pdf"
create_pdf(output_file)