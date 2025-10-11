import re
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, ListFlowable, ListItem, Table, TableStyle
from reportlab.lib.units import inch

brian_butler_data = {
    "name": "Brian Butler",
    "headline": "Aspiring Data Analyst",
    "contact": {
        "location": "Fairfax, VA | San Diego, CA",
        "linkedin": "https://www.linkedin.com/in/brian-butler-18036b33b/",
        "email": "butlerbrian67@gmail.com",
        "phone": "619-866-5916"
    },
    "about": "A self-taught and social problem solver who can efficiently integrate new ideas and is emboldened about discovering effective courses of action through data analysis. Actively seeking opportunities related to data analysis, especially in the tech, consulting, and research industries.",
    "education": {
        "university": "George Mason University",
        "degree": "Bachelor of Applied Statistics",
        "details": "Deans List : 2023 -2025 | Minor in Criminology and Math | GPA: 3.5/4.0"
    },
    "experience": [

        {
            "company": "LEAP (Leadership Education For Asian Pacifics)",
            "role": "Fellow",
            "location": "Remote",
            "dates": "Aug 2025 - Present",
            "description": [
                "Develop leadership skills through workshops, training, and hands-on opportunities designed to equip students with practical tools for success.",
                "Gain exposure to diverse industries by connecting with professionals, exploring career pathways, and broadening post-graduate possibilities.",
                "Receive personalized mentorship and support tailored to each fellow’s goals, ensuring guidance and preparation for life after college."
            ]
        },
        {
            "company": "George Mason University",
            "role": "Researcher with Early Justice Strategies Lab",
            "location": "Fairfax County, VA",
            "dates": "Feb 2025 - Present",
            "description": [
                "Analyzed results using analytical software and created reports for strategic review.",
                "Completed A/B testing and data analysis to improve research outcomes and methodologies.",
                "Used Excel to extract data from databases and perform data analysis for better actionable insights."
            ]
        },
        {
            "company": "American Samoa Government",
            "role": "Summer Intern for Department of Commerce",
            "location": "Pago Pago, American Samoa",
            "dates": "May 2025 - Aug 2025",
            "description": [
                "Applied machine learning in Python to identify key patterns and optimize the organization of the Statistical Yearbook.",
                "Developed a multivariate predictive model in Python and Excel to forecast cost-of-living indices, applying regression and time series forecasting.",
                "Conducted cost analysis and feasibility modeling in Excel for the Pago Pago Sky Tram project, evaluating a projected $35 million capital investment."
            ]
        },
        {
            "company": "Health-Link Society",
            "role": "Head Chapter Development Manager",
            "location": "Washington DC-Baltimore Area",
            "dates": "Dec 2024 - May 2025",
            "description": [
                "Identified opportunities for new chapter development by conducting demographic and statistical research.",
                "Allocated financial support to chapters through fundraising and grant acquisitions, resulting in a 30% increase in chapter funding.",
                "Engaged with prospective chapters and forged partnerships, contributing to a 20% growth in chapter membership."
            ]
        },
        {
            "company": "Nonprofit Alliance Against Domestic & Sexual Violence",
            "role": "Research Intern",
            "location": "Remote",
            "dates": "Jun 2024 - Aug 2024",
            "description": [
                "Utilized Python and Excel to analyze survey data, identifying resource gaps and informing a report on underserved populations.",
                "Conducted a statistical evaluation of a pilot intervention program using SPSS, demonstrating a 25% improvement in client outcomes.",
                "Designed data visualizations in Tableau for grant proposals, contributing to securing funding for two new initiatives."
            ]
        }
    ],
    "certifications": [
        {
            "title": "Google Analytics Certification",
            "issuer": "Google Skillshop",
            "date": "Issued Jun 2025"
        }
    ],
    "skills": {
        "Industry Knowledge": "Business Analytics, Data Visualization, Data Analysis, A/B Testing, Statistical Modeling",
        "Languages & Software": "Python (Pandas, Scikit-learn), SQL, R, Tableau, Power BI, Excel, SPSS, Git"
    }
}


def create_resume_pdf(output_filename):
    """
    Generates a compact, single-page PDF resume.
    """
    doc = SimpleDocTemplate(
        output_filename,
        pagesize=letter,
        leftMargin=0.6 * inch,
        rightMargin=0.6 * inch,
        topMargin=0.4 * inch,
        bottomMargin=0.4 * inch
    )

    styles = getSampleStyleSheet()
    story = []

    name_style = ParagraphStyle(
        'Name',
        parent=styles['h1'],
        fontName='Helvetica-Bold',
        fontSize=19,  # TWEAK: Reduced from 22
        alignment=1,
        spaceAfter=1  # TWEAK: Reduced from 2
    )
    contact_style = ParagraphStyle(
        'Contact',
        parent=styles['Normal'],
        fontSize=9,  # TWEAK: Reduced from 9.5
        alignment=1,
        spaceAfter=8,  # TWEAK: Reduced from 12
        textColor=colors.darkgrey
    )
    section_title_style = ParagraphStyle(
        'SectionTitle',
        parent=styles['h2'],
        fontName='Helvetica-Bold',
        fontSize=10,
        leading=12,
        textColor=colors.black
    )
    body_style = ParagraphStyle(
        'Body',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,  # TWEAK: Reduced from 9.5
        leading=11,  # TWEAK: Reduced from 14 for tighter line spacing
        spaceAfter=2  # TWEAK: Reduced from 6
    )

    def add_section_header(title):
        p = Paragraph(title.upper(), section_title_style)
        table = Table([[p]], colWidths=[doc.width])
        table.setStyle(TableStyle([
            ('LINEBELOW', (0, 0), (-1, -1), 1, colors.black),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 2),  # TWEAK: Reduced from 3
            ('TOPPADDING', (0, 0), (-1, -1), 8),  # TWEAK: Reduced from 10
        ]))
        story.append(table)
        story.append(Spacer(1, 2))  # TWEAK: Reduced from 4


    story.append(Paragraph(brian_butler_data['name'], name_style))
    contact_info = (
        f"{brian_butler_data['contact']['location']} &nbsp;&nbsp;|&nbsp;&nbsp; "
        f"{brian_butler_data['contact']['phone']} &nbsp;&nbsp;|&nbsp;&nbsp; "
        f"{brian_butler_data['contact']['email']} &nbsp;&nbsp;|&nbsp;&nbsp; "
        f"{brian_butler_data['contact']['linkedin']}"
    )
    story.append(Paragraph(contact_info, contact_style))

    add_section_header("About")
    story.append(Paragraph(brian_butler_data['about'], body_style))

    add_section_header("Education")
    edu = brian_butler_data['education']
    edu_text = f"<b>{edu['university']}</b><br/>{edu['degree']}<br/><i>{edu['details']}</i>"
    story.append(Paragraph(edu_text, body_style))

    add_section_header("Experience")
    for job in brian_butler_data['experience']:
        left_col = f"<b>{job['role']}</b>, <i>{job['company']}</i>, {job['location']}"
        right_col = f"<b>{job['dates']}</b>"

        table = Table(
            [[Paragraph(left_col, body_style), Paragraph(right_col, body_style)]],
            colWidths=[doc.width * 0.75, doc.width * 0.25]  # Adjusted column widths for better balance
        )
        table.setStyle(TableStyle([('ALIGN', (1, 0), (1, 0), 'RIGHT'), ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                                   ('BOTTOMPADDING', (0, 0), (-1, -1), 1)]))
        story.append(table)

        bullets = [ListItem(Paragraph(item, body_style), leftIndent=15, bulletIndent=8) for item in job['description']]
        story.append(ListFlowable(bullets, bulletType='bullet', start='•'))
        story.append(Spacer(1, 4))  # TWEAK: Reduced from 8

    add_section_header("Licenses & Certifications")
    for cert in brian_butler_data['certifications']:
        left_col = f"<b>{cert['title']}</b> - <i>{cert['issuer']}</i>"
        right_col = f"<b>{cert['date']}</b>"
        table = Table(
            [[Paragraph(left_col, body_style), Paragraph(right_col, body_style)]],
            colWidths=[doc.width * 0.75, doc.width * 0.25]
        )
        table.setStyle(TableStyle([('ALIGN', (1, 0), (1, 0), 'RIGHT')]))
        story.append(table)

    add_section_header("Skills")
    skills_text = ""
    for key, value in brian_butler_data['skills'].items():
        skills_text += f"<b>{key}:</b> {value}<br/>"
    story.append(Paragraph(skills_text, body_style))

    # --- Build PDF ---
    doc.build(story)
    print(f"PDF '{output_filename}' created successfully.")


if __name__ == "__main__":
    create_resume_pdf("Brian_Butler_Resume_Compact.pdf")