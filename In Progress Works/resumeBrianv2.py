from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, ListFlowable, ListItem, Table, TableStyle
from reportlab.lib.units import inch


def create_resume_pdf(output_filename):
    """
    Generates a visually full, one-page resume for Brian Butler that fills the page and includes clickable blue links.
    """
    doc = SimpleDocTemplate(
        output_filename,
        pagesize=letter,
        leftMargin=0.6 * inch,
        rightMargin=0.6 * inch,
        topMargin=0.5 * inch,
        bottomMargin=0.5 * inch
    )

    styles = getSampleStyleSheet()

    # --- TEXT STYLES ---
    name_style = ParagraphStyle(
        'Name',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=17,
        alignment=1,
        spaceAfter=6
    )

    contact_style = ParagraphStyle(
        'Contact',
        parent=styles['Normal'],
        fontSize=9.5,
        alignment=1,
        spaceAfter=8
    )

    section_header = ParagraphStyle(
        'SectionHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=11,
        textColor=colors.black,
        spaceBefore=10,
        spaceAfter=4
    )

    body_style = ParagraphStyle(
        'Body',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=11.5,
        spaceAfter=2
    )

    link_style = ParagraphStyle(
        'Link',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        textColor=colors.blue,
        alignment=1,
        leading=11,
        spaceBefore=6
    )

    story = []

    # --- HEADER ---
    story.append(Paragraph("Brian Butler", name_style))
    story.append(Paragraph("Fairfax, VA 22030 | butlerbrian67@gmail.com | 619-866-5916", contact_style))
    links_html = (
        '<a href="https://leetcode.com/u/Ban_Brian/" color="blue"><u>LeetCode</u></a> | '
        '<a href="https://www.linkedin.com/in/brian-butler-18036b33b/" color="blue"><u>LinkedIn</u></a> | '
        '<a href="https://github.com/Ban-Brian" color="blue"><u>GitHub</u></a>'
    )
    story.append(Paragraph(links_html, link_style))
    story.append(Spacer(1, 6))

    # --- EDUCATION ---
    story.append(Paragraph("EDUCATION", section_header))
    story.append(Paragraph(
        "Bachelor's Degree in Mathematical Statistics, Minor: Criminology | George Mason University | Expected Graduation: May 2027",
        body_style))
    story.append(Paragraph(
        "Relevant Coursework: Statistics, Discrete Mathematics, Business Analytics, Computer Science for Data (CDS 130)",
        body_style
    ))
    story.append(Spacer(1, 8))

    # --- WORK EXPERIENCE ---
    story.append(Paragraph("WORK EXPERIENCE", section_header))
    work_experiences = [
        {
            "title": "Research Assistant, George Mason University | Fairfax, VA | February 2025 - Present",
            "bullets": [
                "Optimized data workflows with Python and Excel, increasing experiment throughput by <b>20%</b>.",
                "Improved result reliability by <b>15%</b> through A/B testing and statistical analysis in R and Python.",
                "Automated weekly reporting with Python and Excel, creating dashboards to track lab metrics."
            ]
        },
        {
            "title": "Summer Intern, Department of Commerce, American Samoa Government | Pago Pago, AS | May 2025 - Aug 2025",
            "bullets": [
                "Analyzed economic data impacting ~<b>1,000</b> local businesses using Excel and statistical tools.",
                "Identified <b>3</b> major growth sectors through trend analysis and community surveys.",
                "Worked on a tourism development initiative expected to increase international visitors by <b>15%</b>.",
                "Created <b>10+</b> economic dashboards and visual summaries for senior officials and public reports."
            ]
        },
        {
            "title": "Research Intern, Nonprofit Alliance Against Domestic & Sexual Violence | San Diego, CA | June 2024 - Aug 2024",
            "bullets": [
                "Processed and analyzed survey data from over <b>150</b> respondents using Python and Excel.",
                "Evaluated pilot intervention effectiveness in SPSS, showing <b>25%</b> improvement in outcomes.",
                "Built predictive models to identify at-risk populations with <b>70%</b> precision.",
                "Developed <b>2</b> Tableau dashboards used in <b>2</b> grant proposals."
            ]
        }
    ]

    for job in work_experiences:
        story.append(Paragraph(job["title"], body_style))
        bullets = [ListItem(Paragraph(b, body_style), leftIndent=12, bulletIndent=6) for b in job["bullets"]]
        story.append(ListFlowable(bullets, bulletType='bullet'))
        story.append(Spacer(1, 5))

    # --- PROJECTS ---
    story.append(Paragraph("PROJECTS", section_header))
    projects = [
        {
            "title": "Exploratory Data Analysis on Pre-Trial Data | Feb 2025 - Current",
            "desc": "Analyzed <b>20,000+</b> records using Python (Pandas, Seaborn, Matplotlib), identifying <b>5+</b> key risk factors and delivering actionable insights via dashboards to Fairfax County stakeholders."
        },
        {
            "title": "Cost-of-Living Index Forecasting for American Samoa | May 2025 - Aug 2025",
            "desc": "Built multivariate time series model in Python to forecast cost-of-living trends for <b>50,000</b> residents, supporting government planning with predictive insights."
        },
        {
            "title": "Predictive Modeling for Nonprofit Alliance | June 2024 - Aug 2024",
            "desc": "Built XGBoost models on <b>10,000+</b> entries achieving <b>87%</b> accuracy to identify at-risk individuals; informed outreach and supported grant proposals that secured <b>$25K</b> funding."
        },
        {
            "title": "Baseline Trend Analysis & Outcome Projections Dashboard | Jan 2025 - Mar 2025",
            "desc": "Developed Python dashboard to track weekly client % and project outcomes for <b>500+</b> records; used LinearRegression to forecast <b>10-week</b> trends, improving stakeholder planning and resource allocation."
        }
    ]

    for project in projects:
        story.append(Paragraph(project["title"], body_style))
        story.append(Paragraph(project["desc"], body_style))
        story.append(Spacer(1, 5))

    # --- TECHNICAL SKILLS & MEMBERSHIPS SIDE BY SIDE ---
    story.append(Spacer(1, 10))
    tech_skills = [
        Paragraph("TECHNICAL SKILLS", section_header),
        Paragraph(
            "Machine Learning (TENSORS, Keras, XGBoost, Scikit-Learn) | Python Programming (NumPy, Pandas, Matplotlib, Seaborn) | SQL (Joins, CTEs, Subqueries, Window Functions) | Predictive Modeling (Classification, Regression, A/B Testing) | R (Tidyverse, ggplot2, dplyr) | Data Visualization (Tableau, Excel, Seaborn)",
            body_style
        )
    ]

    memberships = [
        Paragraph("MEMBERSHIPS", section_header),
        Paragraph("Member of American Statistical Association (ASA)", body_style),
        Paragraph("Web Design Editor, GMU Literary Club", body_style),
        Paragraph("Active Chess Club Member with <b>1700+</b> Elo", body_style),
        Paragraph("Billiards Club Member", body_style),
        Paragraph("George Mason Sports Analytics Club Member", body_style)
    ]

    two_col_table = Table(
        [[tech_skills, memberships]],
        colWidths=[3.75 * inch, 2.85 * inch]
    )
    two_col_table.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('LEFTPADDING', (0, 0), (-1, -1), 0),
        ('RIGHTPADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(two_col_table)

    # --- BUILD ---
    doc.build(story)
    print(f"PDF '{output_filename}' created successfully with embedded blue links.")


if __name__ == "__main__":
    create_resume_pdf("Brian_Butler_Resume_Full_Page_Links.pdf")