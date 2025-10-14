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
        "Bachelor's Degree in Mathematical Statistics | George Mason University | Expected Graduation: May 2027",
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
            "title": "Cost-of-Living Index Forecasting for American Samoa | May 2025 - Aug 2025",
            "desc": "Built multivariate time series model in Python to forecast cost-of-living trends for <b>50,000</b> residents, supporting government planning with predictive insights."
        },
        {
            "title": "Predictive Modeling for Nonprofit Alliance | June 2024 - Aug 2024",
            "desc": "Built XGBoost models on <b>10,000+</b> entries achieving <b>87%</b> accuracy to identify at-risk individuals; informed outreach and supported grant proposals that secured <b>$25K</b> funding."
        },
        {
            "title": "Predicting Road Accident Risk Using Ensemble ML | October 2025",
            "desc": "Developed a stacked ensemble model combining <b>CatBoost</b> and <b>LightGBM</b> with a <b>Ridge regression</b> meta-model to predict accident risk. Engineered interaction and polynomial features, applied <b>K-Fold cross-validation</b>, and achieved high-accuracy predictions (RMSE ~0.056). Demonstrated advanced <b>feature engineering, ensemble learning, and predictive modeling</b> skills."
        }
    ]

    for project in projects:
        story.append(Paragraph(project["title"], body_style))
        story.append(Paragraph(project["desc"], body_style))
        story.append(Spacer(1, 5))

    # --- TECHNICAL SKILLS & MEMBERSHIPS SIDE BY SIDE ---
    story.append(Spacer(1, 16))  # increased spacing before table

    # TECHNICAL SKILLS
    tech_skills = [
        Paragraph("TECHNICAL SKILLS", section_header),
        Paragraph(
            "Machine Learning: XGBoost, LightGBM, CatBoost, TensorFlow, Keras<br/>"
            "Python: NumPy, Pandas, Matplotlib, Seaborn<br/>"
            "SQL: Joins, CTEs, Window Functions<br/>"
            "Predictive Modeling: Regression, Classification, A/B Testing<br/>"
            "R: Tidyverse, ggplot2, dplyr<br/>"
            "Data Visualization: Tableau, Excel, Seaborn",
            body_style
        )
    ]

    # MEMBERSHIPS & ACHIEVEMENTS
    memberships = [
        Paragraph("MEMBERSHIPS & ACHIEVEMENTS", section_header),
        Paragraph("Member, American Statistical Association ", body_style),
        Paragraph("Achieved rank #210/5,000 in Kaggle Playground Competition", body_style),
        Paragraph("Active Chess Club Member, Elo 1700+", body_style),
        Paragraph("Billiards Club Member", body_style),
        Paragraph("George Mason Sports Analytics Club Vice-President", body_style)
    ]

    # Create a cleaner two-column layout with more separation
    two_col_table = Table(
        [[tech_skills, memberships]],
        colWidths=[3.8 * inch, 2.8 * inch],
        hAlign='LEFT',
        spaceBefore=12,  # space above table
        spaceAfter=12  # space below table
    )
    two_col_table.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('LEFTPADDING', (0, 0), (-1, -1), 0),
        ('RIGHTPADDING', (0, 0), (-1, -1), 10),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4)
    ]))

    story.append(two_col_table)

    # --- BUILD ---
    doc.build(story)
    print(f"PDF '{output_filename}' created successfully with embedded blue links.")


if __name__ == "__main__":
    create_resume_pdf("Brian_Butler_Resume_Full_Page_Links.pdf")