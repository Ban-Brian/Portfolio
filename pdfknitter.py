from fpdf import FPDF

from reesume import resume
def generate_resume_pdf(resume):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", style='B', size=16)
    pdf.cell(200, 10, resume["Name"], ln=True, align='C')
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, resume["Location"], ln=True, align='C')
    pdf.cell(200, 10, f"Email: {resume['Contact']['Email']} | Phone: {resume['Contact']['Phone']}", ln=True, align='C')

    pdf.ln(10)
    pdf.set_font("Arial", style='B', size=14)
    pdf.cell(0, 10, "Summary", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, resume["Summary"])

    pdf.ln(5)
    pdf.set_font("Arial", style='B', size=14)
    pdf.cell(0, 10, "Skills", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, ', '.join(resume["Skills"]))

    pdf.ln(5)
    pdf.set_font("Arial", style='B', size=14)
    pdf.cell(0, 10, "Work History", ln=True)
    pdf.set_font("Arial", size=12)
    for job in resume["Work History"]:
        pdf.cell(0, 10, f"{job['Position']} - {job['Company']} ({job['Dates']})", ln=True)
        pdf.set_font("Arial", size=11)
        for responsibility in job["Responsibilities"]:
            pdf.multi_cell(0, 7, f"- {responsibility}")
        pdf.ln(3)
        pdf.set_font("Arial", size=12)

    pdf.ln(5)
    pdf.set_font("Arial", style='B', size=14)
    pdf.cell(0, 10, "Education", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10,
             f"{resume['Education']['Degree']} - {resume['Education']['Institution']} ({resume['Education']['Expected Graduation']})",
             ln=True)

    pdf.output("Brian_Butler_Resume.pdf")
    print("Resume PDF generated successfully!")


# Call the function
generate_resume_pdf(resume)
