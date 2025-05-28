resume = {
    "Name": "Brian Butler",
    "Location": "Fairfax, VA 22030",
       "Contact": {
        "Email": "butlerbrian67@gmail.com",
        "Phone": "619-866-5916"
    },
    "Projects": [
            {
                "Title": "Exploratory Data Analysis on Pre-Trial Data",
                "Description": "Performed comprehensive exploratory data analysis using Python and visualization libraries to uncover trends and inform research decisions for academic projects."
            },
            {
                "Title": "Predictive Modeling for Nonprofit Alliance",
                "Description": "Developed and validated predictive models in Python to identify at-risk groups, supporting targeted outreach strategies and improving client outcomes for the Nonprofit Alliance Against Domestic & Sexual Violence."
            },
            {
                "Title": "Machine Learning Projects",
                "Description": "Built and evaluated machine learning models for various datasets, applying techniques such as classification, regression, and clustering to solve real-world problems and present actionable insights."
            }
        ],
    "Websites": {
        "LinkedIn": "https://www.linkedin.com/in/brian-butler-18036b33b/",
        "GitHub": "https://github.com/Ban-Brian/The-Forge-Website",
        "LeetCode": "https://leetcode.com/u/Ban_Brian/"
    },
    "Skills": [
        "Experimental design",
        "Python programming language",
        "SQL",
        "Research design",
        "R Code",
        "Incidents management"
    ],
    "Memberships": [
        "Member of American Statistical Association",
        "Web Design Editor of the GMU Literary Club",
        "Chess Club Member"
    ],
    "Work Experience": [
        {
            "Position": "Research Assistant",
            "Company": "George Mason University",
            "Location": "Fairfax, VA",
            "Dates": "February 2025 - Current",
            "Responsibilities": [
                "Adhered to laboratory safety procedures to maintain compliance with quality control standards.",
                "Developed new protocols and improved existing laboratory processes.",
                "Analyzed results using analytical software and created reports.",
                "Completed AB testing and data analysis to improve research outcomes.",
                "Used SQL to extract data from databases and perform data analysis."
            ]
        },
        {
            "Position": " Head Chapter Development Manager ",
            "Company": "Health Link Society",
            "Location": "Remote, VA",
            "Dates": "December 2024 - Current",
            "Responsibilities": [
                "Leading the expansion of National chapters by developing growth strategies and fostering collaboration among teams.",
                "Provided training for on boarding chapters as well as other managers.",
                "Delivering compelling presentations and driving alignment to achieve measurable outcomes.",
                "Used Tableu to create reports and dashboards for stakeholders with data from R."
            ]
        },
        {
            "Position": "Research Intern",
            "Company": "Nonprofit Alliance Against Domestic & Sexual Violence",
            "Location": "San Diego, CA",
            "Dates": "June 2024 - August 2024",
            "Responsibilities": [
                "Utilized Python and Excel to analyze survey data, identifying resource gaps and informing a report on underserved populations.",
                "Conducted a statistical evaluation of a pilot intervention program using SPSS, demonstrating a 25% improvement in client outcomes.",
                "Built predictive models in Python to identify at-risk groups, supporting targeted outreach strategies.",
                "Designed data visualizations in Tableau for grant proposals, contributing to securing funding for two new initiatives."
            ]
        },
        {
            "Position": "Youth Outreach Leader",
            "Company": "Rock Church",
            "Location": "San Marcos, CA",
            "Dates": "March 2018 - August 2023",
            "Responsibilities": [
                "Organized six youth road programs, engaging over 200 participants annually to promote personal growth and community involvement.",
                "Developed a mentorship tracking system in Google Sheets, enhancing retention and engagement rates by 15%.",
            ]
        }
    ],
    "Education": {
        "Degree": "Bachelor's Degree in Applied Statistics",
        "Minor": "Criminology and Math",
        "Institution": "George Mason University",
        "Expected Graduation": "May 2027",
        "GPA": "3.5",
        "Relevant Coursework": [
        "Calculus ",
        "Discrete Mathematics",
        "Business Analytics 201",
        "Computer Data Science 130"
    ]
    }
}

# Print resume for verification
import json
print(json.dumps(resume, indent=4))