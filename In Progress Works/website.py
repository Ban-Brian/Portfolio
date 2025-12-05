from flask import Flask

app = Flask(__name__)


@app.route("/")
def home():
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Brian Butler — Statistics & Research</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Syne:wght@400;500;600;700;800&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-primary: #0a0a0f;
            --bg-card: #12121a;
            --bg-card-hover: #1a1a25;
            --text-primary: #f4f4f5;
            --text-secondary: #9898a6;
            --accent-1: #6366f1;
            --accent-2: #8b5cf6;
            --accent-3: #ec4899;
            --accent-4: #06b6d4;
            --gradient-1: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #ec4899 100%);
            --gradient-2: linear-gradient(135deg, #06b6d4 0%, #6366f1 100%);
            --border: rgba(255, 255, 255, 0.08);
        }

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        html {
            scroll-behavior: smooth;
        }

        body {
            font-family: 'Space Grotesk', sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            line-height: 1.6;
            overflow-x: hidden;
            cursor: default;
        }

        .cursor-glow {
            position: fixed;
            width: 400px;
            height: 400px;
            border-radius: 50%;
            background: radial-gradient(circle, rgba(99, 102, 241, 0.15) 0%, transparent 70%);
            pointer-events: none;
            z-index: 0;
            transform: translate(-50%, -50%);
            transition: opacity 0.3s ease;
        }

        .grain {
            position: fixed;
            inset: 0;
            pointer-events: none;
            opacity: 0.04;
            background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 400 400' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noiseFilter'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noiseFilter)'/%3E%3C/svg%3E");
            z-index: 1000;
        }

        .floating-shapes {
            position: fixed;
            inset: 0;
            pointer-events: none;
            overflow: hidden;
        }

        .shape {
            position: absolute;
            border-radius: 50%;
            filter: blur(80px);
            opacity: 0.4;
            animation: morphFloat 20s ease-in-out infinite;
        }

        .shape-1 {
            width: 500px;
            height: 500px;
            background: var(--accent-1);
            top: -150px;
            right: -100px;
            animation-delay: 0s;
        }

        .shape-2 {
            width: 400px;
            height: 400px;
            background: var(--accent-3);
            bottom: -100px;
            left: -100px;
            animation-delay: -7s;
        }

        .shape-3 {
            width: 300px;
            height: 300px;
            background: var(--accent-4);
            top: 50%;
            left: 50%;
            animation-delay: -14s;
        }

        @keyframes morphFloat {
            0%, 100% { 
                transform: translate(0, 0) scale(1) rotate(0deg);
                border-radius: 50%;
            }
            25% { 
                transform: translate(50px, -30px) scale(1.1) rotate(90deg);
                border-radius: 40% 60% 70% 30%;
            }
            50% { 
                transform: translate(-20px, 40px) scale(0.9) rotate(180deg);
                border-radius: 60% 40% 30% 70%;
            }
            75% { 
                transform: translate(30px, 20px) scale(1.05) rotate(270deg);
                border-radius: 30% 70% 50% 50%;
            }
        }

        .container {
            max-width: 1100px;
            margin: 0 auto;
            padding: 0 2rem;
            position: relative;
            z-index: 1;
        }

        header {
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            justify-content: center;
            position: relative;
            padding: 4rem 0;
        }

        .header-content {
            opacity: 0;
            animation: fadeUp 1s ease forwards;
        }

        .tag {
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            font-size: 0.8rem;
            font-weight: 600;
            letter-spacing: 0.1em;
            text-transform: uppercase;
            color: var(--text-primary);
            background: var(--gradient-1);
            padding: 0.6rem 1.25rem;
            border-radius: 100px;
            margin-bottom: 2rem;
            animation: shimmer 3s ease-in-out infinite;
        }

        .tag::before {
            content: '';
            width: 8px;
            height: 8px;
            background: #22c55e;
            border-radius: 50%;
            animation: blink 2s ease-in-out infinite;
        }

        @keyframes blink {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.3; }
        }

        @keyframes shimmer {
            0%, 100% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
        }

        h1 {
            font-family: 'Syne', sans-serif;
            font-size: clamp(3.5rem, 12vw, 8rem);
            font-weight: 800;
            line-height: 1;
            letter-spacing: -0.03em;
            margin-bottom: 1.5rem;
            background: var(--gradient-1);
            background-size: 200% 200%;
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            animation: gradientShift 8s ease infinite;
        }

        @keyframes gradientShift {
            0%, 100% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
        }

        h1 span {
            display: block;
            font-family: 'Space Grotesk', sans-serif;
            font-size: 1.25rem;
            font-weight: 400;
            letter-spacing: 0;
            margin-top: 1rem;
            -webkit-text-fill-color: var(--text-secondary);
            background: none;
        }

        .intro {
            max-width: 580px;
            font-size: 1.125rem;
            color: var(--text-secondary);
            margin-bottom: 2.5rem;
        }

        .contact-links {
            display: flex;
            gap: 1.5rem;
            flex-wrap: wrap;
            margin-bottom: 2rem;
        }

        .contact-links a, .contact-links span {
            color: var(--text-secondary);
            text-decoration: none;
            font-size: 0.9rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
            transition: all 0.3s ease;
        }

        .contact-links a:hover {
            color: var(--accent-1);
            transform: translateX(4px);
        }

        .social-links {
            display: flex;
            gap: 1rem;
        }

        .social-links a {
            width: 50px;
            height: 50px;
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 16px;
            display: flex;
            align-items: center;
            justify-content: center;
            color: var(--text-secondary);
            text-decoration: none;
            font-size: 0.85rem;
            font-weight: 600;
            transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
            position: relative;
            overflow: hidden;
        }

        .social-links a::before {
            content: '';
            position: absolute;
            inset: 0;
            background: var(--gradient-1);
            opacity: 0;
            transition: opacity 0.3s ease;
        }

        .social-links a:hover {
            border-color: transparent;
            transform: translateY(-4px) scale(1.05);
            box-shadow: 0 10px 30px -10px rgba(99, 102, 241, 0.5);
        }

        .social-links a:hover::before {
            opacity: 1;
        }

        .social-links a span {
            position: relative;
            z-index: 1;
        }

        .scroll-hint {
            position: absolute;
            bottom: 3rem;
            left: 50%;
            transform: translateX(-50%);
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 0.75rem;
            color: var(--text-secondary);
            font-size: 0.75rem;
            letter-spacing: 0.15em;
            text-transform: uppercase;
            opacity: 0;
            animation: fadeUp 1s ease 0.8s forwards;
        }

        .scroll-arrow {
            width: 24px;
            height: 40px;
            border: 2px solid var(--accent-1);
            border-radius: 12px;
            position: relative;
        }

        .scroll-arrow::before {
            content: '';
            position: absolute;
            width: 4px;
            height: 8px;
            background: var(--accent-1);
            border-radius: 2px;
            top: 8px;
            left: 50%;
            transform: translateX(-50%);
            animation: scrollBounce 2s ease-in-out infinite;
        }

        @keyframes scrollBounce {
            0%, 100% { transform: translateX(-50%) translateY(0); opacity: 1; }
            50% { transform: translateX(-50%) translateY(12px); opacity: 0.3; }
        }

        section {
            padding: 6rem 0;
        }

        .section-header {
            display: flex;
            align-items: center;
            gap: 1.5rem;
            margin-bottom: 3rem;
            opacity: 0;
            transform: translateY(30px);
            transition: all 0.8s ease;
        }

        .section-header.visible {
            opacity: 1;
            transform: translateY(0);
        }

        .section-number {
            font-family: 'Syne', sans-serif;
            font-size: 0.9rem;
            font-weight: 700;
            background: var(--gradient-1);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }

        h2 {
            font-family: 'Syne', sans-serif;
            font-size: clamp(2rem, 5vw, 3.5rem);
            font-weight: 700;
        }

        .section-line {
            flex: 1;
            height: 2px;
            background: var(--gradient-1);
            opacity: 0.3;
            border-radius: 1px;
        }

        .about-content {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 4rem;
            align-items: start;
        }

        .about-text {
            font-size: 1.125rem;
            color: var(--text-secondary);
            opacity: 0;
            transform: translateY(30px);
            transition: all 0.8s ease 0.2s;
        }

        .about-text.visible {
            opacity: 1;
            transform: translateY(0);
        }

        .stats-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1.25rem;
        }

        .stat-card {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 20px;
            padding: 1.75rem;
            opacity: 0;
            transform: translateY(30px);
            transition: all 0.5s cubic-bezier(0.175, 0.885, 0.32, 1.275);
            position: relative;
            overflow: hidden;
        }

        .stat-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 3px;
            background: var(--gradient-1);
            transform: scaleX(0);
            transform-origin: left;
            transition: transform 0.4s ease;
        }

        .stat-card.visible {
            opacity: 1;
            transform: translateY(0);
        }

        .stat-card:hover {
            background: var(--bg-card-hover);
            transform: translateY(-8px) scale(1.02);
            box-shadow: 0 20px 40px -20px rgba(99, 102, 241, 0.3);
        }

        .stat-card:hover::before {
            transform: scaleX(1);
        }

        .stat-value {
            font-family: 'Syne', sans-serif;
            font-size: 2.75rem;
            font-weight: 800;
            background: var(--gradient-1);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 0.25rem;
        }

        .stat-label {
            font-size: 0.85rem;
            color: var(--text-secondary);
        }

        .skills-section {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 1.5rem;
        }

        .skill-category {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 24px;
            padding: 2rem;
            opacity: 0;
            transform: translateY(30px);
            transition: all 0.5s cubic-bezier(0.175, 0.885, 0.32, 1.275);
            position: relative;
        }

        .skill-category.visible {
            opacity: 1;
            transform: translateY(0);
        }

        .skill-category:hover {
            background: var(--bg-card-hover);
            transform: translateY(-4px);
        }

        .skill-category h3 {
            font-family: 'Syne', sans-serif;
            font-size: 0.85rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            background: var(--gradient-1);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 1.25rem;
        }

        .skill-tags {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
        }

        .skill-tag {
            background: rgba(99, 102, 241, 0.1);
            border: 1px solid rgba(99, 102, 241, 0.2);
            padding: 0.5rem 1rem;
            border-radius: 10px;
            font-size: 0.875rem;
            color: var(--text-secondary);
            transition: all 0.3s ease;
            cursor: default;
        }

        .skill-tag:hover {
            background: var(--gradient-1);
            border-color: transparent;
            color: white;
            transform: scale(1.05);
        }

        .experience-list {
            display: flex;
            flex-direction: column;
            gap: 2rem;
        }

        .exp-card {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 24px;
            padding: 2.5rem;
            display: grid;
            grid-template-columns: 220px 1fr;
            gap: 3rem;
            opacity: 0;
            transform: translateY(30px);
            transition: all 0.5s cubic-bezier(0.175, 0.885, 0.32, 1.275);
            position: relative;
            overflow: hidden;
        }

        .exp-card::before {
            content: '';
            position: absolute;
            left: 0;
            top: 0;
            bottom: 0;
            width: 4px;
            background: var(--gradient-1);
            transform: scaleY(0);
            transform-origin: top;
            transition: transform 0.5s ease;
        }

        .exp-card.visible {
            opacity: 1;
            transform: translateY(0);
        }

        .exp-card:hover {
            background: var(--bg-card-hover);
            border-color: rgba(99, 102, 241, 0.3);
        }

        .exp-card:hover::before {
            transform: scaleY(1);
        }

        .exp-meta {
            border-right: 1px solid var(--border);
            padding-right: 2rem;
        }

        .exp-type {
            display: inline-block;
            font-size: 0.7rem;
            font-weight: 700;
            letter-spacing: 0.1em;
            text-transform: uppercase;
            background: var(--gradient-1);
            color: white;
            padding: 0.4rem 0.8rem;
            border-radius: 6px;
            margin-bottom: 1rem;
        }

        .exp-title {
            font-family: 'Syne', sans-serif;
            font-size: 1.35rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
        }

        .exp-org {
            font-size: 0.9rem;
            color: var(--text-secondary);
            margin-bottom: 0.25rem;
        }

        .exp-date {
            font-size: 0.8rem;
            color: var(--text-secondary);
            opacity: 0.7;
        }

        .exp-content h4 {
            font-size: 0.8rem;
            font-weight: 600;
            color: var(--text-secondary);
            margin-bottom: 1rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }

        .exp-content ul {
            list-style: none;
        }

        .exp-content li {
            position: relative;
            padding-left: 1.75rem;
            margin-bottom: 0.85rem;
            color: var(--text-secondary);
            font-size: 0.95rem;
        }

        .exp-content li::before {
            content: '';
            position: absolute;
            left: 0;
            top: 0.5em;
            width: 8px;
            height: 8px;
            background: var(--gradient-1);
            border-radius: 50%;
        }

        .highlight {
            color: var(--text-primary);
            font-weight: 600;
            background: linear-gradient(transparent 60%, rgba(99, 102, 241, 0.3) 60%);
        }

        .projects-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 1.5rem;
        }

        .project-card {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 24px;
            padding: 2rem;
            opacity: 0;
            transform: translateY(30px);
            transition: all 0.5s cubic-bezier(0.175, 0.885, 0.32, 1.275);
            position: relative;
            overflow: hidden;
        }

        .project-card.visible {
            opacity: 1;
            transform: translateY(0);
        }

        .project-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 100%;
            background: var(--gradient-1);
            opacity: 0;
            transition: opacity 0.3s ease;
            z-index: 0;
        }

        .project-card:hover {
            transform: translateY(-8px) scale(1.02);
            border-color: transparent;
            box-shadow: 0 25px 50px -20px rgba(99, 102, 241, 0.4);
        }

        .project-card:hover::before {
            opacity: 0.05;
        }

        .project-card > * {
            position: relative;
            z-index: 1;
        }

        .project-date {
            display: inline-block;
            font-size: 0.7rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            background: var(--gradient-2);
            color: white;
            padding: 0.35rem 0.75rem;
            border-radius: 6px;
            margin-bottom: 1rem;
        }

        .project-card h3 {
            font-family: 'Syne', sans-serif;
            font-size: 1.25rem;
            font-weight: 700;
            margin-bottom: 1rem;
        }

        .project-card p {
            font-size: 0.9rem;
            color: var(--text-secondary);
            line-height: 1.7;
        }

        .project-metrics {
            display: flex;
            gap: 1.25rem;
            margin-top: 1.5rem;
            padding-top: 1.5rem;
            border-top: 1px solid var(--border);
        }

        .metric {
            font-size: 0.8rem;
            color: var(--text-secondary);
            display: flex;
            align-items: center;
            gap: 0.4rem;
        }

        .metric strong {
            font-family: 'Syne', sans-serif;
            font-weight: 700;
            background: var(--gradient-1);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }

        .edu-card {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 24px;
            padding: 3rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 2rem;
            opacity: 0;
            transform: translateY(30px);
            transition: all 0.5s cubic-bezier(0.175, 0.885, 0.32, 1.275);
            position: relative;
            overflow: hidden;
        }

        .edu-card::before {
            content: '';
            position: absolute;
            inset: 0;
            background: var(--gradient-1);
            opacity: 0.03;
        }

        .edu-card.visible {
            opacity: 1;
            transform: translateY(0);
        }

        .edu-card:hover {
            border-color: rgba(99, 102, 241, 0.3);
        }

        .edu-degree {
            font-family: 'Syne', sans-serif;
            font-size: 1.75rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
            position: relative;
        }

        .edu-school {
            color: var(--text-secondary);
            margin-bottom: 1rem;
            position: relative;
        }

        .edu-courses {
            font-size: 0.875rem;
            color: var(--text-secondary);
            max-width: 500px;
            position: relative;
        }

        .edu-year {
            font-family: 'Syne', sans-serif;
            font-size: 5rem;
            font-weight: 800;
            background: var(--gradient-1);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            opacity: 0.4;
            position: relative;
        }

        footer {
            border-top: 1px solid var(--border);
            padding: 4rem 0;
            text-align: center;
            position: relative;
        }

        .footer-name {
            font-family: 'Syne', sans-serif;
            font-size: 1.75rem;
            font-weight: 700;
            margin-bottom: 1.5rem;
            background: var(--gradient-1);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }

        .footer-contact {
            display: flex;
            justify-content: center;
            gap: 2rem;
            margin-bottom: 2rem;
            flex-wrap: wrap;
        }

        .footer-contact a {
            color: var(--text-secondary);
            text-decoration: none;
            font-size: 0.9rem;
            transition: all 0.3s ease;
            padding: 0.5rem 1rem;
            border-radius: 8px;
        }

        .footer-contact a:hover {
            color: white;
            background: var(--gradient-1);
        }

        .footer-copy {
            font-size: 0.85rem;
            color: var(--text-secondary);
            opacity: 0.6;
        }

        @keyframes fadeUp {
            from { opacity: 0; transform: translateY(40px); }
            to { opacity: 1; transform: translateY(0); }
        }

        @media (max-width: 900px) {
            .projects-grid {
                grid-template-columns: 1fr;
            }

            .skills-section {
                grid-template-columns: 1fr;
            }
        }

        @media (max-width: 768px) {
            .about-content {
                grid-template-columns: 1fr;
                gap: 2rem;
            }

            .exp-card {
                grid-template-columns: 1fr;
                gap: 1.5rem;
            }

            .exp-meta {
                border-right: none;
                border-bottom: 1px solid var(--border);
                padding-right: 0;
                padding-bottom: 1.5rem;
            }

            .edu-card {
                flex-direction: column;
                text-align: center;
            }

            .edu-year {
                order: -1;
            }

            .contact-links {
                flex-direction: column;
                gap: 0.75rem;
            }

            .floating-shapes .shape {
                opacity: 0.2;
            }
        }
    </style>
</head>
<body>
    <div class="cursor-glow" id="cursorGlow"></div>
    <div class="grain"></div>
    <div class="floating-shapes">
        <div class="shape shape-1"></div>
        <div class="shape shape-2"></div>
        <div class="shape shape-3"></div>
    </div>

    <div class="container">
        <header>
            <div class="header-content">
                <div class="tag">Available for Opportunities</div>
                <h1>
                    Brian Butler
                    <span>Mathematical Statistics & Data Science</span>
                </h1>
                <p class="intro">
                    Building predictive models and transforming complex data into actionable insights. 
                    Experience spanning government research, nonprofit analytics, and market microstructure analysis.
                </p>
                <div class="contact-links">
                    <a href="mailto:butlerbrian67@gmail.com">butlerbrian67@gmail.com</a>
                    <a href="tel:619-866-5916">619-866-5916</a>
                    <span>Fairfax, VA</span>
                </div>
                <div class="social-links">
                    <a href="https://github.com/Ban-Brian" target="_blank" title="GitHub"><span>GH</span></a>
                    <a href="https://www.linkedin.com/in/brian-butler-18036b33b/" target="_blank" title="LinkedIn"><span>IN</span></a>
                    <a href="https://leetcode.com/u/Ban_Brian/" target="_blank" title="LeetCode"><span>LC</span></a>
                </div>
            </div>
            <div class="scroll-hint">
                <span>Scroll</span>
                <div class="scroll-arrow"></div>
            </div>
        </header>

        <section id="about">
            <div class="section-header">
                <span class="section-number">01</span>
                <h2>About</h2>
                <div class="section-line"></div>
            </div>
            <div class="about-content">
                <div class="about-text">
                    <p>
                        I'm a Mathematical Statistics student at George Mason University with 
                        hands-on experience in data analysis, machine learning, and statistical modeling.
                        From analyzing economic data for the American Samoa government to building 
                        predictive models for nonprofit organizations, I specialize in turning raw data 
                        into strategic insights that drive real-world impact.
                    </p>
                </div>
                <div class="stats-grid">
                    <div class="stat-card">
                        <div class="stat-value">87%</div>
                        <div class="stat-label">Model Accuracy Achieved</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">$25K</div>
                        <div class="stat-label">Grant Funding Supported</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">80%</div>
                        <div class="stat-label">Manual Work Reduced</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">1000+</div>
                        <div class="stat-label">Enterprises Analyzed</div>
                    </div>
                </div>
            </div>
        </section>

        <section id="skills">
            <div class="section-header">
                <span class="section-number">02</span>
                <h2>Technical Skills</h2>
                <div class="section-line"></div>
            </div>
            <div class="skills-section">
                <div class="skill-category">
                    <h3>Machine Learning</h3>
                    <div class="skill-tags">
                        <span class="skill-tag">XGBoost</span>
                        <span class="skill-tag">LightGBM</span>
                        <span class="skill-tag">CatBoost</span>
                        <span class="skill-tag">TensorFlow</span>
                        <span class="skill-tag">Keras</span>
                        <span class="skill-tag">Scikit-learn</span>
                    </div>
                </div>
                <div class="skill-category">
                    <h3>Python Ecosystem</h3>
                    <div class="skill-tags">
                        <span class="skill-tag">NumPy</span>
                        <span class="skill-tag">Pandas</span>
                        <span class="skill-tag">Matplotlib</span>
                        <span class="skill-tag">Seaborn</span>
                        <span class="skill-tag">Plotly</span>
                    </div>
                </div>
                <div class="skill-category">
                    <h3>SQL & Databases</h3>
                    <div class="skill-tags">
                        <span class="skill-tag">PostgreSQL</span>
                        <span class="skill-tag">MySQL</span>
                        <span class="skill-tag">SQLite</span>
                        <span class="skill-tag">MongoDB</span>
                        <span class="skill-tag">CTEs</span>
                        <span class="skill-tag">Window Functions</span>
                    </div>
                </div>
                <div class="skill-category">
                    <h3>Analytics & Visualization</h3>
                    <div class="skill-tags">
                        <span class="skill-tag">R</span>
                        <span class="skill-tag">SPSS</span>
                        <span class="skill-tag">Tableau</span>
                        <span class="skill-tag">Power BI</span>
                        <span class="skill-tag">Excel</span>
                    </div>
                </div>
            </div>
        </section>

        <section id="experience">
            <div class="section-header">
                <span class="section-number">03</span>
                <h2>Experience</h2>
                <div class="section-line"></div>
            </div>
            <div class="experience-list">
                <div class="exp-card">
                    <div class="exp-meta">
                        <div class="exp-type">Current Role</div>
                        <div class="exp-title">Research Assistant</div>
                        <div class="exp-org">George Mason University</div>
                        <div class="exp-date">Feb 2025 - Present</div>
                    </div>
                    <div class="exp-content">
                        <h4>Key Contributions</h4>
                        <ul>
                            <li>Increased data processing efficiency by <span class="highlight">20%</span> using Python and Excel automation</li>
                            <li>Improved result reliability <span class="highlight">15%</span> through A/B testing, hypothesis evaluation, and reproducible analysis in R and Python</li>
                            <li>Automated weekly reporting pipelines and interactive dashboards, reducing manual work <span class="highlight">80%</span> and enabling real-time metric tracking</li>
                        </ul>
                    </div>
                </div>
                <div class="exp-card">
                    <div class="exp-meta">
                        <div class="exp-type">Internship</div>
                        <div class="exp-title">Summer Intern</div>
                        <div class="exp-org">Dept. of Commerce, American Samoa</div>
                        <div class="exp-date">May 2025 - Aug 2025</div>
                    </div>
                    <div class="exp-content">
                        <h4>Key Contributions</h4>
                        <ul>
                            <li>Analyzed economic and business data from <span class="highlight">1,000+</span> local enterprises, identifying 3 key growth sectors through trend and regression analysis</li>
                            <li>Supported tourism development initiative projected to increase international visitors <span class="highlight">15%</span> using community survey insights</li>
                            <li>Built <span class="highlight">10+</span> dashboards and visual reports in Excel and Power BI, improving accessibility of economic data for senior officials</li>
                        </ul>
                    </div>
                </div>
                <div class="exp-card">
                    <div class="exp-meta">
                        <div class="exp-type">Internship</div>
                        <div class="exp-title">Research Intern</div>
                        <div class="exp-org">Nonprofit Alliance Against Domestic & Sexual Violence</div>
                        <div class="exp-date">Jun 2024 - Aug 2024</div>
                    </div>
                    <div class="exp-content">
                        <h4>Key Contributions</h4>
                        <ul>
                            <li>Cleaned and analyzed <span class="highlight">150+</span> survey responses using Excel, Python, and SPSS to evaluate intervention effectiveness</li>
                            <li>Utilized advanced Excel functions to visualize outcomes and conduct comparative analysis, supporting <span class="highlight">25%</span> improvement in program results</li>
                            <li>Built predictive models (<span class="highlight">70%</span> precision) to identify at-risk populations and developed 2 Tableau dashboards for successful grant proposals</li>
                        </ul>
                    </div>
                </div>
            </div>
        </section>

        <section id="projects">
            <div class="section-header">
                <span class="section-number">04</span>
                <h2>Projects</h2>
                <div class="section-line"></div>
            </div>
            <div class="projects-grid">
                <div class="project-card">
                    <div class="project-date">October 2025</div>
                    <h3>Predicting Road Accident Risk Using Ensemble ML</h3>
                    <p>
                        Developed a stacked ensemble model combining CatBoost and LightGBM with a Ridge regression meta-model. 
                        Engineered interaction and polynomial features, applied K-Fold cross-validation for robust predictions.
                    </p>
                    <div class="project-metrics">
                        <span class="metric"><strong>~0.056</strong> RMSE</span>
                        <span class="metric"><strong>Ensemble</strong> Learning</span>
                    </div>
                </div>
                <div class="project-card">
                    <div class="project-date">May - Aug 2025</div>
                    <h3>Cost-of-Living Index Forecasting</h3>
                    <p>
                        Built multivariate time series model in Python to forecast cost-of-living trends for 50,000 residents 
                        of American Samoa, supporting government planning with predictive insights.
                    </p>
                    <div class="project-metrics">
                        <span class="metric"><strong>50K</strong> Residents</span>
                        <span class="metric"><strong>Time Series</strong> Analysis</span>
                    </div>
                </div>
                <div class="project-card">
                    <div class="project-date">2025</div>
                    <h3>Crypto Market Microstructure Analysis</h3>
                    <p>
                        Python-based analysis of Coinbase, Binance, and Kraken order books. Measured bid-ask spreads, 
                        order book depth, price impact, and volatility. Built regression models predicting short-term returns.
                    </p>
                    <div class="project-metrics">
                        <span class="metric"><strong>3</strong> Exchanges</span>
                        <span class="metric"><strong>Algorithmic</strong> Insights</span>
                    </div>
                </div>
                <div class="project-card">
                    <div class="project-date">Jun - Aug 2024</div>
                    <h3>Predictive Modeling for Nonprofit</h3>
                    <p>
                        Built XGBoost models on 10,000+ entries achieving 87% accuracy to identify at-risk individuals. 
                        Informed outreach strategies and supported grant proposals that secured funding.
                    </p>
                    <div class="project-metrics">
                        <span class="metric"><strong>87%</strong> Accuracy</span>
                        <span class="metric"><strong>$25K</strong> Funding</span>
                    </div>
                </div>
            </div>
        </section>

        <section id="education">
            <div class="section-header">
                <span class="section-number">05</span>
                <h2>Education</h2>
                <div class="section-line"></div>
            </div>
            <div class="edu-card">
                <div>
                    <div class="edu-degree">B.S. Mathematical Statistics</div>
                    <div class="edu-school">George Mason University</div>
                    <div class="edu-courses">
                        Statistics, Discrete Mathematics, Higher Mathematics, Computer Science for Data (CDS 130)
                    </div>
                </div>
                <div class="edu-year">2027</div>
            </div>
        </section>

        <footer>
            <div class="footer-name">Brian Butler</div>
            <div class="footer-contact">
                <a href="mailto:butlerbrian67@gmail.com">butlerbrian67@gmail.com</a>
                <a href="tel:619-866-5916">619-866-5916</a>
                <a href="https://github.com/Ban-Brian" target="_blank">GitHub</a>
                <a href="https://www.linkedin.com/in/brian-butler-18036b33b/" target="_blank">LinkedIn</a>
                <a href="https://leetcode.com/u/Ban_Brian/" target="_blank">LeetCode</a>
            </div>
            <div class="footer-copy">&copy; 2025 Brian Butler. All rights reserved.</div>
        </footer>
    </div>

    <script>
        // Cursor glow effect
        const cursorGlow = document.getElementById('cursorGlow');
        let mouseX = 0, mouseY = 0;
        let glowX = 0, glowY = 0;

        document.addEventListener('mousemove', (e) => {
            mouseX = e.clientX;
            mouseY = e.clientY;
        });

        function animateGlow() {
            glowX += (mouseX - glowX) * 0.1;
            glowY += (mouseY - glowY) * 0.1;
            cursorGlow.style.left = glowX + 'px';
            cursorGlow.style.top = glowY + 'px';
            requestAnimationFrame(animateGlow);
        }
        animateGlow();

        // Intersection Observer for scroll animations
        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.classList.add('visible');
                }
            });
        }, { threshold: 0.1, rootMargin: '0px 0px -50px 0px' });

        document.querySelectorAll('.section-header, .about-text, .stat-card, .skill-category, .exp-card, .project-card, .edu-card').forEach(el => {
            observer.observe(el);
        });

        // Staggered animation delays
        document.querySelectorAll('.stat-card').forEach((card, i) => {
            card.style.transitionDelay = `${i * 0.1}s`;
        });

        document.querySelectorAll('.skill-category').forEach((card, i) => {
            card.style.transitionDelay = `${i * 0.1}s`;
        });

        document.querySelectorAll('.project-card').forEach((card, i) => {
            card.style.transitionDelay = `${i * 0.1}s`;
        });

        document.querySelectorAll('.exp-card').forEach((card, i) => {
            card.style.transitionDelay = `${i * 0.15}s`;
        });
    </script>
</body>
</html>
"""


if __name__ == "__main__":
    app.run(debug=True)