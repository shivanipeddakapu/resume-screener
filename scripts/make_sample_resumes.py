"""Generate the fictional sample resumes in samples/ (requires reportlab: pip install -r requirements-dev.txt).

usage: python scripts/make_sample_resumes.py [out_dir]   (default: samples/)
"""

import sys
import zipfile
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent / "samples"
OUT.mkdir(parents=True, exist_ok=True)

RESUMES = {
    "Arjun_Mehta": dict(
        title="Senior Software Engineer",
        contact="arjun.mehta@example.com | Bengaluru",
        summary="Backend engineer with 4 years building large-scale distributed systems in Java and Python.",
        exp=[
            "Software Engineer II, ShopKart (2022-Present): Designed a distributed order-indexing service on Kafka and Cassandra handling 50K req/s; built fault-tolerant query APIs in Java/Spring Boot; reduced storage cost 30% with tiered S3 storage.",
            "Software Engineer, PayLane (2020-2022): Built microservices in Python (FastAPI) and Go; implemented CI/CD with Jenkins and Docker; worked in two-week agile sprints.",
        ],
        skills="Java, Python, Go, Spring Boot, Kafka, Cassandra, AWS, Docker, Kubernetes, distributed systems, system design, agile/Scrum",
        edu="B.Tech in Computer Science, NIT Warangal (2020)",
    ),
    "Sneha_Reddy": dict(
        title="Software Development Engineer",
        contact="sneha.reddy@example.com | Hyderabad",
        summary="SDE with 2.5 years of experience building scalable cloud services and search infrastructure.",
        exp=[
            "SDE, CloudNova (2023-Present): Built a scalable full-text search and indexing pipeline with Elasticsearch; wrote C++ and Python services for low-latency query serving; on-call for production systems.",
            "Associate Engineer, DataBridge (2022-2023): Developed REST APIs in Java; wrote unit and integration tests; participated in agile ceremonies.",
        ],
        skills="C++, Java, Python, Elasticsearch, Redis, PostgreSQL, AWS, Linux, data structures & algorithms, agile",
        edu="B.E. in Computer Science, Osmania University (2022)",
    ),
    "Rahul_Verma": dict(
        title="Full Stack Developer",
        contact="rahul.verma@example.com | Pune",
        summary="Full stack developer with 1.5 years of professional experience in JavaScript and Node.js.",
        exp=[
            "Full Stack Developer, WebCraft Studios (2024-Present): Built React front-ends and Node.js/Express APIs for e-commerce clients; MongoDB schema design; deployed on AWS EC2.",
            "Software Intern, AppWorks (2023): Bug fixes and small features in a React Native app.",
        ],
        skills="JavaScript, TypeScript, React, Node.js, Express, MongoDB, HTML/CSS, Git, AWS EC2",
        edu="B.Sc. in Information Technology, Pune University (2023)",
    ),
    "Ananya_Iyer": dict(
        title="Data Analyst",
        contact="ananya.iyer@example.com | Chennai",
        summary="Data analyst with 3 years of experience in reporting, dashboards and business insights.",
        exp=[
            "Data Analyst, RetailMax (2022-Present): Built Power BI dashboards for sales KPIs; wrote SQL queries on Snowflake; automated Excel reports with Python (pandas).",
            "Junior Analyst, MarketPulse (2021-2022): Survey data cleaning and visualisation in Tableau.",
        ],
        skills="SQL, Python (pandas), Power BI, Tableau, Excel, Snowflake, statistics",
        edu="B.Com with Business Analytics, Madras Christian College (2021)",
    ),
    "Kiran_Das": dict(
        title="Computer Science Graduate",
        contact="kiran.das@example.com | Kolkata",
        summary="Recent CS graduate passionate about backend development and distributed systems.",
        exp=[
            "Intern, TechSeed (Summer 2025): Built a URL-shortener service in Java with Redis caching.",
            "Project: Distributed key-value store in Go using the Raft consensus algorithm (academic project).",
        ],
        skills="Java, Go, Python, Redis, Git, Linux, data structures & algorithms",
        edu="B.Tech in Computer Science, Jadavpur University (2025)",
    ),
    "Meera_Nair": dict(
        title="Digital Marketing Specialist",
        contact="meera.nair@example.com | Kochi",
        summary="Marketing professional with 5 years of experience in SEO, social media and campaigns.",
        exp=[
            "Marketing Manager, BrightAds (2021-Present): Led social media campaigns with 2M reach; managed Google Ads budgets; content strategy.",
            "SEO Executive, WebRank (2019-2021): Keyword research and on-page optimisation.",
        ],
        skills="SEO, Google Ads, Meta Ads, content marketing, Canva, Google Analytics",
        edu="MBA in Marketing, Kerala University (2019)",
    ),
}

ss = getSampleStyleSheet()
for name, r in RESUMES.items():
    path = OUT / f"{name}.pdf"
    story = [
        Paragraph(name.replace("_", " "), ss["Title"]),
        Paragraph(f"<b>{r['title']}</b> &nbsp; | &nbsp; {r['contact']}", ss["Normal"]),
        Spacer(1, 12),
        Paragraph("Summary", ss["Heading2"]),
        Paragraph(r["summary"], ss["Normal"]),
        Paragraph("Experience", ss["Heading2"]),
    ]
    story += [Paragraph("&bull; " + e, ss["Normal"]) for e in r["exp"]]
    story += [
        Paragraph("Skills", ss["Heading2"]),
        Paragraph(r["skills"], ss["Normal"]),
        Paragraph("Education", ss["Heading2"]),
        Paragraph(r["edu"], ss["Normal"]),
    ]
    SimpleDocTemplate(str(path), pagesize=A4).build(story)

with zipfile.ZipFile(OUT / "sample_resumes.zip", "w") as zf:
    for p in sorted(OUT.glob("*.pdf")):
        zf.write(p, p.name)
print("wrote", OUT / "sample_resumes.zip")
