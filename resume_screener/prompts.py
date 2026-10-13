"""Prompt templates."""

from .schemas import JobRequirements

NOT_FOUND = "The uploaded resumes don't say."


def requirements_prompt(job_description: str) -> str:
    # A worked example steers small models far better than a list of rules (which they tend to echo back).
    return f"""Extract the hiring requirements from a job posting as JSON.

Example posting:
"Data Engineer. You will build batch and streaming pipelines. Requirements: 3+ years of experience,
strong SQL and Python, experience with Spark or Airflow. Experience with dbt is a plus.
Degree in Computer Science, Statistics or similar."

Example answer:
{{"title": "Data Engineer", "must_have": ["SQL", "Python", "Spark", "Airflow"], "nice_to_have": ["dbt"],
"min_years": 3, "degree": "bachelor", "degree_fields": ["Computer Science", "Statistics"]}}

Now the real posting:
\"\"\"
{job_description}
\"\"\"

Answer with JSON only. Skills are short technology names, never sentences."""


def facts_prompt(req: JobRequirements, resume_text: str) -> str:
    wanted = req.must_have + [s for s in req.nice_to_have if s not in req.must_have]
    return f"""You are reviewing one resume for the role "{req.title}".
Extract facts from the resume text only. If something is not stated, leave it out — do not guess.

Requirement list (use these exact spellings in `skills`): {wanted}

How to fill the fields
- skills: include a requirement only if the resume shows it, and quote a short phrase as evidence.
  A related tool counts when it clearly implies the requirement (e.g. "EC2" implies "AWS").
- years_experience: add up paid roles; internships and academic projects count as 0.
- relevant_years: years in jobs doing the same kind of work as "{req.title}" (e.g. any software development
  job counts for a software engineering role); 0 for an unrelated career.
- degree: bachelor for B.Tech / B.E. / B.Sc / B.Com / BA, master for M.Tech / M.Sc / MBA / MA.
- strengths / gaps: judge against this role, a few words each.

Resume text:
\"\"\"
{resume_text}
\"\"\""""


def answer_prompt(question: str, context: str) -> str:
    return f"""Answer a recruiter's question using only the resume excerpts below. Each excerpt block
starts with "### <file name>".

Rules
- Name the candidate and put the file name in square brackets after each claim, e.g. [Jane_Doe.pdf].
- If several candidates fit, give one "- " bullet per candidate with a few words on what their resume shows.
- If the excerpts do not contain the answer, reply exactly: {NOT_FOUND}
- Be brief and factual; never add details that are not in the excerpts.

Resume excerpts:
{context}

Question: {question}
Answer:"""


def brief_prompt(req: JobRequirements, shortlist_table: str, threshold: float) -> str:
    return f"""Write a short hiring brief for the role "{req.title}" from the screening results below.
Scores are 0-100 and were computed by the screening tool; the shortlist cut-off is {threshold:g}.

Format (Markdown, under 150 words):
1. One sentence on the overall strength of the pool.
2. A bullet for each of the top 3 candidates: name, score, the main reason to interview them.
3. One sentence on the most common skill gap, using the count given.
Use only the information below and copy numbers as given. Do not decide who gets hired.

Results:
{shortlist_table}"""
