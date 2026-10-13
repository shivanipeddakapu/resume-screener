"""What the LLM is asked to extract. It reports facts; all scoring happens in Python."""

from typing import Literal

from pydantic import BaseModel, Field

DegreeLevel = Literal["none", "diploma", "bachelor", "master", "doctorate"]


class JobRequirements(BaseModel):
    title: str = Field(description="Role name, e.g. 'Backend Engineer'")
    must_have: list[str] = Field(description="Skills or technologies the role requires. Short names, at most 10.")
    nice_to_have: list[str] = Field(default_factory=list, description="Skills mentioned as a plus or preferred. At most 8.")
    min_years: float = Field(default=0, description="Minimum years of professional experience asked for; 0 if none")
    degree: DegreeLevel = Field(default="none", description="Lowest degree level the role asks for")
    degree_fields: list[str] = Field(default_factory=list, description="Fields of study the role prefers, if any")


class SkillEvidence(BaseModel):
    skill: str = Field(description="The skill exactly as written in the requirement list")
    evidence: str = Field(description="Short phrase from the resume showing it, copied word for word")


class CandidateFacts(BaseModel):
    name: str = Field(description="Candidate's full name as written on the resume")
    headline: str = Field(description="One sentence describing the candidate's background")
    years_experience: float = Field(description="Years of paid, non-internship work experience; 0 for students or freshers")
    relevant_years: float = Field(description="Of those years, how many were in roles similar to the target role")
    skills: list[SkillEvidence] = Field(description="Requirement-list skills the resume shows, each with evidence")
    degree: DegreeLevel = Field(description="Highest completed degree")
    degree_field: str = Field(description="Field of that degree, e.g. 'Computer Science'; empty if none")
    strengths: list[str] = Field(default_factory=list, description="Up to 3 strengths for THIS role, a few words each")
    gaps: list[str] = Field(default_factory=list, description="Up to 3 gaps or concerns for THIS role, a few words each")
