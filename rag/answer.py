"""Answers a question from the retrieved chunks, as a Pydantic object with citations."""
from pydantic import BaseModel, Field

from rag.chunking import Chunk

MODEL = "claude-haiku-5-5"

SYSTEM = """אתה עונה על שאלות של לקוחות חברת הרכב "אופק רכב".
ענה רק על סמך הקטעים שבתוך <chunk>. אל תשתמש בידע כללי.
לכל טענה בתשובה, הוסף ציטוט: מזהה הקטע וציטוט מדויק מתוכו, מילה במילה.
אם הקטעים לא מספיקים כדי לענות, החזר insufficient_context=true, בלי ציטוטים.
ענה בעברית, במשפטים קצרים."""


class Citation(BaseModel):
    chunk_id: str = Field(description="The chunk id, exactly as it appears in <chunk id=...>")
    quote: str = Field(description="An exact quote from that chunk, word for word")


class Answer(BaseModel):
    answer: str
    citations: list[Citation]
    insufficient_context: bool


def build_context(chunks: list[Chunk]) -> str:
    return "\n\n".join(f'<chunk id="{c.id}">\n{c.text}\n</chunk>' for c in chunks)


import time
from google import genai
from google.genai import errors


def ask(question, chunks, client=None):
    client = client or genai.Client(http_options={"api_version": "v1beta"})

    for attempt in range(5):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=f"{build_context(chunks)}\n\nשאלה: {question}",
                config={
                    "system_instruction": SYSTEM,
                    "response_mime_type": "application/json",
                    "response_schema": Answer,
                },
            )
            break
        except errors.ClientError as e:
            if e.code != 429 or attempt == 4:
                raise
            time.sleep(15 * (attempt + 1))

    from types import SimpleNamespace   # ייבוא מהספרייה הסטנדרטית, לא מ־google

    um = response.usage_metadata
    usage = SimpleNamespace(
        input_tokens=um.prompt_token_count or 0,
        output_tokens=(um.candidates_token_count or 0) + (um.thoughts_token_count or 0),
    )
    return response.parsed, usage

def verify_citations(answer: Answer, chunks: list[Chunk]) -> list[Citation]:
    def norm(s: str) -> str:
        return " ".join(s.split())          # רווחים, טאבים ושורות חדשות הופכים לרווח בודד

    by_id = {c.id: norm(c.text) for c in chunks}
    bad = []
    for cit in answer.citations:
        text = by_id.get(cit.chunk_id)
        quote = norm(cit.quote)
        if text is None or not quote or quote not in text:
            bad.append(cit)
    return bad
