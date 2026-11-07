"""
StudyForge - turn lectures/documents into notes, flashcards and quizzes.

Stack: Streamlit (UI) + Google Gemini Flash (AI).
Run with:  streamlit run app.py
"""

import importlib
import io
import json
import os
import re
import subprocess
import sys

import streamlit as st

# --------------------------------------------------------------------------- #
# Page config (must be the first Streamlit call)
# --------------------------------------------------------------------------- #
st.set_page_config(page_title="StudyForge", page_icon="📚", layout="wide")


# --------------------------------------------------------------------------- #
# Dependency handling: install a missing package at runtime if needed
# --------------------------------------------------------------------------- #
def _pip_install(pip_name: str) -> None:
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "--quiet", pip_name]
    )
    importlib.invalidate_caches()


def _ensure(import_name: str, pip_name: str):
    """Import a module, installing its pip package first if it is missing."""
    try:
        return importlib.import_module(import_name)
    except ImportError:
        _pip_install(pip_name)
        return importlib.import_module(import_name)


try:
    from google import genai
    from google.genai import types
except ImportError:
    try:
        with st.spinner("Installing google-genai (first run only)…"):
            _pip_install("google-genai")
        from google import genai
        from google.genai import types
    except Exception as exc:  # noqa: BLE001
        st.error(
            "The `google-genai` package is not installed and automatic install "
            f"failed ({exc}).\n\n"
            "Fix: put a file named `requirements.txt` in the SAME folder/repo "
            "root as this app, containing the line `google-genai`, then reboot "
            "the app. Locally, run `pip install google-genai`."
        )
        st.stop()

DEFAULT_MODEL = "gemini-2.5-flash"
MAX_CHARS = 400_000  # safety cap on text sent to the model


# --------------------------------------------------------------------------- #
# Text extraction
# --------------------------------------------------------------------------- #
def extract_pdf(file) -> str:
    PdfReader = _ensure("pypdf", "pypdf").PdfReader

    reader = PdfReader(file)
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append(f"[Page {i}]\n{text}")
    return "\n\n".join(pages)


def extract_docx(file) -> str:
    Document = _ensure("docx", "python-docx").Document

    doc = Document(file)
    parts = [p.text for p in doc.paragraphs if p.text.strip()]
    for table in doc.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells]
            if any(cells):
                parts.append(" | ".join(cells))
    return "\n".join(parts)


def extract_pptx(file) -> str:
    Presentation = _ensure("pptx", "python-pptx").Presentation

    prs = Presentation(file)
    slides = []
    for i, slide in enumerate(prs.slides, start=1):
        texts = []
        for shape in slide.shapes:
            if getattr(shape, "has_text_frame", False) and shape.has_text_frame:
                for para in shape.text_frame.paragraphs:
                    line = "".join(run.text for run in para.runs).strip()
                    if line:
                        texts.append(line)
        if slide.has_notes_slide and slide.notes_slide.notes_text_frame:
            note = slide.notes_slide.notes_text_frame.text.strip()
            if note:
                texts.append(f"(Speaker notes) {note}")
        if texts:
            slides.append(f"[Slide {i}]\n" + "\n".join(texts))
    return "\n\n".join(slides)


def extract_text_file(file) -> str:
    return file.read().decode("utf-8", errors="ignore")


def extract_any(uploaded) -> str:
    name = uploaded.name.lower()
    data = io.BytesIO(uploaded.getvalue())
    if name.endswith(".pdf"):
        return extract_pdf(data)
    if name.endswith(".docx"):
        return extract_docx(data)
    if name.endswith(".pptx"):
        return extract_pptx(data)
    if name.endswith((".txt", ".md")):
        return extract_text_file(data)
    return ""


# --------------------------------------------------------------------------- #
# Gemini helpers
# --------------------------------------------------------------------------- #
def get_client(api_key: str):
    return genai.Client(api_key=api_key)


def call_gemini(client, model: str, prompt: str, json_mode: bool = False) -> str:
    config = types.GenerateContentConfig(
        temperature=0.3,
        response_mime_type="application/json" if json_mode else "text/plain",
    )
    response = client.models.generate_content(
        model=model, contents=prompt, config=config
    )
    return response.text or ""


def parse_json(raw: str):
    """Parse JSON, tolerating markdown code fences."""
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.IGNORECASE)
    return json.loads(raw)


# --------------------------------------------------------------------------- #
# Prompts
# --------------------------------------------------------------------------- #
def notes_prompt(text: str, style: str, language: str) -> str:
    detail = {
        "Concise": "Keep notes tight: short bullets, only the essentials.",
        "Balanced": "Cover all main ideas with clear explanations and key examples.",
        "Detailed": "Be thorough: explain every concept, include examples, steps, "
        "formulas and edge cases found in the source.",
    }[style]
    return f"""You are an expert study-notes writer. Turn the source material below into
excellent study notes in {language}, formatted in Markdown.

Rules:
- Stay faithful to the source. Do NOT invent facts that are not supported by it.
- {detail}
- Use this structure:
  # Title
  ## TL;DR  (3-5 bullets)
  ## Key Concepts & Definitions  (bold each term)
  ## Detailed Notes  (organized with ## / ### headings that follow the source's flow)
  ## Formulas / Rules / Steps  (only if present in the source)
  ## Examples  (only if present in the source)
  ## Memory Aids  (mnemonics, analogies, or comparison tables to help recall)
  ## Key Takeaways  (the 5-10 things that must be remembered)

SOURCE MATERIAL:
\"\"\"
{text}
\"\"\"
"""


def flashcards_prompt(text: str, n: int, language: str) -> str:
    return f"""You are an expert teacher creating flashcards for memorization, in {language}.
Create exactly {n} question-answer flashcards from the source material below.

Rules:
- Cover the WHOLE source evenly, prioritizing the most important ideas.
- Mix question types: definitions, "why/how" explanations, comparisons, processes/steps,
  and application questions.
- Each card tests ONE idea. Questions must be clear and self-contained.
- Answers must be accurate, complete but short (1-4 sentences), and based only on the source.

Return ONLY a JSON array, no other text, with this shape:
[{{"question": "...", "answer": "..."}}]

SOURCE MATERIAL:
\"\"\"
{text}
\"\"\"
"""


def quiz_prompt(text: str, n: int, difficulty: str, language: str) -> str:
    return f"""You are an expert exam writer, writing in {language}.
Create exactly {n} multiple-choice questions ({difficulty} difficulty) from the source
material below.

Rules:
- Each question has exactly 4 options, with exactly one correct answer.
- Distractors must be plausible, not silly.
- Cover the whole source; base everything only on it.
- Include a short explanation of why the correct answer is right.

Return ONLY a JSON array, no other text, with this shape:
[{{"question": "...", "options": ["A", "B", "C", "D"], "answer_index": 0, "explanation": "..."}}]

SOURCE MATERIAL:
\"\"\"
{text}
\"\"\"
"""


# --------------------------------------------------------------------------- #
# Session state
# --------------------------------------------------------------------------- #
def init_state():
    defaults = {
        "source_text": "",
        "source_names": [],
        "notes": "",
        "cards": [],
        "card_idx": 0,
        "show_answer": False,
        "known": set(),
        "quiz": [],
        "quiz_submitted": False,
        "quiz_version": 0,
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)


def reset_outputs():
    st.session_state.notes = ""
    st.session_state.cards = []
    st.session_state.card_idx = 0
    st.session_state.show_answer = False
    st.session_state.known = set()
    st.session_state.quiz = []
    st.session_state.quiz_submitted = False
    st.session_state.quiz_version += 1


init_state()

# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.title("📚 StudyForge")
    st.caption("Upload → Notes → Flashcards → Quiz")

    api_key = st.text_input(
        "Gemini API key",
        type="password",
        value=os.getenv("GEMINI_API_KEY", ""),
        help="Get a free key at https://aistudio.google.com/apikey",
    )
    model = st.text_input("Model", value=DEFAULT_MODEL)
    language = st.selectbox(
        "Output language",
        ["English", "Urdu", "Arabic", "Hindi", "French", "Spanish", "German"],
    )

    st.divider()
    note_style = st.radio("Notes detail", ["Concise", "Balanced", "Detailed"], index=1)
    n_cards = st.slider("Number of flashcards", 5, 60, 20, step=5)
    n_quiz = st.slider("Number of quiz questions", 5, 30, 10, step=5)
    difficulty = st.selectbox("Quiz difficulty", ["Easy", "Medium", "Hard"], index=1)

# --------------------------------------------------------------------------- #
# Main: upload
# --------------------------------------------------------------------------- #
st.header("Turn any lecture or document into a memorization kit")

uploads = st.file_uploader(
    "Upload lecture slides, PDFs, Word docs or text files",
    type=["pdf", "docx", "pptx", "txt", "md"],
    accept_multiple_files=True,
)

pasted = st.text_area(
    "…or paste lecture text / transcript here (optional)", height=120
)

if st.button("📥 Load material", type="primary"):
    chunks, names, problems = [], [], []
    for f in uploads or []:
        try:
            text = extract_any(f)
        except Exception as e:  # noqa: BLE001
            problems.append(f"{f.name}: could not be read ({e})")
            continue
        if text.strip():
            chunks.append(f"=== {f.name} ===\n{text}")
            names.append(f.name)
        else:
            problems.append(
                f"{f.name}: no text found (scanned PDFs need OCR first)."
            )
    if pasted.strip():
        chunks.append(f"=== Pasted text ===\n{pasted}")
        names.append("Pasted text")

    for p in problems:
        st.warning(p)

    if not chunks:
        st.error("Nothing to process. Upload a file or paste some text.")
    else:
        full = "\n\n".join(chunks)
        if len(full) > MAX_CHARS:
            st.warning(
                f"Material is very long; using the first {MAX_CHARS:,} characters."
            )
            full = full[:MAX_CHARS]
        st.session_state.source_text = full
        st.session_state.source_names = names
        reset_outputs()

if st.session_state.source_text:
    st.success(
        f"Loaded {len(st.session_state.source_names)} source(s): "
        f"{', '.join(st.session_state.source_names)} "
        f"({len(st.session_state.source_text):,} characters)"
    )
else:
    st.info("Load some material to get started.")
    st.stop()

if not api_key:
    st.warning("Enter your Gemini API key in the sidebar to generate content.")
    st.stop()

client = get_client(api_key)

tab_notes, tab_cards, tab_quiz = st.tabs(["📝 Notes", "🃏 Flashcards", "✅ Quiz"])

# --------------------------------------------------------------------------- #
# Notes tab
# --------------------------------------------------------------------------- #
with tab_notes:
    if st.button("Generate notes"):
        with st.spinner("Writing your notes…"):
            try:
                st.session_state.notes = call_gemini(
                    client,
                    model,
                    notes_prompt(st.session_state.source_text, note_style, language),
                )
            except Exception as e:  # noqa: BLE001
                st.error(f"Gemini error: {e}")

    if st.session_state.notes:
        st.markdown(st.session_state.notes)
        st.download_button(
            "⬇️ Download notes (.md)",
            st.session_state.notes,
            file_name="study_notes.md",
            mime="text/markdown",
        )

# --------------------------------------------------------------------------- #
# Flashcards tab
# --------------------------------------------------------------------------- #
def next_card():
    st.session_state.card_idx += 1
    st.session_state.show_answer = False


with tab_cards:
    if st.button("Generate flashcards"):
        with st.spinner("Creating flashcards…"):
            try:
                raw = call_gemini(
                    client,
                    model,
                    flashcards_prompt(st.session_state.source_text, n_cards, language),
                    json_mode=True,
                )
                cards = parse_json(raw)
                cards = [
                    c for c in cards if isinstance(c, dict) and c.get("question")
                ]
                st.session_state.cards = cards
                st.session_state.card_idx = 0
                st.session_state.show_answer = False
                st.session_state.known = set()
            except Exception as e:  # noqa: BLE001
                st.error(f"Could not generate flashcards: {e}")

    cards = st.session_state.cards
    if cards:
        idx = st.session_state.card_idx
        total = len(cards)

        if idx >= total:
            known = len(st.session_state.known)
            st.success(f"Round finished! You knew {known} of {total} cards.")
            missed = [i for i in range(total) if i not in st.session_state.known]
            col1, col2 = st.columns(2)
            if missed and col1.button("🔁 Review only missed cards"):
                st.session_state.cards = [cards[i] for i in missed]
                st.session_state.card_idx = 0
                st.session_state.known = set()
                st.session_state.show_answer = False
                st.rerun()
            if col2.button("🔄 Restart all cards"):
                st.session_state.card_idx = 0
                st.session_state.known = set()
                st.session_state.show_answer = False
                st.rerun()
        else:
            card = cards[idx]
            st.progress(idx / total, text=f"Card {idx + 1} of {total}")
            st.subheader(card["question"])

            if not st.session_state.show_answer:
                if st.button("Show answer"):
                    st.session_state.show_answer = True
                    st.rerun()
            else:
                st.info(card.get("answer", ""))
                c1, c2 = st.columns(2)
                if c1.button("✅ I knew it"):
                    st.session_state.known.add(idx)
                    next_card()
                    st.rerun()
                if c2.button("❌ Need review"):
                    next_card()
                    st.rerun()

        with st.expander("See all Q&A"):
            for i, c in enumerate(cards, start=1):
                st.markdown(f"**{i}. {c['question']}**")
                st.markdown(c.get("answer", ""))
                st.divider()

        st.download_button(
            "⬇️ Download flashcards (.json)",
            json.dumps(cards, ensure_ascii=False, indent=2),
            file_name="flashcards.json",
            mime="application/json",
        )

# --------------------------------------------------------------------------- #
# Quiz tab
# --------------------------------------------------------------------------- #
with tab_quiz:
    if st.button("Generate quiz"):
        with st.spinner("Writing quiz questions…"):
            try:
                raw = call_gemini(
                    client,
                    model,
                    quiz_prompt(
                        st.session_state.source_text, n_quiz, difficulty, language
                    ),
                    json_mode=True,
                )
                quiz = parse_json(raw)
                quiz = [
                    q
                    for q in quiz
                    if isinstance(q, dict)
                    and isinstance(q.get("options"), list)
                    and isinstance(q.get("answer_index"), int)
                    and 0 <= q["answer_index"] < len(q["options"])
                ]
                st.session_state.quiz = quiz
                st.session_state.quiz_submitted = False
                st.session_state.quiz_version += 1
            except Exception as e:  # noqa: BLE001
                st.error(f"Could not generate quiz: {e}")

    quiz = st.session_state.quiz
    if quiz:
        ver = st.session_state.quiz_version
        with st.form(f"quiz_form_{ver}"):
            answers = []
            for i, q in enumerate(quiz):
                answers.append(
                    st.radio(
                        f"{i + 1}. {q['question']}",
                        q["options"],
                        index=None,
                        key=f"q_{ver}_{i}",
                    )
                )
                st.write("")
            submitted = st.form_submit_button("Submit answers")

        if submitted:
            st.session_state.quiz_submitted = True

        if st.session_state.quiz_submitted:
            score = 0
            for i, q in enumerate(quiz):
                chosen = st.session_state.get(f"q_{ver}_{i}")
                correct = q["options"][q["answer_index"]]
                if chosen == correct:
                    score += 1
                    st.success(f"Q{i + 1}: Correct — {correct}")
                else:
                    st.error(
                        f"Q{i + 1}: Your answer: {chosen or 'No answer'} | "
                        f"Correct: {correct}"
                    )
                if q.get("explanation"):
                    st.caption(q["explanation"])
            st.metric("Score", f"{score} / {len(quiz)}")
