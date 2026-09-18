# Mastery annotation guidelines

Purpose: check whether the silver state labels used to train the model (derived from the tutor's move type) agree with a human judgement of the student's mastery. This is a validity check, not a new training signal.

## What you see (annotation_sheet.csv)

For each of 60 items: the math problem, its reference answer, and the dialogue so far. The dialogue stops right before a tutor turn. The tutor's next message is hidden on purpose. Do not try to guess it, and do not open `annotation_key_DO_NOT_OPEN.csv` until you have finished all items.

## What to rate

For each of the four subskills, rate the student's mastery **at that point in the dialogue**, using only what the student has said so far:

| Subskill | Ask yourself |
|---|---|
| problem_comprehension | Does the student understand what the problem asks and which quantities matter? |
| arithmetic_execution | Does the student carry out calculations correctly? |
| step_sequencing | Does the student order the solution steps sensibly? |
| self_correction | Does the student notice and repair their own mistakes, or respond well when a mistake is pointed out? |

Scale (write the number in the `*_rating` column):

- **1 = low**: clear evidence of a problem with this subskill.
- **2 = medium**: mixed or partial evidence.
- **3 = high**: clear evidence the student handles this subskill.
- **blank** = cannot tell from the dialogue so far (for example, no evidence about self-correction yet). Leave it blank rather than guessing.

## Rules

1. Rate each subskill separately. If the evidence really is the same for all four, say so, but do not copy one rating into all four columns by default.
2. Judge the student, not the tutor.
3. Use the `notes` column for anything ambiguous (optional).
4. Do the items in the order given, in one or two sittings, without looking anything up.
5. If a second person annotates, they must work independently from a fresh copy (`annotation_sheet_B.csv`) without seeing your ratings.

## Afterwards

Save your completed copy as `annotation/annotation_sheet_A.csv` and run:

    python scripts/analyze_annotation.py annotation/annotation_sheet_A.csv

With a second annotator, pass both files to get inter-rater agreement.
