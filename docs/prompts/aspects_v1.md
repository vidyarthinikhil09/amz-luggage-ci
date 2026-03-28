You are extracting aspect-level sentiment from Amazon luggage reviews.

Return ONLY valid JSON that matches this schema:
{
  "aspects": [
    {
      "aspect": "wheels|handle|zipper|material|durability|size|weight|service|other",
      "polarity": "positive|negative|neutral",
      "evidence": "short exact quote from the review",
      "reason": "brief why this is positive/negative"
    }
  ]
}

Rules:
- Evidence must be copied exactly from review text.
- If an aspect is not mentioned, do not include it.
- Prefer concrete issues (e.g., wheel wobble, zipper jam) over vague statements.
