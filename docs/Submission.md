# When and How to Submit

## Early Submission Principle

### Submit Immediately After Unit 5 (or Units 1–3)
The moment `scripts/predict.py` runs end-to-end and produces a valid `id, label` CSV, that's a submittable solution. **Do not wait for it to be perfect.**

---

## 10-Submission Strategy

```text
Submission 1    ← After Units 1+2+3 (rule-based only, no LLM)
│                 Goal: Get on leaderboard. Establish baseline F1.
│                 Build time: ~2-3 days
│
Submissions 2-3 ← After Unit 4 (add Laya)
│                 Tune threshold: test 0.5, then 0.65
│
Submissions 4-5 ← Improve parser/extractor based on error analysis
│                 Inspect false positives/negatives; fix edge cases
│
Submissions 6-7 ← Model balancing (Gemini-heavy vs. Laya-heavy)
│
Submissions 8-10← Final tuning & threshold calibration after leaderboard feedback
```

---

## Submission Specifications

According to the competition rules, submissions are **Docker-based**.

### Execution Command
The evaluation platform executes the following command inside your Docker container:

```bash
pip install .
python scripts/predict.py --input <path_to_test.csv> --output <path_to_predictions.csv>
```

### Pre-Submission Checklist
Before submitting each version, ensure you have:
1. `scripts/predict.py` running end-to-end and producing valid predictions.
2. `pyproject.toml` with pinned dependencies.
3. `Dockerfile` that builds and runs cleanly.
4. Submission archive zipped and under the **40 GB** size limit.

> [!NOTE]
> **Failed submissions do not count against your 10 limit:**
> *"Only successful submissions count; failed ones do not consume the limit."*
> If your Docker build fails or the script crashes, it will not consume an attempt. Still, **always test your Docker image locally** before uploading.

---

## Key Recommendation

**Do not over-engineer before your first submission.**

Even a trivial baseline (e.g., constant prediction or simple heuristic) is worth testing early just to verify that the Docker pipeline and platform evaluation execute smoothly.

- **First Real Milestone:** Complete **Units 1 + 2 + 3** &rarr; Submit &rarr; Inspect Leaderboard &rarr; Iterate.