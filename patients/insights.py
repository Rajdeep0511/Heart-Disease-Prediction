"""Plain-language flags based on widely used reference ranges.

These are NOT model explanations. They simply point out entered values that
fall outside common reference ranges, so the patient knows what to discuss
with a clinician.
"""


def flags(p) -> list[str]:
    out = []
    if p.trestbps >= 140:
        out.append(f"Resting blood pressure of {p.trestbps} mmHg is in the high range (140 or above).")
    elif p.trestbps >= 130:
        out.append(f"Resting blood pressure of {p.trestbps} mmHg is slightly raised (130-139).")
    if p.chol >= 240:
        out.append(f"Serum cholesterol of {p.chol} mg/dl is high (240 or above).")
    elif p.chol >= 200:
        out.append(f"Serum cholesterol of {p.chol} mg/dl is borderline (200-239).")
    if p.fbs:
        out.append("Fasting blood sugar is above 120 mg/dl.")
    if p.exang:
        out.append("You reported chest pain brought on by exercise.")
    if p.oldpeak >= 2:
        out.append(f"ST depression of {p.oldpeak:g} during exercise is notable.")
    elif p.oldpeak >= 1:
        out.append(f"ST depression of {p.oldpeak:g} during exercise is mildly raised.")
    if p.ca >= 1:
        out.append(f"{p.ca} major vessel(s) were coloured on fluoroscopy.")
    if p.thal in (1, 2):
        out.append("The thallium stress test showed a defect.")
    max_hr = 220 - p.age
    if p.thalach < 0.7 * max_hr:
        out.append(f"Peak heart rate of {p.thalach} bpm is under 70% of the age-predicted maximum (~{max_hr} bpm).")
    return out
