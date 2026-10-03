from datetime import date

from django.contrib.auth.models import User
from django.db import models

YES_NO = [(0, "No"), (1, "Yes")]
SEX = [(0, "Female"), (1, "Male")]
CP = [(0, "Typical angina"), (1, "Atypical angina"), (2, "Non-anginal pain"), (3, "No chest pain (asymptomatic)")]
RESTECG = [(0, "Normal"), (1, "ST-T wave abnormality"), (2, "Left ventricular hypertrophy")]
SLOPE = [(0, "Upsloping"), (1, "Flat"), (2, "Downsloping")]
CA = [(0, "0"), (1, "1"), (2, "2"), (3, "3")]
THAL = [(0, "Normal"), (1, "Fixed defect"), (2, "Reversible defect")]


class Patient(models.Model):
    GENDER = [("F", "Female"), ("M", "Male"), ("O", "Other / prefer not to say")]

    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name="patient")
    date_of_birth = models.DateField(null=True, blank=True)
    gender = models.CharField(max_length=1, choices=GENDER, blank=True)
    phone = models.CharField(max_length=20, blank=True)
    city = models.CharField(max_length=80, blank=True)

    def __str__(self):
        return self.user.get_full_name() or self.user.username

    @property
    def age(self):
        if not self.date_of_birth:
            return None
        today = date.today()
        return today.year - self.date_of_birth.year - (
            (today.month, today.day) < (self.date_of_birth.month, self.date_of_birth.day))

    @property
    def sex_code(self):
        return {"M": 1, "F": 0}.get(self.gender)


class Prediction(models.Model):
    patient = models.ForeignKey(Patient, on_delete=models.CASCADE, related_name="predictions")
    created_at = models.DateTimeField(auto_now_add=True)

    age = models.PositiveSmallIntegerField()
    sex = models.PositiveSmallIntegerField(choices=SEX)
    cp = models.PositiveSmallIntegerField(choices=CP)
    trestbps = models.PositiveSmallIntegerField()
    chol = models.PositiveSmallIntegerField()
    fbs = models.PositiveSmallIntegerField(choices=YES_NO)
    restecg = models.PositiveSmallIntegerField(choices=RESTECG)
    thalach = models.PositiveSmallIntegerField()
    exang = models.PositiveSmallIntegerField(choices=YES_NO)
    oldpeak = models.FloatField()
    slope = models.PositiveSmallIntegerField(choices=SLOPE)
    ca = models.PositiveSmallIntegerField(choices=CA)
    thal = models.PositiveSmallIntegerField(choices=THAL)

    probability = models.FloatField()
    model_name = models.CharField(max_length=60)

    class Meta:
        ordering = ["-created_at"]

    LOW, MODERATE, HIGH = 0.30, 0.60, None

    @property
    def percent(self):
        return round(self.probability * 100)

    @property
    def band(self):
        if self.probability < self.LOW:
            return "low"
        if self.probability < self.MODERATE:
            return "moderate"
        return "high"

    @property
    def band_label(self):
        return {"low": "Lower likelihood", "moderate": "Moderate likelihood", "high": "Higher likelihood"}[self.band]
