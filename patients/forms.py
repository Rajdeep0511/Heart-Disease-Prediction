from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User
from django.core.validators import MaxValueValidator, MinValueValidator

from .models import Patient, Prediction


class RegisterForm(UserCreationForm):
    first_name = forms.CharField(label="Full name", max_length=80)
    email = forms.EmailField(required=False)
    date_of_birth = forms.DateField(required=False, widget=forms.DateInput(attrs={"type": "date"}))
    gender = forms.ChoiceField(choices=[("", "Select")] + Patient.GENDER, required=False)
    phone = forms.CharField(max_length=20, required=False)
    city = forms.CharField(max_length=80, required=False)

    class Meta(UserCreationForm.Meta):
        model = User
        fields = ("username", "first_name", "email")

    def save(self, commit=True):
        user = super().save(commit=commit)
        if commit:
            Patient.objects.create(
                user=user,
                date_of_birth=self.cleaned_data.get("date_of_birth"),
                gender=self.cleaned_data.get("gender", ""),
                phone=self.cleaned_data.get("phone", ""),
                city=self.cleaned_data.get("city", ""),
            )
        return user


class ProfileForm(forms.ModelForm):
    first_name = forms.CharField(label="Full name", max_length=80)
    email = forms.EmailField(required=False)

    class Meta:
        model = Patient
        fields = ("date_of_birth", "gender", "phone", "city")
        widgets = {"date_of_birth": forms.DateInput(attrs={"type": "date"}, format="%Y-%m-%d")}

    field_order = ("first_name", "email", "date_of_birth", "gender", "phone", "city")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        user = self.instance.user
        self.fields["first_name"].initial = user.first_name
        self.fields["email"].initial = user.email

    def save(self, commit=True):
        patient = super().save(commit=commit)
        user = patient.user
        user.first_name = self.cleaned_data["first_name"]
        user.email = self.cleaned_data["email"]
        user.save(update_fields=["first_name", "email"])
        return patient


class PredictionForm(forms.ModelForm):
    """The 13 clinical inputs. Order and grouping mirror a typical check-up."""

    GROUPS = [
        ("About you", ["age", "sex"]),
        ("Symptoms", ["cp", "exang"]),
        ("Blood pressure and blood tests", ["trestbps", "chol", "fbs"]),
        ("Heart tests", ["restecg", "thalach", "oldpeak", "slope", "ca", "thal"]),
    ]

    class Meta:
        model = Prediction
        fields = ["age", "sex", "cp", "exang", "trestbps", "chol", "fbs",
                  "restecg", "thalach", "oldpeak", "slope", "ca", "thal"]
        labels = {
            "age": "Age (years)", "sex": "Sex", "cp": "Chest pain type",
            "exang": "Chest pain during exercise?", "trestbps": "Resting blood pressure (mmHg)",
            "chol": "Serum cholesterol (mg/dl)", "fbs": "Fasting blood sugar above 120 mg/dl?",
            "restecg": "Resting ECG result", "thalach": "Maximum heart rate reached (bpm)",
            "oldpeak": "ST depression during exercise", "slope": "Slope of the peak-exercise ST segment",
            "ca": "Major vessels coloured by fluoroscopy (0-3)", "thal": "Thallium stress test result",
        }
        help_texts = {
            "trestbps": "Systolic (upper) number, measured at rest.",
            "chol": "Total cholesterol from a blood test.",
            "thalach": "Highest heart rate during a stress test.",
            "oldpeak": "In mm, compared with rest. Use 0 if none.",
            "ca": "From a cardiac catheterisation or fluoroscopy report.",
            "thal": "From a thallium or nuclear stress test report.",
        }

    RANGES = {  # (min, max, step, placeholder)
        "age": (18, 100, None, "e.g. 54"), "trestbps": (80, 230, None, "e.g. 130"),
        "chol": (100, 600, None, "e.g. 240"), "thalach": (60, 230, None, "e.g. 150"),
        "oldpeak": (0, 8, "0.1", "e.g. 1.0"),
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for name, (lo, hi, step, ph) in self.RANGES.items():
            field = self.fields[name]
            field.widget.attrs.update({"min": lo, "max": hi, "placeholder": ph, "inputmode": "decimal"})
            if step:
                field.widget.attrs["step"] = step
            field.validators.extend([MinValueValidator(lo), MaxValueValidator(hi)])
        for field in self.fields.values():
            field.widget.attrs.setdefault("class", "control")

    def grouped(self):
        return [(title, [self[n] for n in names]) for title, names in self.GROUPS]
