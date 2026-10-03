from django.contrib import messages
from django.contrib.auth import login
from django.contrib.auth.decorators import login_required
from django.shortcuts import get_object_or_404, redirect, render
from django.views.decorators.http import require_POST

from heartml.predictor import ModelNotTrained, load_metrics, predict_risk
from heartml.schema import FEATURES

from . import insights
from .forms import PredictionForm, ProfileForm, RegisterForm
from .models import Patient, Prediction


def _patient(user) -> Patient:
    patient, _ = Patient.objects.get_or_create(user=user)
    return patient


def home(request):
    latest = None
    if request.user.is_authenticated:
        latest = _patient(request.user).predictions.first()
    return render(request, "home.html", {"latest": latest})


def register(request):
    if request.user.is_authenticated:
        return redirect("home")
    form = RegisterForm(request.POST or None)
    if request.method == "POST" and form.is_valid():
        user = form.save()
        login(request, user)
        messages.success(request, "Your account is ready. Enter your measurements to get started.")
        return redirect("predict")
    return render(request, "registration/register.html", {"form": form})


@login_required
def profile(request):
    patient = _patient(request.user)
    form = ProfileForm(request.POST or None, instance=patient)
    editing = request.method == "POST" or request.GET.get("edit")
    if request.method == "POST" and form.is_valid():
        form.save()
        messages.success(request, "Your details were saved.")
        return redirect("profile")
    return render(request, "profile.html", {"patient": patient, "form": form, "editing": bool(editing)})


@login_required
def predict(request):
    patient = _patient(request.user)
    if request.method == "POST":
        form = PredictionForm(request.POST)
        if form.is_valid():
            try:
                result = predict_risk({k: float(v) for k, v in form.cleaned_data.items() if k in FEATURES})
            except ModelNotTrained as exc:
                messages.error(request, str(exc))
            else:
                prediction = form.save(commit=False)
                prediction.patient = patient
                prediction.probability = result["probability"]
                prediction.model_name = result["model_name"]
                prediction.save()
                return redirect("result", pk=prediction.pk)
    else:
        initial = {}
        if patient.age and 18 <= patient.age <= 100:
            initial["age"] = patient.age
        if patient.sex_code is not None:
            initial["sex"] = patient.sex_code
        form = PredictionForm(initial=initial)
    return render(request, "predict.html", {"form": form})


@login_required
def result(request, pk):
    prediction = get_object_or_404(Prediction, pk=pk, patient=_patient(request.user))
    return render(request, "result.html", {"p": prediction, "flags": insights.flags(prediction)})


@login_required
def history(request):
    items = list(_patient(request.user).predictions.all())
    return render(request, "history.html", {"items": items})


@login_required
@require_POST
def delete_prediction(request, pk):
    get_object_or_404(Prediction, pk=pk, patient=_patient(request.user)).delete()
    messages.success(request, "Check deleted.")
    return redirect("history")


def model_info(request):
    m = load_metrics()
    ctx = {"m": None}
    if m:
        top = max((r["importance"] for r in m["permutation_importance"]), default=0) or 1
        ctx = {
            "m": m,
            "table": m["cv"]["table"],
            "importance": [{**r, "width": max(0, round(100 * r["importance"] / top))}
                           for r in m["permutation_importance"]],
            "sources": m["dataset_report"]["sources"],
            "skipped": m["dataset_report"].get("skipped_files", []),
        }
    return render(request, "model_info.html", ctx)
