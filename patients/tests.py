from django.contrib.auth.models import User
from django.test import TestCase
from django.urls import reverse

from heartml.predictor import predict_risk
from patients.models import Patient, Prediction

# Row from the Cleveland data: 63-year-old man, typical angina, no disease recorded.
SAMPLE = dict(age=63, sex=1, cp=0, exang=0, trestbps=145, chol=233, fbs=1,
              restecg=0, thalach=150, oldpeak=2.3, slope=2, ca=0, thal=1)
# Clear disease pattern: asymptomatic, exercise angina, ST depression, 3 vessels, reversible defect.
SICK = dict(age=62, sex=1, cp=3, exang=1, trestbps=150, chol=290, fbs=0,
            restecg=2, thalach=105, oldpeak=3.5, slope=1, ca=3, thal=2)
# Young woman, atypical angina, great exercise capacity, no ST change.
HEALTHY = dict(age=35, sex=0, cp=1, exang=0, trestbps=115, chol=180, fbs=0,
               restecg=0, thalach=185, oldpeak=0.0, slope=0, ca=0, thal=0)


class ModelSanityTests(TestCase):
    def test_direction_is_correct(self):
        """Guards against the inverted-label bug in the Kaggle heart.csv encoding."""
        sick, healthy = predict_risk(SICK)["probability"], predict_risk(HEALTHY)["probability"]
        self.assertGreater(sick, 0.7)
        self.assertLess(healthy, 0.3)


class FlowTests(TestCase):
    def setUp(self):
        self.client.post(reverse("register"), {
            "username": "asha", "first_name": "Asha K", "email": "a@example.com",
            "password1": "Str0ng-pass-123", "password2": "Str0ng-pass-123",
            "date_of_birth": "1970-05-01", "gender": "F", "phone": "999", "city": "Moga"})

    def test_register_creates_patient_and_logs_in(self):
        self.assertTrue(Patient.objects.filter(user__username="asha").exists())
        self.assertEqual(self.client.get(reverse("profile")).status_code, 200)

    def test_pages_require_login(self):
        self.client.logout()
        for name in ("predict", "history", "profile"):
            r = self.client.get(reverse(name))
            self.assertEqual(r.status_code, 302)
            self.assertIn("/login/", r["Location"])

    def test_predict_form_prefills_from_profile(self):
        html = self.client.get(reverse("predict")).content.decode()
        self.assertIn('value="%d"' % Patient.objects.get().age, html)

    def test_predict_saves_and_shows_result(self):
        r = self.client.post(reverse("predict"), SAMPLE)
        self.assertEqual(r.status_code, 302)
        p = Prediction.objects.get()
        self.assertTrue(0 <= p.probability <= 1)
        page = self.client.get(r["Location"])
        self.assertContains(page, "estimated likelihood")
        self.assertContains(page, "Values worth discussing")

    def test_validation_rejects_out_of_range(self):
        r = self.client.post(reverse("predict"), {**SAMPLE, "trestbps": 5000})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(Prediction.objects.count(), 0)

    def test_history_and_privacy(self):
        self.client.post(reverse("predict"), SAMPLE)
        pk = Prediction.objects.get().pk
        self.assertContains(self.client.get(reverse("history")), "Past checks")
        # another user cannot see or delete it
        other = User.objects.create_user("bob", password="x-Pass-12345")
        Patient.objects.create(user=other)
        self.client.force_login(other)
        self.assertEqual(self.client.get(reverse("result", args=[pk])).status_code, 404)
        self.assertEqual(self.client.post(reverse("delete_prediction", args=[pk])).status_code, 404)
        self.assertEqual(Prediction.objects.count(), 1)

    def test_delete_own_prediction(self):
        self.client.post(reverse("predict"), SAMPLE)
        pk = Prediction.objects.get().pk
        self.client.post(reverse("delete_prediction", args=[pk]))
        self.assertEqual(Prediction.objects.count(), 0)

    def test_profile_edit(self):
        self.client.post(reverse("profile"), {"first_name": "New Name", "email": "n@example.com",
                                              "date_of_birth": "1970-05-01", "gender": "F", "phone": "1", "city": "Delhi"})
        self.assertEqual(User.objects.get(username="asha").first_name, "New Name")

    def test_public_pages(self):
        self.client.logout()
        for name in ("home", "login", "register", "model_info"):
            self.assertEqual(self.client.get(reverse(name)).status_code, 200)
