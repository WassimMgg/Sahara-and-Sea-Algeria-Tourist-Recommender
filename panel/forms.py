"""
panel/forms.py
==============
ModelForms used by the control panel CRUD pages. Widgets only carry CSS
classes — all styling lives in static/panel/panel.css.
"""

from django import forms
from django.contrib.auth.models import User

from api.models import Attraction, Rating, Visitor

TEXT = {"class": "fld"}
AREA = {"class": "fld", "rows": 6}
SELECT = {"class": "fld"}


class AttractionForm(forms.ModelForm):
    class Meta:
        model = Attraction
        fields = ["id", "name", "city", "region", "category",
                  "place_type", "description", "image_url"]
        widgets = {
            "id": forms.NumberInput(attrs=TEXT),
            "name": forms.TextInput(attrs=TEXT),
            "city": forms.TextInput(attrs=TEXT),
            "region": forms.TextInput(attrs=TEXT),
            "category": forms.TextInput(attrs=TEXT),
            "place_type": forms.TextInput(attrs={**TEXT, "list": "place-types"}),
            "description": forms.Textarea(attrs=AREA),
            "image_url": forms.URLInput(attrs={**TEXT, "id": "id_image_url"}),
        }
        help_texts = {
            "id": "Numeric id — must match the item ids the trained model knows about.",
            "place_type": "Short type used for filtering and recommendations "
                          "(e.g. “Roman Ruins”).",
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.instance.pk:            # the id doubles as the model item id
            self.fields["id"].disabled = True


class VisitorForm(forms.ModelForm):
    class Meta:
        model = Visitor
        fields = ["id", "name", "age", "gender", "home_country"]
        widgets = {
            "id": forms.NumberInput(attrs=TEXT),
            "name": forms.TextInput(attrs=TEXT),
            "age": forms.NumberInput(attrs=TEXT),
            "gender": forms.TextInput(attrs=TEXT),
            "home_country": forms.TextInput(attrs=TEXT),
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.instance.pk:
            self.fields["id"].disabled = True


class RatingForm(forms.ModelForm):
    class Meta:
        model = Rating
        fields = ["attraction", "visitor", "account", "rating", "visit_date"]
        widgets = {
            "attraction": forms.Select(attrs=SELECT),
            "visitor": forms.Select(attrs=SELECT),
            "account": forms.Select(attrs=SELECT),
            "rating": forms.NumberInput(attrs={**TEXT, "min": 1, "max": 5, "step": 0.5}),
            "visit_date": forms.TextInput(attrs={**TEXT, "placeholder": "YYYY-MM-DD"}),
        }
        help_texts = {
            "visitor": "Historical rater — leave empty for an app-account rating.",
            "account": "App account — leave empty for a historical rating.",
        }

    def clean(self):
        data = super().clean()
        if not data.get("visitor") and not data.get("account"):
            raise forms.ValidationError(
                "Pick a rater: either a historical visitor or an app account.")
        if data.get("visitor") and data.get("account"):
            raise forms.ValidationError(
                "A rating belongs to one rater — visitor OR account, not both.")
        return data

    def clean_rating(self):
        value = self.cleaned_data["rating"]
        if not 1 <= value <= 5:
            raise forms.ValidationError("Rating must be between 1 and 5.")
        return value


class UserCreateForm(forms.ModelForm):
    password = forms.CharField(
        widget=forms.PasswordInput(attrs=TEXT), min_length=6,
        help_text="At least 6 characters.")

    class Meta:
        model = User
        fields = ["username", "email", "is_staff", "is_superuser"]
        widgets = {
            "username": forms.TextInput(attrs=TEXT),
            "email": forms.EmailInput(attrs=TEXT),
        }
        help_texts = {
            "is_staff": "Staff members can sign in to this panel.",
            "is_superuser": "Full access flag (kept for parity with Django auth).",
        }

    def save(self, commit=True):
        user = super().save(commit=False)
        user.set_password(self.cleaned_data["password"])
        if commit:
            user.save()
        return user


class UserEditForm(forms.ModelForm):
    new_password = forms.CharField(
        required=False, widget=forms.PasswordInput(attrs=TEXT), min_length=6,
        help_text="Leave empty to keep the current password.")

    class Meta:
        model = User
        fields = ["username", "email", "is_active", "is_staff", "is_superuser"]
        widgets = {
            "username": forms.TextInput(attrs=TEXT),
            "email": forms.EmailInput(attrs=TEXT),
        }
        help_texts = {
            "is_active": "Inactive accounts cannot sign in anywhere.",
            "is_staff": "Staff members can sign in to this panel.",
        }

    def save(self, commit=True):
        user = super().save(commit=False)
        if self.cleaned_data.get("new_password"):
            user.set_password(self.cleaned_data["new_password"])
        if commit:
            user.save()
        return user
