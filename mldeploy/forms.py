from django import forms
from .models import ImageModel

class ImageForm(forms.ModelForm):
    class Meta:
              model = ImageModel
              fields = "__all__"

    gender = forms.TypedChoiceField(choices=[('Male', 'Male'), ('Female', 'Female')])
    age = forms.IntegerField()
    salary = forms.IntegerField()