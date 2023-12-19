
from django import forms
from .models import *

class TweetUploadForm(forms.ModelForm):

    class Meta:
        model = Tweet
        fields = ['full_text', 'username', 'idt']
