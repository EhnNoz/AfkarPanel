from django import forms
from .models import *


class CreateInfo(forms.ModelForm):
    class Meta:
        model = SendInfo
        fields = ['subject', 'platform']

class ReportInfo(forms.ModelForm):
    class Meta:
        model = ReportId
        fields = ['report_id']




class DocumentForm(forms.ModelForm):
    class Meta:
        model = Document
        fields = '__all__'


class NPNDocumentForm(forms.ModelForm):
    class Meta:
        model = NPNDocument
        fields = '__all__'