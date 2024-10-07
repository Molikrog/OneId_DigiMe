
from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User
from tensorflow.python.distribute.distribute_utils import value_container
from .models import IDCheck


from .models import Documents

class SignUpForm(UserCreationForm):
    email = forms.EmailField(label="", widget=forms.TextInput(attrs={'class':'form-control', 'placeholder':'Email Address'}))
    first_name = forms.CharField(label="", max_length=100, widget=forms.TextInput(attrs={'class':'form-control', 'placeholder':'First Name'}))
    last_name = forms.CharField(label="", max_length=100, widget=forms.TextInput(attrs={'class':'form-control', 'placeholder':'Last Name'}))


    class Meta:
        model = User
        fields = ('username', 'first_name', 'last_name', 'email', 'password1', 'password2')


    def __init__(self, *args, **kwargs):
        super(SignUpForm, self).__init__(*args, **kwargs)

        self.fields['username'].widget.attrs['class'] = 'form-control'
        self.fields['username'].widget.attrs['placeholder'] = 'User Name'
        self.fields['username'].label = ''
        self.fields['username'].help_text = '<span class="form-text text-muted"><small>Required. 150 characters or fewer. Letters, digits and @/./+/-/_ only.</small></span>'

        self.fields['password1'].widget.attrs['class'] = 'form-control'
        self.fields['password1'].widget.attrs['placeholder'] = 'Password'
        self.fields['password1'].label = ''
        self.fields['password1'].help_text = '<ul class="form-text text-muted small"><li>Your password can\'t be too similar to your other personal information.</li><li>Your password must contain at least 8 characters.</li><li>Your password can\'t be a commonly used password.</li><li>Your password can\'t be entirely numeric.</li></ul>'

        self.fields['password2'].widget.attrs['class'] = 'form-control'
        self.fields['password2'].widget.attrs['placeholder'] = 'Confirm Password'
        self.fields['password2'].label = ''
        self.fields['password2'].help_text = '<span class="form-text text-muted"><small>Enter the same password as before, for verification.</small></span>'



class DocumentsForm(forms.ModelForm):
    class Meta:
        model = Documents
        fields = ['government_id', 'drivers_license', 'index', 'medical_insurance']
        widgets = {
            'government_id': forms.ClearableFileInput(attrs={'class': 'form-control'}),
            'drivers_license': forms.ClearableFileInput(attrs={'class': 'form-control'}),
            'index': forms.ClearableFileInput(attrs={'class': 'form-control'}),
            'medical_insurance': forms.ClearableFileInput(attrs={'class': 'form-control'}),
        }

class IDCheckForm(forms.Form):
    username = forms.CharField(max_length=255, label="Username")
    picture = forms.ImageField(required=True, label="Upload or Take a Picture")