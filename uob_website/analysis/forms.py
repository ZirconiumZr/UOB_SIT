from django import forms
from django.forms import widgets
from matplotlib.pyplot import cla


from .models import AnalysisSelection, Upload

class LoginForm(forms.Form):
    user_name = forms.CharField(max_length=20)
    
    # def clean_message(self):
    #     user_id = self.cleaned_data.get("userid")
    #     dbuser = User.objects.filter(user_id = user_id)
    #     print(user_id)
        
    #     if not dbuser:
    #         raise forms.ValidationError("User does not exist in our db!")
    #     return user_id


class UploadModelForm(forms.ModelForm):
    class Meta:
        model = Upload
        fields = ('description', 'document', )
        widgets = {'upload_id':forms.HiddenInput()}


class AnalysisSelectionForm(forms.Form):
    ANALYSIS_CHOICES = AnalysisSelection.objects.values_list("analysisSelection_id", "analysis_name")
    preferred_choices = ['1',]

    # analysisChoices = forms.MultipleChoiceField(
    #                                             choices=ANALYSIS_CHOICES,
    #                                             label="Select Analysis Types:",
    #                                             initial=[1, ],
    #                                             # initial=[c[0] for c in ANALYSIS_CHOICES],
    #                                             # initial=preferred_choices,
    #                                             # initial=[value for value, title in ANALYSIS_CHOICES],
    #                                             widget=forms.CheckboxSelectMultiple(attrs={'class':'form-check-input me-1','name':'cb_inputs'}),
    #                                             required=False
    #                                             )
    analysisChoices = forms.MultipleChoiceField(choices = ANALYSIS_CHOICES,
                                                label="Select Analysis Types:",
                                                initial=[1, ],
                                                widget=forms.CheckboxSelectMultiple(),
                                                required=False)

    # if analysisChoices.widget.attrs.value == "1":
    #     analysisChoices.widget.attrs.update({'class':'form-check-input me-1','disabled':'disabled'})
    # else:
    analysisChoices.widget.attrs.update({'class':'form-check-input me-1'})


class CheckboxSelectMultipleWithDisabledOption(forms.CheckboxSelectMultiple):
    preferred_choices = ['1',]
    def create_option(self, *args, **kwargs):
        options_dict = super().create_option(*args, **kwargs)
        options_dict['attrs']['class']='form-check-input me-1'
        options_dict['attrs']['name']='analysisChoices'
        print('options_dict: ',options_dict['attrs'])
        
        # if options_dict['value'] == '1':
        #     options_dict['attrs']['disabled'] = 'disabled'

        return options_dict

