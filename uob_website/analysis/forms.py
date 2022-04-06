from django import forms
from django.forms import widgets


from .models import User, AnalysisSelection

class LoginForm(forms.Form):
    user_name = forms.CharField(max_length=20)
    
    # def clean_message(self):
    #     user_id = self.cleaned_data.get("userid")
    #     dbuser = User.objects.filter(user_id = user_id)
    #     print(user_id)
        
    #     if not dbuser:
    #         raise forms.ValidationError("User does not exist in our db!")
    #     return user_id


class AnalysisSelectionForm(forms.Form):
   
    analysisChoices = forms.MultipleChoiceField(
                                                choices=AnalysisSelection.objects.values_list("analysisSelection_id", "analysis_name"),
                                                label="Select Analysis Types:",
                                                initial=[1, ],
                                                widget=widgets.CheckboxSelectMultiple(),
                                                required=True
                                                )

    # analysisChoices = forms.ModelMultipleChoiceField(
    #     queryset = AnalysisSelection.objects.values_list("analysisSelection_id", "analysis_name"),
    # )
