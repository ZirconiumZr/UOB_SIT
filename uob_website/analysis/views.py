from django.http import HttpResponse, Http404, HttpResponseRedirect
from django.shortcuts import render,get_object_or_404
from django.template import loader
from django.urls import reverse
from django.views import View
from django.views.generic.edit import DeleteView, CreateView
from django.contrib.auth.decorators import login_required

from .forms import AnalysisSelectionForm
from .models import Audio, STTresult, AnalysisSelection 

# Create your views here.
@login_required
def main(request):
    num_audios=  Audio.objects.all().count()
    return render(request, "analysis/main.html", {'num_audios':num_audios})
    
def upload(request):
    context={}
    return render(request, template_name='analysis/upload.html', context=context)

def about(request):
    context={}
    return render(request, template_name='analysis/about.html', context=context)

def history(request):
    audioList = Audio.objects.all()
    # template = loader.get_template('analysis/history.html')
    context = {
        'audioList': audioList,
    }
    # return HttpResponse(template.render(context, request))
    return render(request, template_name='analysis/history.html', context=context)



def report(request, audio_id):
    # try:
    #     audio = Audio.objects.get(pk=audio_id)
    # except Audio.DoesNotExist:
    #     raise Http404("Audio does not exist")
    # return HttpResponse("You're looking at report of %s." % audio_id)
    audio = get_object_or_404(Audio,pk=audio_id)
    return render(request, 'analysis/report.html', {'audio': audio})



def analysis_selection(request, audio_id):
# class AnalysisSelectionView(CreateView):
    
    def get_cancel_url(self):
        url = reverse('analysis:analysis_selection')
        return url
    
    # def analysis_selection(request, audio_id):
    audio = get_object_or_404(Audio,pk=audio_id)
    
    if request.method == 'POST':
        analysisSelectionForm = AnalysisSelectionForm(request.POST)
        if analysisSelectionForm.is_valid():
            temp = analysisSelectionForm.cleaned_data.get("analysisChoices")
            print(temp)
            
            # TODO: Start to run analysis processes for audio_id = xxx...
            
            audioList = Audio.objects.all()
            context = {
                'audioList': audioList,
            }
            return HttpResponseRedirect(reverse('analysis:history'))  #'/index/?page=2'
    
    else:
        analysisSelectionForm = AnalysisSelectionForm()
            
    context = {
        'audio': audio,
        'form': analysisSelectionForm,
    }
    return render(request, 'analysis/analysis_selection.html',  context)



