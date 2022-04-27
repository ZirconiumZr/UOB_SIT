from datetime import datetime
import json
import os
import threading

from django.http import HttpResponse, Http404, HttpResponseRedirect
from django.shortcuts import redirect, render, get_object_or_404
from django.template import loader
from django.urls import reverse
from django.views import View
from django.views.generic.edit import DeleteView, CreateView
from django.contrib.auth.decorators import login_required
from django.core.files.storage import FileSystemStorage

from uob_website.settings import MEDIA_ROOT

from .forms import AnalysisSelectionForm, UploadModelForm
from .models import AnalysisSelection, Audio, STTresult
from analysis import uob_main, uob_storage, uob_mainprocess, uob_utils
from .uob_init import (
    FLG_REDUCE_NOISE,
    FLG_SPEECH_ENHANCE,
    FLG_SUPER_RES,
    sttModel,
    sdModel,
)

# Create your views here.
@login_required
def main(request):
    num_audios=  Audio.objects.all().count()
    return render(request, "analysis/main.html", {'num_audios':num_audios})
    
def upload(request):
    # context={}
    # return render(request, template_name='analysis/upload.html', context=context)

    # if request.method == 'POST' and request.FILES['myfile']:
    #     myfile = request.FILES['myfile']
    #     fs = FileSystemStorage()
    #     filename = fs.save(myfile.name, myfile)
    #     uploaded_file_url = fs.url(filename)
    #     return render(request, 'analysis/upload.html', {
    #         'uploaded_file_url': uploaded_file_url
    #     })
    # return render(request, 'analysis/upload.html')
    
    if request.method == 'POST':
        upload_id = uob_storage.dbGenerateUploadId()
        uploadForm = UploadModelForm(request.POST, request.FILES, initial={'upload_id': upload_id})
        if uploadForm.is_valid():
            uploadFile = request.FILES['document']
            # file_type = uploadFile.document.url.split('.')[-1]
            
            fs = FileSystemStorage()
            upload_filename = fs.save(uploadFile.name, uploadFile)
            upload_filepath = MEDIA_ROOT.replace("\\","/")
            upload_file_url = fs.url(upload_filename)
            upload_desc = str(uploadForm.cleaned_data.get("description"))
            
            upload_file_count, upload_filetype_general, upzip_audio_name, upzip_audio_path = uob_mainprocess.process_upload_file(upload_filename=upload_filename, upload_filepath=upload_filepath)
            print("view print: count + filetype_general", upload_file_count, upload_filetype_general)

            print('upload_filename: ',upload_filename)
            print('upload_filepath: ',upload_filepath)
            print('upload_file_url: ',upload_file_url)
            print('description: ',upload_desc)
            
            if upload_filetype_general == 'audio':
                audio = Audio()
                audio.audio_id = uob_storage.dbGenerateAudioId()
                audio.audio_name = upzip_audio_name
                audio.path_orig = upzip_audio_path
                audio.upload_filename = upload_filename
                audio.path_upload = upload_filepath
                audio.upload_file_count = upload_file_count
                audio.audio_meta = uob_utils.get_audio_meta(audioname=audio.audio_name, audiopath=audio.path_orig)
                audio.description = upload_desc
                audio.analysis = {}
                audio.save()
            elif upload_filetype_general == 'compressed':
                for unzip_audioname in os.listdir(upzip_audio_path):
                    audio = Audio()
                    audio.audio_id = uob_storage.dbGenerateAudioId()
                    audio.audio_name = unzip_audioname
                    audio.path_orig = upzip_audio_path
                    audio.upload_filename = upload_filename
                    audio.path_upload = upload_filepath
                    audio.upload_file_count = upload_file_count
                    audio.audio_meta = uob_utils.get_audio_meta(audioname=audio.audio_name, audiopath=audio.path_orig)
                    audio.description = upload_desc
                    audio.analysis = {}
                    audio.save()
                    
                
            # return HttpResponseRedirect(reverse('analysis:upload'))
            uploadForm = UploadModelForm()
            return render(request, 'analysis/upload.html', {'upload_file_url': upload_file_url,'uploadForm':uploadForm})

    else:
        uploadForm = UploadModelForm()
        
    return render(request, 'analysis/upload.html', {'uploadForm': uploadForm})


def about(request):
    context={}
    return render(request, template_name='analysis/about.html', context=context)

def history(request):
    if request.method == 'POST':
        ### Single Delete (flag audio.flg_delete as "X")
        if request.POST.get('modal_singleDelete'):
            audio_to_delete = Audio.objects.get(pk=request.POST.get('modal_singleDelete'))
            if audio_to_delete != None:
                audio_to_delete.flg_delete = 'X'
                audio_to_delete.save(update_fields=['flg_delete'])
            
        ### Single Analysis Start
        if request.POST.get('modal_singleStart'):
            audio_to_analysis =  Audio.objects.get(pk=request.POST.get('modal_singleStart'))
            if audio_to_analysis != None:
                # analysis_selection(request, audio_id=audio_to_analysis.audio_id)
                print(audio_to_analysis.audio_meta)
                analysisSelectionForm = AnalysisSelectionForm(request.POST)
                if analysisSelectionForm.is_valid():
                    print('valid checked')
                    analysis_selected = analysisSelectionForm.cleaned_data.get('analysisChoices')
                    print('analysisChoices: ', analysis_selected)
                    choices = AnalysisSelection.objects.all()
                    print(choices)
                    json_analysis_selection = {}
                    
                    ### * Start to Analyze
                    
                    starttime = datetime.now()
                    for i in analysis_selected:
                        i_int = int(i)-1
                        json_analysis_selection[str(i_int)] = choices[i_int].analysis_name  #example: {"0":"SD", "1":"STT"}
                        if choices[i_int].analysis_name == 'SD+STT':
                            print('*'*30)
                            print("SD+STT")
                            sdstt = STTresult.objects.filter(audio_id = audio_to_analysis.audio_id)
                            if not sdstt:
                                # TODO: multi threading, to close Modal and run model at the same time
                                t1 = AnalysisThread(func=uob_main.sd_and_stt, args=(audio_to_analysis, starttime, choices[i_int].analysis_name))
                                t1.start()
                                t1.join()
                                result = t1.get_result()
                                print("*****result*****:",result)
                                # sdstt_df = uob_main.sd_and_stt(audio=audio_to_analysis, starttime=starttime, analysis_name=choices[i_int].analysis_name)
                                # result = sdstt_df
                            else:
                                # raise ValueError("Audio "+audio_to_analysis.audio_id+" has been performed analysis of "+choices[i_int].analysis_name)
                                print("Warning: "+"Audio "+audio_to_analysis.audio_id+" has been performed analysis of "+choices[i_int].analysis_name)
                        if choices[i_int].analysis_name == 'use case 1':
                            print('*'*30)
                            print("use case 1")
                        if choices[i_int].analysis_name == 'use case 3':
                            print('*'*30)
                            print("use case 3")
                    endtime = datetime.now()

                    print('*' * 30,'\n  Finished!!',)
                    print('start from:', starttime) 
                    print('end at:', endtime) 
                    print('duration: ', endtime-starttime)
                    
                    
                    ### * Save to Log Table
                    params = json.dumps({"NR":FLG_REDUCE_NOISE, "SE":FLG_SPEECH_ENHANCE, "SR":FLG_SUPER_RES, "SD":sdModel, "STT":sttModel})
                    analysis_name = json.dumps(json_analysis_selection)
                    message = ''
                    process_time = '{"starttime":"%s", "endtime":"%s", "duration":"%s"}'%(starttime,endtime,endtime-starttime)
                    uob_storage.dbInsertLog(audio_id=audio_to_analysis.audio_id, params=params, analysis_name=analysis_name, process_time=process_time)
                        
                else:
                    print('analysis form not valid')
        
    else:
        print("not post")

    
    
    # if audio_ids != None:
    #     for audio_id in audio_ids:
    #         audio = get_object_or_404(Audio, id=audio_id)
    #         if audio != None:
    #             audio.flg_delete = 'X'
    #             audio.save(update_fields=['flg_delete'])
    
    audioList = Audio.objects.filter(flg_delete=None)
    audioList = audioList.order_by('-create_date','-create_time')
    analysisSelectionForm = AnalysisSelectionForm()
    context = {
        'audioList': audioList,
        'analysisSelectionForm': analysisSelectionForm
    }
    # return HttpResponse(template.render(context, request))
    return render(request, template_name='analysis/history.html', context=context)



@login_required(login_url="/accounts/login/")
def report(request, audio_id):
    audio = get_object_or_404(Audio,pk=audio_id)
    audio_meta = json.loads(audio.audio_meta)
    zip_folder = os.path.splitext(audio.upload_filename)[0]
    sttResult = STTresult.objects.filter(audio_id = audio_id)
    sttResult = sttResult.order_by("slice_id")
    
    context = {'audio': audio, 
               'audio_meta': audio_meta, 
               'zip_folder': zip_folder,
               'sttResult': sttResult
               }
    
    return render(request, 'analysis/report.html', context=context)



def analysis_selection(request, audio_id):
# class AnalysisSelectionView(CreateView):
    
    def get_cancel_url(self):
        url = reverse('analysis:analysis_selection')
        return url
    
    # def analysis_selection(request, audio_id):
    audio = get_object_or_404(Audio,pk=audio_id)
    
    if request.method == 'POST':
        analysisSelectionForm = AnalysisSelectionForm(request.POST) #, initial={'analysisChoices':[2,]}
        print('POST checked')
        if analysisSelectionForm.is_valid():
            # temp = analysisSelectionForm.fields['analysisChoices'].initial
            temp = analysisSelectionForm.cleaned_data.get('analysisChoices')
            print('temp: ', temp)
            print('valid checked')
            # temp = analysisSelectionForm.fields['analysisChoices'].choices
            # temp:  [(1, 'SD+STT'), (2, 'use case 1'), (3, 'use case 3')]

            analysisChoices = request.POST.getlist('cb_inputs')
            print('analysisChoices: ', analysisChoices)

            # TODO: Start to run analysis processes for audio_id = xxx...
            
            
            context = {
                'audio': audio,
                'form': AnalysisSelectionForm(),
            }
            
            # return HttpResponseRedirect(reverse('analysis:history'))  #'/index/?page=2'
            return render(request, 'analysis/analysis_selection.html', context=context)
        else:
            print('not valid')
    
    else:
        analysisSelectionForm = AnalysisSelectionForm()
        # analysisChoices = request.GET.getlist('analysisChoices')
        analysisChoices = analysisSelectionForm.fields['analysisChoices']
        print('analysisChoices: ', analysisChoices)
        print('analysisChoices.widget_attrs: ', analysisChoices.widget_attrs)
        print('analysisChoices.widget: ', analysisChoices.widget)
        print('analysisChoices.widget.attrs: ', analysisChoices.widget.attrs)
        print('analysisChoices.choices: ', analysisChoices.choices)
        print('analysisChoices.widget.attrs: ', analysisChoices.widget.attrs)
        
    context = {
        'audio': audio,
        'form': analysisSelectionForm,
    }
    return render(request, 'analysis/analysis_selection.html',  context)



class AnalysisThread(threading.Thread):

    def __init__(self,func,args=()):
        super(AnalysisThread,self).__init__()
        self.func = func
        self.args = args

    def run(self):
        self.result = self.func(*self.args)

    def get_result(self):
        try:
            return self.result
        except Exception:
            return None