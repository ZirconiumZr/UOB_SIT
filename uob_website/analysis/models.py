import time
from django.db import models



class Audio(models.Model):
    class Meta:
        managed=False
    audio_id = models.CharField(primary_key=True,max_length=30)
    audio_name = models.CharField(max_length=50,null=False,default='')
    path_orig = models.CharField(max_length=100,null=False,default='')
    audio_name_processed = models.CharField(max_length=50,null=True,default=None)
    path_processed = models.CharField(max_length=100,null=True,default=None)
    upload_filename = models.CharField(max_length=50,null=False,default='')
    upload_file_count = models.IntegerField(null=False,default=1)
    description = models.TextField(default="") #models.CharField(max_length=200,null=True)
    audio_info = models.CharField(max_length=100,null=True)
    create_by = models.CharField(max_length=50,null=False,default="undetect user")
    create_date = models.DateField(null=False,default=time.strftime("%Y-%m-%d"))
    create_time = models.TimeField(null=False,default=time.strftime("%H:%M:%S"))
    update_by = models.CharField(max_length=50,null=False,default="undetect user")
    update_date = models.DateField(null=False,default=time.strftime("%Y-%m-%d"))
    update_time = models.TimeField(null=False,default=time.strftime("%H:%M:%S"))


    def __str__(self):
        return self.audio_name
    


class STTresult(models.Model):
    class Meta:
        unique_together=(('audio_id','slice_id'),)
        managed = False
    
    audio_id = models.ForeignKey(Audio, on_delete=models.CASCADE)# models.CharField(primary_key=True,max_length=30)
    slice_id = models.IntegerField(primary_key=True) 
    start_time = models.FloatField(null=False,default=0)
    end_time = models.FloatField(null=False,default=0)
    duration = models.FloatField(null=False,default=0)
    speaker_label = models.CharField(max_length=20,null=False,default='') 
    text = models.CharField(max_length=5000,null=True)
    slice_name = models.CharField(max_length=50,null=False,default="")
    slice_path = models.CharField(max_length=100,null=False,default="")
    create_by = models.CharField(max_length=50,null=False,default="undetect user")
    create_date = models.DateField(null=False,default=time.strftime("%Y-%m-%d"))
    create_time = models.TimeField(null=False,default=time.strftime("%H:%M:%S"))
    
    def __str__(self):
        return self.slice_name


class User(models.Model):
    user_id = models.CharField(primary_key=True, max_length=30)
    user_name = models.CharField(max_length=50,null=False,unique=True)
    password = models.CharField(max_length=50,null=False,default='123456')
    
    def __str__(self):
        return self.user_name
    

class AnalysisSelection(models.Model):
    class Meta:
        managed=False
    analysisSelection_id = models.AutoField(primary_key=True)
    # analysisSelection_id = models.IntegerField(primary_key=True)
    analysis_name = models.CharField(max_length=50,null=False,default="None")
    
    def __str__(self):
        return self.analysis_name

