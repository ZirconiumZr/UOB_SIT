from django.contrib import admin

# Register your models here.
from .models import Audio, AnalysisSelection, User

admin.site.register(Audio)

admin.site.register(AnalysisSelection)

admin.site.register(User)