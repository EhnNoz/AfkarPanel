from django.contrib import admin
from . import models

admin.site.register(models.SendInfo)
admin.site.register(models.Document)
admin.site.register(models.NPNDocument)