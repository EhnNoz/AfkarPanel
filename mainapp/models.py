from django.db import models


class SendInfo(models.Model):
    subject=models.CharField('موضوع',max_length=25, blank=False, null=False)

    PLATFORM = (
        ('tweeter', 'توییتر'),
        ('instagram', 'اینستاگرام'),
        ('TELEGRAM', 'تلگرام'),
    )

    platform = models.CharField('پلتفرم',max_length=10, choices=PLATFORM, default="tweeter")


class ReportId(models.Model):
    report_id = models.IntegerField('شماره گزارش', blank=False, null=False)


class Document(models.Model):
    CATEGORY_CHOICES = (
        ('intagram', 'Instagram'),
        ('telegram', 'Telegram'),
        ('twitter', 'Twitter')
    )
    platform = models.CharField(verbose_name='پلتفرم',max_length=20, choices=CATEGORY_CHOICES)
    column = models.CharField(verbose_name='نام ستون', max_length=30, blank=False, null=False)
    req_file = models.FileField(verbose_name='بارگذاری فایل',upload_to='documents/')
    uploaded_at = models.DateTimeField(auto_now_add=True)


class NPNDocument(models.Model):
    # CATEGORY_CHOICES = (
    #     ('intagram', 'Instagram'),
    #     ('telegram', 'Telegram'),
    #     ('twitter', 'Twitter')
    # )
    # platform = models.CharField(verbose_name='پلتفرم',max_length=20, choices=CATEGORY_CHOICES)
    column = models.CharField(verbose_name='نام ستون', max_length=30, blank=False, null=False)
    req_file = models.FileField(verbose_name='بارگذاری فایل',upload_to='documents/')
    uploaded_at = models.DateTimeField(auto_now_add=True)