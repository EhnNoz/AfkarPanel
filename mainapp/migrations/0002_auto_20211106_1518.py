# Generated by Django 3.2.8 on 2021-11-06 11:48

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('mainapp', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='ReportId',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('report_id', models.IntegerField(verbose_name='شماره گزارش')),
            ],
        ),
        migrations.AlterField(
            model_name='sendinfo',
            name='platform',
            field=models.CharField(choices=[('tweeter', 'توییتر'), ('instagram', 'اینستاگرام'), ('TELEGRAM', 'تلگرام')], default='tweeter', max_length=10, verbose_name='پلتفرم'),
        ),
        migrations.AlterField(
            model_name='sendinfo',
            name='subject',
            field=models.CharField(max_length=25, verbose_name='موضوع'),
        ),
    ]
