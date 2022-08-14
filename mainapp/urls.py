from django.urls import path
from . import views

app_name='mainapp'
urlpatterns = [
    path('test', views.inputtest, name='test'),
    path('waiting', views.inputwait, name='wait'),
    path('rec', views.recpage, name='rec'),
    path('recback', views.recback, name='recback'),
    path('bias', views.bias_detect, name='bias'),
    path('simple_sentiment', views.simple_sentiment, name='simple_sentiment'),
    path('multi_sentiment', views.multi_sentiment, name='multi_sentiment'),
    path('simple_orient', views.simple_orient, name='simple_orient'),
    path('multi_orient', views.multi_orient, name='multi_orient'),
    path('calc_hashtag', views.calc_hashtag, name='calc_hashtag'),
    path('calc_emoji', views.calc_emoji, name='calc_emoji'),
    path('complete_report', views.complete_report, name='complete_report'),
    path('calc_keyword', views.calc_keyword, name='calc_keyword'),
]