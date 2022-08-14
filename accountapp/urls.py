from django.urls import path
from . import views

app_name='accountapp'
urlpatterns=[
    path('test', views.hometest),
    path('', views.loginpage, name='login'),
    path('logout', views.logoutpage, name='logout'),
]