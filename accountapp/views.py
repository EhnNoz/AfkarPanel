from django.shortcuts import render, redirect
from django.shortcuts import HttpResponse
from django.contrib.auth.forms import AuthenticationForm
from django.contrib.auth import login, logout


def hometest(request):
    return HttpResponse('Hello, this is test page')


def loginpage(request):
    if request.method == 'POST':
        form = AuthenticationForm(data=request.POST)
        if form.is_valid():
            user=form.get_user()
            login(request, user)
            return redirect('mainapp:test')
        else:
            return redirect('accountapp:login')

    else:
        form = AuthenticationForm()
    return render(request, 'accounts/login.html', {'form': form})


def logoutpage(request):
    if request.method =='POST':
        logout(request)
        return redirect('accountapp:login')


# Create your views here.
