from django.core.files.storage import FileSystemStorage
from django.shortcuts import render, redirect
from django.shortcuts import HttpResponse
from django.contrib.auth.decorators import login_required
from django.template import RequestContext

from .forms import *
from .models import *
from django.core import serializers
from json import dumps
import pandas as pd
from .self_functions.AIcode import *

# @login_required(login_url="/account/login")
# def home(request):
#     documents = Document.objects.all()
#     return render(request, 'input/test.html', { 'documents': documents })
@login_required(login_url="/account/login")
def inputtest(request):
    if request.method == "POST":
        form = CreateInfo(request.POST)
        if form.is_valid():
            print(request.POST.dict())
            form.save()

            return redirect('mainapp:wait')
    else:
        form = CreateInfo()
    return render(request, 'input/test.html', {'form': form})


@login_required(login_url="/account/login")
def inputwait(request):
    _info = SendInfo.objects.all().last()
    info = getattr(_info, 'id')
    print(info)
    return render(request, 'input/waiting.html',{'info': info})




@login_required(login_url="/account/login")
def recpage(request):
    try:
        query= request.GET.get('q')
        if query:
            form=SendInfo.objects.filter(id=query).values_list('subject', flat=True)
            form=form[0]
        else:
            form=''
    except:
        form='چنین گزارشی وجود ندارد!'
    return render(request, 'input/recpage.html', {'form': form})

@login_required(login_url="/account/login")
def recback(request):
    return redirect('mainapp:test')
    return render(request, 'input/recpage.html')


@login_required(login_url="/account/login")
def bias_detect(request):
    if request.method == 'POST':
        form = DocumentForm(request.POST, request.FILES)
        if form.is_valid():
            req_file = request.FILES['req_file']
            platform = request.POST['platform']
            column = str(request.POST['column'])
            form.save()

            if platform == 'twitter':
                dataset_name = 'twt'
            elif platform == 'intagram':
                dataset_name = 'insta'
            else:
                dataset_name = 'tel'

            dataset_df = pd.read_excel(f'F:\DataSet\party\{dataset_name}.xlsx', index_col=False)
            try:
                input_df = pd.read_excel(f'F:\sourcecode\\tahlilgar\media\documents\{req_file}')
            except ValueError:
                return render(request, 'input/error_temp.html')

            fs = FileSystemStorage()
            filename = fs.save(req_file.name, req_file)
            uploaded_file_url = fs.url(filename)
            print(uploaded_file_url)
            input_df = input_df.rename(columns={column: 'id_name'})

            try:
                merge_df = pd.merge(input_df, dataset_df, on='id_name')
            except KeyError:
                return render(request, 'input/error_temp.html')

            merge_df.to_excel(f'F:\sourcecode\\tahlilgar{uploaded_file_url}')

            return render(request, 'input/output_temp.html', {
                'uploaded_file_url': uploaded_file_url
            })
        else:
            return render(request, 'input/error_temp.html')

            # return redirect('home')
    else:
        form = DocumentForm()
    return render(request, 'input/bias_detect.html', {
        'form': form
    })

@login_required(login_url="/account/login")
def simple_sentiment(request):
    if request.method == 'POST':
        form = NPNDocumentForm(request.POST, request.FILES)
        if form.is_valid():
            req_file = request.FILES['req_file']
            column = str(request.POST['column'])
            column_cap = column
            column_source = column
            form.save()

            try:
                input_df = pd.read_excel(f'F:\sourcecode\\tahlilgar\media\documents\{req_file}')
            except ValueError:
                return render(request, 'input/error_temp.html')

            fs = FileSystemStorage()
            filename = fs.save(req_file.name, req_file)
            uploaded_file_url = fs.url(filename)

            try:
                call_cleantext = CleanText(input_df, column_cap)
                get_pun_list = call_cleantext.clean_punctual()
                tmp_get_ex_emoji = call_cleantext.extract_emojis()
                tmp_get_emoji_list = call_cleantext.convert_emojies()
                tmp_get_norm_list = call_cleantext.normalize_text()
            except:
                return render(request, 'input/error_temp.html')
            try:
                call_sentiment = Sentiment(input_df,tmp_get_norm_list ,1, column_cap, column_source)
                sent_df_1 = call_sentiment.calc_sentiment()
            except:
                return render(request, 'input/error_temp.html')

            sent_df_1.to_excel(f'F:\sourcecode\\tahlilgar{uploaded_file_url}')

            return render(request, 'input/output_temp.html', {
                'uploaded_file_url': uploaded_file_url
            })
        else:
            return render(request, 'input/error_temp.html')

            # return redirect('home')
    else:
        form = NPNDocumentForm()
    return render(request, 'input/bias_detect.html', {
        'form': form
    })


@login_required(login_url="/account/login")
def multi_sentiment(request):
    if request.method == 'POST':
        form = NPNDocumentForm(request.POST, request.FILES)
        if form.is_valid():
            req_file = request.FILES['req_file']
            column = str(request.POST['column'])
            column_cap = column
            column_source = column
            form.save()

            try:
                input_df = pd.read_excel(f'F:\sourcecode\\tahlilgar\media\documents\{req_file}')
            except ValueError:
                return render(request, 'input/error_temp.html')

            fs = FileSystemStorage()
            filename = fs.save(req_file.name, req_file)
            uploaded_file_url = fs.url(filename)

            try:
                call_cleantext = CleanText(input_df, column_cap)
                get_pun_list = call_cleantext.clean_punctual()
                tmp_get_ex_emoji = call_cleantext.extract_emojis()
                tmp_get_emoji_list = call_cleantext.convert_emojies()
                tmp_get_norm_list = call_cleantext.normalize_text()
            except:
                return render(request, 'input/error_temp.html')
            try:
                call_sentiment = Sentiment(input_df,tmp_get_norm_list ,2, column_cap, column_source)
                sent_df_1 = call_sentiment.calc_sentiment()
                del sent_df_1['init_tag_sent']
            except:
                return render(request, 'input/error_temp.html')


            sent_df_1.to_excel(f'F:\sourcecode\\tahlilgar{uploaded_file_url}')

            return render(request, 'input/output_temp.html', {
                'uploaded_file_url': uploaded_file_url
            })
        else:
            return render(request, 'input/error_temp.html')

            # return redirect('home')
    else:
        form = NPNDocumentForm()
    return render(request, 'input/bias_detect.html', {
        'form': form
    })


@login_required(login_url="/account/login")
def simple_orient(request):
    if request.method == 'POST':
        form = NPNDocumentForm(request.POST, request.FILES)
        if form.is_valid():
            req_file = request.FILES['req_file']
            column = str(request.POST['column'])
            column_cap = column
            column_source = column
            form.save()

            try:
                input_df = pd.read_excel(f'F:\sourcecode\\tahlilgar\media\documents\{req_file}')
            except ValueError:
                return render(request, 'input/error_temp.html')

            fs = FileSystemStorage()
            filename = fs.save(req_file.name, req_file)
            uploaded_file_url = fs.url(filename)

            try:
                call_cleantext = CleanText(input_df, column_cap)
                get_pun_list = call_cleantext.clean_punctual()
                tmp_get_ex_emoji = call_cleantext.extract_emojis()
                tmp_get_emoji_list = call_cleantext.convert_emojies()
                tmp_get_norm_list = call_cleantext.normalize_text()
            except:
                return render(request, 'input/error_temp.html')
            try:
                call_orientation = Orientation(input_df,tmp_get_norm_list ,1, column_cap, column_source)
                orient_df_1 = call_orientation.calc_orient()
            except:
                return render(request, 'input/error_temp.html')

            orient_df_1.to_excel(f'F:\sourcecode\\tahlilgar{uploaded_file_url}')

            return render(request, 'input/output_temp.html', {
                'uploaded_file_url': uploaded_file_url
            })
        else:
            return render(request, 'input/error_temp.html')

            # return redirect('home')
    else:
        form = NPNDocumentForm()
    return render(request, 'input/bias_detect.html', {
        'form': form
    })


@login_required(login_url="/account/login")
def multi_orient(request):
    if request.method == 'POST':
        form = NPNDocumentForm(request.POST, request.FILES)
        if form.is_valid():
            req_file = request.FILES['req_file']
            column = str(request.POST['column'])
            column_cap = column
            column_source = column
            form.save()

            try:
                input_df = pd.read_excel(f'F:\sourcecode\\tahlilgar\media\documents\{req_file}')
            except ValueError:
                return render(request, 'input/error_temp.html')

            fs = FileSystemStorage()
            filename = fs.save(req_file.name, req_file)
            uploaded_file_url = fs.url(filename)

            try:
                call_cleantext = CleanText(input_df, column_cap)
                get_pun_list = call_cleantext.clean_punctual()
                tmp_get_ex_emoji = call_cleantext.extract_emojis()
                tmp_get_emoji_list = call_cleantext.convert_emojies()
                tmp_get_norm_list = call_cleantext.normalize_text()
            except:
                return render(request, 'input/error_temp.html')
            try:
                call_orientation = Orientation(input_df,tmp_get_norm_list ,2, column_cap, column_source)
                orient_df_1 = call_orientation.calc_orient()
                del orient_df_1['init_tag_sent']
            except:
                return render(request, 'input/error_temp.html')


            orient_df_1.to_excel(f'F:\sourcecode\\tahlilgar{uploaded_file_url}')

            return render(request, 'input/output_temp.html', {
                'uploaded_file_url': uploaded_file_url
            })
        else:
            return render(request, 'input/error_temp.html')

            # return redirect('home')
    else:
        form = NPNDocumentForm()
    return render(request, 'input/bias_detect.html', {
        'form': form
    })


@login_required(login_url="/account/login")
def calc_emoji(request):
    if request.method == 'POST':
        form = NPNDocumentForm(request.POST, request.FILES)
        if form.is_valid():
            req_file = request.FILES['req_file']
            column = str(request.POST['column'])
            column_cap = column
            form.save()

            try:
                input_df = pd.read_excel(f'F:\sourcecode\\tahlilgar\media\documents\{req_file}')
            except ValueError:
                return render(request, 'input/error_temp.html')
            fs = FileSystemStorage()
            filename = fs.save(req_file.name, req_file)
            uploaded_file_url = fs.url(filename)

            # df = pd.read_excel(f'F:\sourcecode\simple-file-upload-master{uploaded_file_url}')
            try:
                call_emojiyab = Emojiyab(input_df, column_cap)
                count_emoji = call_emojiyab.find_emoji()
            except KeyError:
                return render(request, 'input/error_temp.html')

            # print(count_cat)
            count_emoji.to_excel(f'F:\sourcecode\\tahlilgar{uploaded_file_url}')

            return render(request, 'input/output_temp.html', {
                'uploaded_file_url': uploaded_file_url
            })

    else:
        form = NPNDocumentForm()
    return render(request, 'input/bias_detect.html', {
        'form': form
    })


@login_required(login_url="/account/login")
def calc_hashtag(request):
    if request.method == 'POST':
        form = NPNDocumentForm(request.POST, request.FILES)
        if form.is_valid():
            req_file = request.FILES['req_file']
            column = str(request.POST['column'])
            column_cap = column
            form.save()

            try:
                input_df = pd.read_excel(f'F:\sourcecode\\tahlilgar\media\documents\{req_file}')
            except ValueError:
                return render(request, 'input/error_temp.html')
            fs = FileSystemStorage()
            filename = fs.save(req_file.name, req_file)
            uploaded_file_url = fs.url(filename)

            # df = pd.read_excel(f'F:\sourcecode\simple-file-upload-master{uploaded_file_url}')
            try:
                call_hashtagyab = Hashtagyab(input_df, column_cap)
                count_hash = call_hashtagyab.find_hashtag()
            except KeyError:
                return render(request, 'input/error_temp.html')

            # print(count_cat)
            count_hash.to_excel(f'F:\sourcecode\\tahlilgar{uploaded_file_url}')

            return render(request, 'input/output_temp.html', {
                'uploaded_file_url': uploaded_file_url
            })

    else:
        form = NPNDocumentForm()
    return render(request, 'input/bias_detect.html', {
        'form': form
    })


@login_required(login_url="/account/login")
def calc_keyword(request):
    if request.method == 'POST':
        form = NPNDocumentForm(request.POST, request.FILES)
        if form.is_valid():
            req_file = request.FILES['req_file']
            column = str(request.POST['column'])
            column_cap = column
            form.save()

            try:
                input_df = pd.read_excel(f'F:\sourcecode\\tahlilgar\media\documents\{req_file}')
            except ValueError:
                return render(request, 'input/error_temp.html')
            fs = FileSystemStorage()
            filename = fs.save(req_file.name, req_file)
            uploaded_file_url = fs.url(filename)

            # df = pd.read_excel(f'F:\sourcecode\simple-file-upload-master{uploaded_file_url}')
            # try:
            call_keyword = KeyWord(input_df, column_cap)
            count_keyword = call_keyword.find_keywords()
            # except KeyError:
            #     # print(KeyError)
            #     return render(request, 'input/error_temp.html')

            # print(count_cat)
            count_keyword.to_excel(f'F:\sourcecode\\tahlilgar{uploaded_file_url}')

            return render(request, 'input/output_temp.html', {
                'uploaded_file_url': uploaded_file_url
            })

    else:
        form = NPNDocumentForm()
    return render(request, 'input/bias_detect.html', {
        'form': form
    })


@login_required(login_url="/account/login")
def complete_report(request):
    if request.method == 'POST':
        form = DocumentForm(request.POST, request.FILES)
        if form.is_valid():
            req_file = request.FILES['req_file']
            platform = request.POST['platform']
            try:
                column = str(request.POST['column'])
                column_cap = column.split('&')[0]
                column_source = column.split('&')[1]
            except:
                return render(request, 'input/error_temp.html')
            form.save()

            if platform == 'twitter':
                dataset_name = 'twt'
            elif platform == 'intagram':
                dataset_name = 'insta'
            else:
                dataset_name = 'tel'

            dataset_df = pd.read_excel(f'F:\DataSet\party\{dataset_name}.xlsx', index_col=False)

            try:
                input_df = pd.read_excel(f'F:\sourcecode\\tahlilgar\media\documents\{req_file}')
            except ValueError:
                return render(request, 'input/error_temp.html')

            try:
                call_cleantext = CleanText(input_df, column_cap)
                get_pun_list = call_cleantext.clean_punctual()
                tmp_get_ex_emoji = call_cleantext.extract_emojis()
                tmp_get_emoji_list = call_cleantext.convert_emojies()
                tmp_get_norm_list = call_cleantext.normalize_text()
            except:
                return render(request, 'input/error_temp.html')

            fs = FileSystemStorage()
            filename = fs.save(req_file.name, req_file)
            uploaded_file_url = fs.url(filename)

            try:
                call_sentiment = Sentiment(input_df,tmp_get_norm_list ,2, column_cap, column_source)
                sent_df_1 = call_sentiment.calc_sentiment()
            except:
                return render(request, 'input/error_temp.html')


            sent_df_1 = sent_df_1.rename(columns={column_source: 'id_name'})
            sent_df_1 = sent_df_1.rename(columns={column_cap: 'متن مطلب'})
            pattern = '[A-Za-z0-9]+|:|/|@|#...\S+|[_.)]'
            sent_df_1['clean_text'] = sent_df_1['متن مطلب'].apply(lambda x: re.sub(pattern, '', x))
            try:
                merge_df = pd.merge(sent_df_1, dataset_df, on='id_name')
            except KeyError:
                return render(request, 'input/error_temp.html')

            for row in range(0, len(merge_df)):
                # print(row)
                init_caption = merge_df.loc[row, 'clean_text']
                init_cat = merge_df.loc[row, 'category']
                # init_id = merge_df.loc[row, 'id_name']
                for flag in range(0, len(sent_df_1)):
                    sec_caption = sent_df_1.loc[flag, 'clean_text']

                    if sec_caption == init_caption:
                        sent_df_1.loc[flag, 'category'] = init_cat

            sent_df_1.to_excel(f'F:\sourcecode\\tahlilgar{uploaded_file_url}')

            return render(request, 'input/output_temp.html', {
                'uploaded_file_url': uploaded_file_url
            })
        else:
            return render(request, 'input/error_temp.html')

            # return redirect('home')
    else:
        form = DocumentForm()
    return render(request, 'input/bias_detect.html', {
        'form': form
    })









