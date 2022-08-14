import difflib
import pickle
import string
import time
from datetime import datetime
from threading import Thread
import numpy
import itertools
from collections import Counter
import keras
import pandas as pd
import re

from nltk import word_tokenize
from nltk.corpus import stopwords
from parsivar import Normalizer, FindStems
import emoji
from . import emojies
from keras_preprocessing.sequence import pad_sequences
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


class CleanText:
    def __init__(self, data_frame, column_name):
        self.cln_list = data_frame[column_name].tolist()

    def __new__(cls, data_frame, column_name, *args, **kwargs):
        data_frame[column_name] = data_frame[column_name].apply(lambda x: x[:400])
        return super().__new__(cls, *args, **kwargs)

    def clean_punctual(self):
        tmp_lst = list(map(lambda x: re.sub(r'https?:\S*', ' ', x), self.cln_list))
        tmp_lst = list(map(lambda x: re.sub(r'@[A-Za-z0-9]\S+', ' ', x), tmp_lst))
        tmp_lst = list(map(lambda x: re.sub(r'[0-9]\S+', ' ', x), tmp_lst))
        self.cln_list = list(map(lambda x: re.sub(r'#|_|:|/d+', ' ', x), tmp_lst))
        return self.cln_list

    def normalize_text(self):
        normalizer = Normalizer(pinglish_conversion_needed=True)
        cln_list = list(map(lambda x: normalizer.normalize(x), self.cln_list))
        self.cln_list = list(map(lambda x: ''.join(ch for ch, _ in itertools.groupby(x)), cln_list))
        return self.cln_list

    def remove_stop_words(self):
        stop_words = set(stopwords.words('RD_persian_01'))
        self.cln_list = list(map(lambda x: ' '.join([w for w in x.split() if not w in stop_words]), self.cln_list))
        return self.cln_list

    def extract_emojis(self):
        self.cln_list = list(
            map(lambda x: ''.join((' ' + c + ' ') if c in emoji.UNICODE_EMOJI['en'] else c for c in x),
                self.cln_list))

        return self.cln_list

    def convert_emojies(self):
        self.cln_list = list(map(lambda x: emojies.replace(x), self.cln_list))
        return self.cln_list

    def frequency_words(self):
        freq = dict(Counter(" ".join(self.cln_list).split()))
        sort_orders = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        sort_orders = sort_orders[:4000]
        # print(sort_orders)
        print(len(sort_orders))
        most_common_word = [i[0] for i in sort_orders]
        most_common_word = set(most_common_word)
        print(most_common_word)
        # print(len(most_common_word))
        self.cln_list = list(
            map(lambda x: ' '.join([w for w in x.split() if w in most_common_word]), self.cln_list))
        return self.cln_list


# %%

class EncodeText:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def encode_text(self, input_list, max_length):
        # integer encode
        encoded = self.tokenizer.texts_to_sequences(input_list)
        # pad encoded sequences
        padded = pad_sequences(encoded, maxlen=max_length, padding='post')
        return padded


class Sentiment:

    def __init__(self, in_file, in_list, state, in_caption, in_source):
        self.in_file = in_file
        self.in_list = in_list
        self.state = state
        self.in_caption = in_caption
        self.in_source = in_source

    def calc_sentiment(self):

        #  +, -, nu
        with open(r'F:\sourcecode\sentiment_analysis_01\model\3_cat_softmax\cnn+bilstm\CNN_BiLSTM_6_tokenizer.pickle',
                  'rb') as handle:
            init_tokenizer = pickle.load(handle)

        filename = r'F:\sourcecode\sentiment_analysis_01\model\3_cat_softmax\cnn+bilstm\CNN_BiLSTM_6_model.h5'
        init_model = keras.models.load_model(filename)

        # shadi & omid tag
        with open(
                r'F:\sourcecode\sentiment_analysis_01\model\2_cat_softmax_omid_shadi\cnn_bilstm\CNN_BiLSTM_6_tokenizer.pickle',
                'rb') as handle:
            pos_tokenizer = pickle.load(handle)

        filename = r'F:\sourcecode\sentiment_analysis_01\model\2_cat_softmax_omid_shadi\cnn_bilstm\CNN_BiLSTM_6_model.h5'
        pos_model = keras.models.load_model(filename)

        # khashm & gham tag
        with open(
                r'F:\sourcecode\sentiment_analysis_01\model\2_cat_softmax_khashm_gham\cnn_bilstm\CNN_BiLSTM_6_tokenizer.pickle',
                'rb') as handle:
            neg_tokenizer = pickle.load(handle)

        filename = r'F:\sourcecode\sentiment_analysis_01\model\2_cat_softmax_khashm_gham\cnn_bilstm\CNN_BiLSTM_6_model.h5'
        neg_model = keras.models.load_model(filename)

        # tars & taaj tag
        with open(
                r'F:\sourcecode\sentiment_analysis_01\model\3_cat_softmax_tars_taajob\cnn_bilstm\CNN_BiLSTM_6_tokenizer.pickle',
                'rb') as handle:
            nu_tokenizer = pickle.load(handle)

        filename = r'F:\sourcecode\sentiment_analysis_01\model\3_cat_softmax_tars_taajob\cnn_bilstm\CNN_BiLSTM_6_model.h5'
        nu_model = keras.models.load_model(filename)

        # %% function of sent

        def calc_class(my_list, my_tokenizer, my_model, category, max_len):
            # caption, tag = zip(*my_list)
            # max_len = 100
            call_encodetext = EncodeText(my_tokenizer)
            encode_text = call_encodetext.encode_text(my_list, max_len)

            # check omid_shadi
            pre_list = list(
                map(lambda x: numpy.argmax(my_model.predict([x.tolist()], verbose=0), axis=1)[0], encode_text))
            # pre_list = list(map(lambda x: numpy.argmax(x, axis=1)[0], pre_list))
            if category == 'مثبت':
                output_list = list(map(lambda x: 'شادی' if x == 1 else 'امید', pre_list))
            elif category == 'منفی':
                output_list = list(map(lambda x: 'غم' if x == 1 else 'خشم', pre_list))
            else:
                output_list = list(map(lambda x: 'ترس' if x == 0 else 'تعجب' if x == 1 else 'بدون واکنش', pre_list))
            # zip_list = list(map(list, zip(my_list, sent_list)))
            # total_list.append(zip_list)
            print(f'{category}:', datetime.now())
            return output_list

        # %%

        # %% Check +, -, nu

        print(f':احساسات اولیه', datetime.now())
        # Text Encoding
        max_len = 100
        call_encodetext = EncodeText(init_tokenizer)
        encode_text = call_encodetext.encode_text(self.in_list, max_len)
        # Predict File
        lable_list = []
        for item in encode_text:
            item_lable = numpy.argmax(init_model.predict([item.tolist()], verbose=0), axis=1)[0]
            lable_list.append(item_lable)
        print(f':احساسات اولیه', datetime.now())
        final_list_sent = list(map(lambda x: 'منفی' if x == 0 else 'بدون واکنش' if x == 1 else 'مثبت', lable_list))
        print('-------')
        # print(tmp_data_df.columns)
        # print(self.in_list)
        # print(self.in_file[self.in_caption])
        zip_list_sent = list(
            map(list, zip(self.in_list, final_list_sent, self.in_file[self.in_caption], self.in_file[self.in_source])))
        print('-------')
        init_final_df = pd.DataFrame()

        # init_final_df['متن مطلب'] = pos_caption + neg_caption + nu_caption
        # init_final_df['init_tag_sent'] = pos_tag + neg_tag + nu_tag
        init_final_df['final_tag_sent'] = final_list_sent
        init_final_df['متن مطلب'] = self.in_file[self.in_caption]
        init_final_df['id_name'] = self.in_file[self.in_source]
        if self.state == 1:
            del init_final_df['id_name']
            return init_final_df

        else:

            # %% calc multi sent tag

            print(f':احساسات ثانویه', datetime.now())

            # Filter data
            list_neg = list(filter(lambda x: x[1] == 'منفی', zip_list_sent))
            list_nu = list(filter(lambda x: x[1] == 'بدون واکنش', zip_list_sent))
            list_pos = list(filter(lambda x: x[1] == 'مثبت', zip_list_sent))

            pos_caption, pos_tag, pos_main_caption, pos_source = zip(*list_pos)
            final_list_pos = calc_class(pos_caption, pos_tokenizer, pos_model, 'مثبت', 100)
            output_pos = list(map(list, zip(pos_caption, pos_tag, final_list_pos, pos_main_caption, pos_source)))

            neg_caption, neg_tag, neg_main_caption, neg_source = zip(*list_neg)
            final_list_neg = calc_class(neg_caption, neg_tokenizer, neg_model, 'منفی', 100)
            output_neg = list(map(list, zip(neg_caption, neg_tag, final_list_neg, neg_main_caption, neg_source)))

            nu_caption, nu_tag, nu_main_caption, nu_source = zip(*list_nu)
            final_list_nu = calc_class(nu_caption, nu_tokenizer, nu_model, 'خنثی', 100)
            output_nu = list(map(list, zip(nu_caption, nu_tag, final_list_nu, nu_main_caption, nu_source)))

            init_final_df = pd.DataFrame()

            # init_final_df['متن مطلب'] = pos_caption + neg_caption + nu_caption
            init_final_df['init_tag_sent'] = pos_tag + neg_tag + nu_tag
            init_final_df['final_tag_sent'] = final_list_pos + final_list_neg + final_list_nu
            init_final_df[self.in_caption] = pos_main_caption + neg_main_caption + nu_main_caption
            init_final_df[self.in_source] = pos_source + neg_source + nu_source
            # init_final_df.drop_duplicates()
            return init_final_df


class Orientation:

    def __init__(self, in_file, in_list, state, in_caption, in_source):
        self.in_file = in_file
        self.in_list = in_list
        self.state = state
        self.in_caption = in_caption
        self.in_source = in_source

    def calc_orient(self):

        #  in, out
        with open(
                r'F:\sourcecode\sentiment_analysis_01\model\2_cat_softmax_in_out_me\cnn_bilstm\CNN_BiLSTM_6_tokenizer.pickle',
                'rb') as handle:
            init_tokenizer_io = pickle.load(handle)

        filename = r'F:\sourcecode\sentiment_analysis_01\model\2_cat_softmax_in_out_me\cnn_bilstm\CNN_BiLSTM_6_model.h5'
        init_model_io = keras.models.load_model(filename)
        # eslah & osol tag
        with open(
                r'F:\sourcecode\sentiment_analysis_01\model\2_cat_softmax_es_os\cnn_bilstm\CNN_BiLSTM_6_tokenizer.pickle',
                'rb') as handle:
            in_tokenizer = pickle.load(handle)

        filename = r'F:\sourcecode\sentiment_analysis_01\model\2_cat_softmax_es_os\cnn_bilstm\CNN_BiLSTM_6_model.h5'
        in_model = keras.models.load_model(filename)

        # monafegh & moaned tag
        with open(
                r'F:\sourcecode\sentiment_analysis_01\model\4_cat_softmax_4party\cnn_bilstm\CNN_BiLSTM_6_tokenizer.pickle',
                'rb') as handle:
            out_tokenizer = pickle.load(handle)

        filename = r'F:\sourcecode\sentiment_analysis_01\model\4_cat_softmax_4party\cnn_bilstm\CNN_BiLSTM_6_model.h5'
        out_model = keras.models.load_model(filename)

        # %% function of sent

        def calc_class(my_list, my_tokenizer, my_model, category, max_len):
            # caption, tag = zip(*my_list)
            # max_len = 100
            call_encodetext = EncodeText(my_tokenizer)
            encode_text = call_encodetext.encode_text(my_list, max_len)

            # check model
            pre_list = list(
                map(lambda x: numpy.argmax(my_model.predict([x.tolist()], verbose=0), axis=1)[0], encode_text))
            # pre_list = list(map(lambda x: numpy.argmax(x, axis=1)[0], pre_list))
            if category == 'داخلی':
                output_list = list(map(lambda x: 'اصولگرا' if x == 1 else 'اصلاح طلب', pre_list))

            else:
                output_list = list(
                    map(lambda x: 'منافق' if x == 0 else 'معاند' if x == 1 else 'جدایی طلب' if x == 2 else 'سلطنت طلب',
                        pre_list))
            # zip_list = list(map(list, zip(my_list, sent_list)))
            # total_list.append(zip_list)
            print(f'{category}:', datetime.now())
            return output_list

        # %%

        # %% Check +, -, nu

        print(f':سوگيري اولیه', datetime.now())
        # Text Encoding
        max_len = 100
        call_encodetext = EncodeText(init_tokenizer_io)
        encode_text = call_encodetext.encode_text(self.in_list, max_len)
        # Predict File
        lable_list = []
        for item in encode_text:
            item_lable = numpy.argmax(init_model_io.predict([item.tolist()], verbose=0), axis=1)[0]
            lable_list.append(item_lable)
        print(f':احساسات اولیه', datetime.now())
        final_list_sent = list(map(lambda x: 'خارجی' if x == 0 else 'داخلی', lable_list))
        print('-------')
        # print(tmp_data_df.columns)
        # print(self.in_list)
        # print(self.in_file[self.in_caption])
        zip_list_sent = list(
            map(list, zip(self.in_list, final_list_sent, self.in_file[self.in_caption], self.in_file[self.in_source])))
        print('-------')
        init_final_df = pd.DataFrame()

        # init_final_df['متن مطلب'] = pos_caption + neg_caption + nu_caption
        # init_final_df['init_tag_sent'] = pos_tag + neg_tag + nu_tag
        init_final_df['final_tag_sent'] = final_list_sent
        init_final_df['متن مطلب'] = self.in_file[self.in_caption]
        init_final_df['id_name'] = self.in_file[self.in_source]
        if self.state == 1:
            del init_final_df['id_name']
            return init_final_df

        else:

            # %% calc multi sent tag

            print(f':احساسات ثانویه', datetime.now())

            # Filter data
            list_in = list(filter(lambda x: x[1] == 'داخلی', zip_list_sent))
            list_out = list(filter(lambda x: x[1] == 'خارجی', zip_list_sent))

            in_caption, in_tag, in_main_caption, in_source = zip(*list_in)
            final_list_in = calc_class(in_caption, in_tokenizer, in_model, 'داخلی', 100)
            output_in = list(map(list, zip(in_caption, in_tag, final_list_in, in_main_caption, in_source)))

            out_caption, out_tag, out_main_caption, out_source = zip(*list_out)
            final_list_out = calc_class(out_caption, out_tokenizer, out_model, 'خارجی', 60)
            output_out = list(map(list, zip(out_caption, out_tag, final_list_out, out_main_caption, out_source)))

            init_final_df = pd.DataFrame()

            # init_final_df['متن مطلب'] = pos_caption + neg_caption + nu_caption
            init_final_df['init_tag_sent'] = in_tag + out_tag
            init_final_df['final_tag_sent'] = final_list_in + final_list_out
            init_final_df[self.in_caption] = in_main_caption + out_main_caption
            init_final_df[self.in_source] = in_source + out_source
            # init_final_df.drop_duplicates()
            return init_final_df


class Hashtagyab:

    def __init__(self, in_file, in_column):
        self.in_file = in_file
        self.in_column = in_column

    def find_hashtag(self):
        Data_f = self.in_file
        pattern = '#(\w+)'
        Data_f['hashtag'] = Data_f[self.in_column].apply(lambda a: re.findall(pattern, a))
        flatten_list = list(itertools.chain.from_iterable(Data_f['hashtag']))
        w = Counter(flatten_list)
        df = pd.DataFrame(list(w.items()), columns=['Name', 'Value'])
        df = df.sort_values(by='Value', ascending=False)
        return df


class Emojiyab:

    def __init__(self, in_file, in_column):
        self.in_file = in_file
        self.in_column = in_column

    def find_emoji(self):
        Data_f = self.in_file

        Pattren = r'[^\w\⁠s,.]'
        Data_f['clean'] = Data_f[self.in_column].apply(lambda a: re.findall(Pattren, a))

        pattern = r'[\\u200c\\n \\S ُ؛,؟,،⣾⠟⠿⠁ُ' + string.punctuation + ']'
        Data_f['clean'] = Data_f['clean'].apply((lambda a: re.sub(pattern, '', str(a))))

        pattern = r'[ِ️…f1U9َ»«a7d🏻6ّxً♀🌹e👇♂ْ✌ٔ🤝🇷🇮🏼🌺🌱”🌸💐“👈🌚✅🏽👑•٪❌3۔💯💎🍃—⬇💣🥀⭕💫‼🏃🤷✊🌻🌼🎖⬅❗🏾⚠🌷✔💥°🎉💢✍➡👉4🌿❄💀٬🌟ٍ💠🌐ٌ⚘《》💞ٰ٫⛓💅🧚⭐💦☘○–🌜❣▪🥳b💃🌫🎥🦋🍀🏆☝🧘💆🎈🌴︎5🤙💨🌾⏬’🌈8💌🤞🎊🆔🍁🥂🐴⚖👤🏿🌙⚪👨🍻🤥⁉👁🇸🦖🌏🏵›‹🤟🍓‘🧿🐈▫🍔🦍＃╥⚡👹🐕👆☠🐒🎂©🤘🌞🤐🎁🌍⚫🇦🌎⬛❁🦄🎐🤚🇺💍☀✿🌊🐰‟🎵🇪✏؏◀🐦✖🤰◇🌤◍🇫🇹🐱🐹×●🐻༺༻🍾🎄🇾🍭☁🌃🍗🍼🎶🏳♻🎙◔﹏⚜🐍¶﮼⤵🎬🐷👷🍳♾🍑🎞☆❓✓🌘「࿐💮🧸❇≪≫🌧￼💁✂🧵ۖ◾🧛🎋🦁🐺↯🦠♈🧺🎮꒰⑅༚꒱˖🇯🏝🥇◽🤠🤴🤛🤜🍯🍷🥃⛑🎸⚓🐪🎗🥗⚰🥢⃟🎤🎭🥁🐐🐝🏙⚧🐳💤🇵🏮﷼🎦🐏≧◡≦ֶָ֢👾🌠💉🥛🌅🐌🍂🎃🏡﴿﴾🦉☎]'
        Data_f['clean'] = Data_f['clean'].apply((lambda a: re.sub(pattern, '', str(a))))

        # my_list1 = Data_f['clean'].tolist()
        flatten_list = list(itertools.chain.from_iterable(Data_f['clean']))
        w = Counter(flatten_list)
        df = pd.DataFrame(list(w.items()), columns=['emoji', 'Value'])
        # df.sort_values(by=['Value'])
        df = df.sort_values(by='Value', ascending=False)
        # data_with_index = data_with_index.drop(None)
        return df


class KeyWord:
    def __init__(self, in_file, in_column):
        self.in_file = in_file
        self.in_column = in_column

    @staticmethod
    def clean_text(text):
        text = emojies.replace(text)
        text = re.sub("@[A-Za-z0-9]+", '', text)
        text = re.sub(r'http\S+', '', text)
        text = re.sub("&lt;/?.*?&gt;", " &lt;&gt; ", text)
        text = re.sub("(\\d|\\W)+", " ", text)

        #### Convert Finiglish To Persian in Package Parsivar
        my_normilize = Normalizer()
        text = my_normilize.normalize(text)
        text_token = word_tokenize(text)
        text = [word for word in text_token if not word in stopwords.words('RD_persian_04')]
        # text = [word for word in text if len(word) >= 1]
        lmtzr = FindStems()
        text = [lmtzr.convert_to_stem(word) for word in text]
        text = " ".join(text)
        return text

    @staticmethod
    def my_normalizer(text):
        my_normilize = Normalizer()
        text = my_normilize.normalize(text)
        return text

    @staticmethod
    def calc_similarity(df):
        seq = difflib.SequenceMatcher()
        if len(df) < 11:
            max_range = len(df)
        else:
            max_range = 11
        for i in range(0, max_range):
            a = df.loc[i, 'name']
            for j in range(i + 1, len(df)):
                b = df.loc[j, 'name']
                seq.set_seqs(a.lower(), b.lower())
                if seq.ratio() * 100 > 65:
                    print('ok')
                    if len(a) >= len(b):
                        df.loc[j, 'name'] = df.loc[i, 'name']
                        # df.loc[j, 'count'] =df.loc[i, 'count']
                    else:
                        df.loc[i, 'name'] = df.loc[j, 'name']
                        # df.loc[i, 'count'] = df.loc[j, 'count']
                    break
        df.drop_duplicates(subset="name",
                           keep='first', inplace=True)
        df = df.reset_index(drop=True)
        return df

    @staticmethod
    def sort_coo(coo_matrix):
        tuples = zip(coo_matrix.col, coo_matrix.data)
        return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)

    @staticmethod
    def extract_topn_from_vector(feature_names, sorted_items, topn=5):
        """get the feature names and tf-idf score of top n items"""
        # use only topn items from vector
        sorted_items = sorted_items[:topn]
        score_vals = []
        feature_vals = []

        for idx, score in sorted_items:
            fname = feature_names[idx]

            # keep track of feature name and its corresponding score
            score_vals.append(round(score, 3))
            feature_vals.append(feature_names[idx])

        # create a tuples of feature,score
        # results = zip(feature_vals,score_vals)
        results = {}
        for idx in range(len(feature_vals)):
            results[feature_vals[idx]] = score_vals[idx]

        return results

    def print_results(idx, keywords, df):
        # now print the results
        # print("\n=====Title=====")
        # print(df['title'][idx])
        # print("\n=====Abstract=====")
        # print(df['abstract'][idx])
        print("\n===Keywords===")
        for k in keywords:
            print(k, keywords[k])

    def find_keywords(self):

        stop_word = stopwords.words("RD_persian_04")
        self.in_file[self.in_column] = self.in_file[self.in_column].apply(lambda x: KeyWord.my_normalizer(x))

        pattern_1 = r"\u200c"
        self.in_file[self.in_column] = self.in_file[self.in_column].apply((lambda a: re.sub(pattern_1, '', str(a))))

        pattern_2 = r'[ِ️…f1U9َ»«a7d🏻6ّxً♀🌹e👇♂ْ✌ٔ🤝🇷🇮🏼🌺🌱”🌸💐“👈🌚✅🏽👑•٪❌3۔💯💎🍃—⬇💣🥀⭕💫‼🏃🤷✊🌻🌼🎖⬅❗🏾⚠🌷✔💥°🎉💢✍➡👉4🌿❄💀٬🌟ٍ💠🌐ٌ⚘《》💞ٰ٫' \
                    r'⛓💅🧚⭐💦☘○–🌜❣▪🥳b💃🌫🎥🦋🍀🏆☝🧘💆🎈🌴︎5🤙💨🌾⏬’🌈8💌🤞🎊🆔🍁🥂🐴⚖👤🏿🌙⚪👨🍻🤥⁉👁🇸🦖🌏🏵›‹🤟🍓‘🧿🐈▫🍔🦍＃╥⚡👹🐕' \
                    r'👆☠🐒🎂©🤘🌞🤐🎁🌍⚫🇦🌎⬛❁🦄🎐🤚🇺💍☀✿🌊🐰‟🎵🇪✏؏◀🐦✖🤰◇🌤◍🇫🇹🐱🐹×●🐻༺༻🍾🎄🇾🍭☁🌃🍗🍼🎶🏳♻🎙◔﹏⚜🐍¶﮼⤵🎬🐷👷🍳♾🍑🎞☆' \
                    r'❓✓🌘「࿐💮🧸❇≪≫🌧￼💁✂🧵ۖ◾🧛🎋🦁🐺↯🦠♈🧺🎮꒰⑅༚꒱˖🇯🏝🥇◽🤠🤴🤛🤜🍯🍷🥃⛑🎸⚓🐪🎗🥗⚰🥢⃟🎤🎭🥁🐐🐝🏙⚧🐳💤🇵🏮﷼🎦🐏≧◡≦' \
                    r'ֶָ֢👾🌠💉🥛🌅🐌🍂🎃🏡﴿﴾🦉☎]'
        self.in_file['clean'] = self.in_file[self.in_column].apply((lambda a: re.sub(pattern_2, '', str(a))))

        # Delete The Hashtag Expression
        pattern_3 = r"#..\S+"
        self.in_file['clean'] = self.in_file['clean'].apply((lambda a: re.sub(pattern_3, ' ', str(a))))

        # %%
        self.in_file['clean'] = self.in_file['clean'].apply(lambda x: KeyWord.clean_text(x))

        # %%
        pattern_4 = r"\u200c"
        self.in_file['clean'] = self.in_file['clean'].apply((lambda a: re.sub(pattern_4, '', str(a))))
        self.in_file['clean'] = self.in_file['clean'].apply(lambda x: KeyWord.my_normalizer(x))

        # %%
        docs = self.in_file['clean']
        # create a vocabulary of words,
        cv = CountVectorizer(max_df=0.95,  # ignore words that appear in 95% of documents
                             max_features=10000,  # the size of the vocabulary
                             ngram_range=(2, 3)  # vocabulary contains single words, bigrams, trigrams
                             )
        word_count_vector = cv.fit_transform(docs)
        count = (cv.vocabulary_)
        tfidf_transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
        tfidf_transformer.fit(word_count_vector)

        # get feature names
        feature_names = cv.get_feature_names()

        def get_keywords(idx, docs):
            # generate tf-idf for the given document
            tf_idf_vector = tfidf_transformer.transform(cv.transform([docs[idx]]))

            # sort the tf-idf vectors by descending order of scores
            sorted_items = KeyWord.sort_coo(tf_idf_vector.tocoo())

            # extract only the top n; n here is 10
            keywords = KeyWord.extract_topn_from_vector(feature_names, sorted_items, 5)

            return keywords

        # %%
        for i in range(len(self.in_file)):
            keywords = get_keywords(i, docs)
            print(i)
            print(keywords)
            b = (list([keywords.keys()][0]))
            print(type(b))
            self.in_file.loc[i, "key"] = str(b)
            # Data_f1.at[i, "key"] = b  # insert list to DataFram using Method .at

        # %%
        # create List From DataFrame And Covert To Flat List
        import ast
        e = list(self.in_file['key'])
        e2 = list(map(lambda x: ast.literal_eval(x), e))
        from itertools import chain
        newlist = list(chain(*e2))

        # %%
        df_Count = pd.DataFrame()
        set(newlist)
        df_Count['name'] = newlist
        # *********************************
        # f = list(self.in_file['caption'])
        #
        # # 70-19-3
        # # idx=20
        # # keywords=get_keywords(idx, docs)

        # %%
        tmp_list = self.in_file[self.in_column].tolist()
        # dfn = pd.DataFrame()
        # dfn['check'] = self.in_file[self.in_column]

        final_list = []
        for item in newlist:
            print(item)
            i = 0
            for report in tmp_list:
                out = re.search(item, report)
                if out:
                    i += 1
            print(i)
            final_list.append(i)

        df_Count['count'] = final_list

        df1 = df_Count.copy(deep=True)
        df1 = df1[df1['count'] != 0]
        df1 = df1[df1['count'] != 1]
        df1 = df1[df1['count'] != 2]
        df1 = df1.reset_index(drop=True)
        df1.to_excel('F:/check.xlsx')
        df1.drop_duplicates(subset="name",
                            keep='first', inplace=True)
        df1.sort_values("count", inplace=True, ascending=False)
        # df_Count.to_excel('count1.xlsx')
        # df.to_excel('count1.xlsx')
        # Reset Index In DataFranm
        df1 = df1.reset_index(drop=True)

        # %%
        df = KeyWord.calc_similarity(df1)
        df = KeyWord.calc_similarity(df)
        # df = KeyWord.calc_similarity(df)
        df = df[:10]
        return df

        # df.to_excel('count2.xlsx')
