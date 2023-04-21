import subprocess
from flask import Flask, render_template, request, make_response
import pandas as pd
import sys
import re, numpy as np, pandas as pd
import threading
from pprint import pprint
import gensim, spacy, logging, warnings
from gensim.utils import  simple_preprocess
from gensim.models import CoherenceModel
import matplotlib.pyplot as plt
from nltk.corpus import stopwords, PlaintextCorpusReader
import gensim.corpora as corpora
import seaborn as sns
import matplotlib.colors as mcolors
from matplotlib.ticker import FuncFormatter
from matplotlib import pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from matplotlib.patches import Rectangle
from gensim.parsing.preprocessing import preprocess_string
from gensim.parsing.preprocessing import remove_stopwords
import os
import nltk
nltk.download('stopwords')
stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use', 'not', 'would', 'say', 'could', '_', 'be', 'know', 'good', 'go', 'get', 'do', 'done', 'try', 'many', 'some', 'nice', 'thank', 'think', 'see', 'rather', 'easy', 'easily', 'lot', 'lack', 'make', 'want', 'seem', 'run', 'need', 'even', 'right', 'line', 'even', 'also', 'may', 'take', 'come'])
import numpy as np
import base64
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from io import BytesIO
import io
import base64
from matplotlib.patches import Rectangle
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
app = Flask(__name__)
df = pd.DataFrame()
global new_data
global old_data
global data_words
bigram_mod = None
trigram_mod = None
data_ready = None
def analyze_text(file):
    sheet_names = ['Jan-2022', 'Feb-2022', 'March-2022', 'April-2022', 'May-2022', 'June-2022', 'July-2022', 'Aug-2022',
                   'Sept-2022', 'Oct-2022', 'Nov-2022', 'Dec-2022']
    df = pd.concat([pd.read_excel(file, sheet_name=name) for name in sheet_names], ignore_index=True)
    df.columns = df.columns.str.strip()
    # Clean data
    df['Release Date'] = pd.to_datetime(df['Release Date'], format='%B,%Y')
    df['Month'] = df['Release Date'].dt.month
    df['Year'] = df['Release Date'].dt.year
    df['Feature'] = df['Feature Title'] + ' ' + df['Feature Description']
    df.drop(['Feature Title', 'Feature Description'], axis=1, inplace=True)
    return df
def process_words(texts, stop_words= stop_words, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    texts = [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]
    texts = [bigram_mod[doc] for doc in texts]
    texts = [trigram_mod[bigram_mod[doc]] for doc in texts]
    texts_out = []
    for sent in texts:
        sent_out = []
        for word in sent:
            if word not in stop_words:
                sent_out.append(word)
        texts_out.append(sent_out)
    return texts_out

def format_topics_sentences(ldamodel,corpus, data):
    # Init output
    sent_topics_df = pd.DataFrame()

    # Get main topic in each document
    for i, row_list in enumerate(ldamodel[corpus]):
        row = row_list[0] if ldamodel.per_word_topics else row_list
        # print(row)
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                print(dir(sent_topics_df))
                # convert the below line to concat the data as append is not working
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
                # sent_topics_df = pd.concat([sent_topics_df, pd.Series([int(topic_num), round(prop_topic,4), topic_keywords])], axis=1, ignore_index=True)

            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    # Add original text to the end of the output
    contents = pd.Series(data)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    return(sent_topics_df)


def plot_word_count_distribution(df_dominant_topic):
    cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]
    fig, axes = plt.subplots(2, 2, figsize=(16, 14), dpi=160, sharex=True, sharey=True)
    for i, ax in enumerate(axes.flatten()):
        df_dominant_topic_sub = df_dominant_topic.loc[df_dominant_topic.Dominant_Topic == i, :]
        doc_lens = [len(d) for d in df_dominant_topic_sub.Text]
        ax.hist(doc_lens, bins=1000, color=cols[i])
        ax.tick_params(axis='y', labelcolor=cols[i], color=cols[i])
        sns.kdeplot(doc_lens, color="black", shade=False, ax=ax.twinx())
        ax.set(xlim=(0, 1000), xlabel='Document Word Count')
        ax.set_ylabel('Number of Documents', color=cols[i])
        ax.set_title('Topic: ' + str(i), fontdict=dict(size=16, color=cols[i]))

    fig.tight_layout()
    fig.subplots_adjust(top=0.90)
    plt.xticks(np.linspace(0, 1000, 9))
    fig.suptitle('Distribution of Document Word Counts by Dominant Topic', fontsize=22)
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    img_str = base64.b64encode(image_png).decode("utf-8")
    return img_str

def plot_word_cloud_topics(lda_model):
    cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]  # more colors: 'mcolors.XKCD_COLORS'

    cloud = WordCloud(stopwords=stop_words,
                      background_color='white',
                      width=2500,
                      height=1800,
                      max_words=10,
                      colormap='tab10',
                      color_func=lambda *args, **kwargs: cols[i],
                      prefer_horizontal=1.0)

    topics = lda_model.show_topics(formatted=False)

    fig, axes = plt.subplots(2, 2, figsize=(10, 10), sharex=True, sharey=True)

    for i, ax in enumerate(axes.flatten()):
        fig.add_subplot(ax)
        topic_words = dict(topics[i][1])
        cloud.generate_from_frequencies(topic_words, max_font_size=300)
        plt.gca().imshow(cloud)
        plt.gca().set_title('Topic ' + str(i), fontdict=dict(size=16))
        plt.gca().axis('off')

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.axis('off')
    plt.margins(x=0, y=0)
    plt.tight_layout()
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    img_str = base64.b64encode(image_png).decode("utf-8")
    return img_str

def plot_word_importance(lda_model,data_ready):
    from collections import Counter
    topics = lda_model.show_topics(formatted=False)
    data_flat = [w for w_list in data_ready for w in w_list]
    counter = Counter(data_flat)

    out = []
    for i, topic in topics:
        for word, weight in topic:
            out.append([word, i, weight, counter[word]])

    df = pd.DataFrame(out, columns=['word', 'topic_id', 'importance', 'word_count'])

    # Plot Word Count and Weights of Topic Keywords
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharey=True, dpi=160)
    cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]
    for i, ax in enumerate(axes.flatten()):
        ax.bar(x='word', height="word_count", data=df.loc[df.topic_id == i, :], color=cols[i], width=0.5, alpha=0.3,
               label='Word Count')
        ax_twin = ax.twinx()
        ax_twin.bar(x='word', height="importance", data=df.loc[df.topic_id == i, :], color=cols[i], width=0.2,
                    label='Weights')
        ax.set_ylabel('Word Count', color=cols[i])
        ax_twin.set_ylim(0, 0.030);
        ax.set_ylim(0, 3500)
        ax.set_title('Topic: ' + str(i), color=cols[i], fontsize=16)
        ax.tick_params(axis='y', left=False)
        ax.set_xticklabels(df.loc[df.topic_id == i, 'word'], rotation=30, horizontalalignment='right')
        ax.legend(loc='upper left');
        ax_twin.legend(loc='upper right')

    fig.tight_layout(w_pad=2)
    fig.suptitle('Word Count and Importance of Topic Keywords', fontsize=22, y=1.05)
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    img_str = base64.b64encode(image_png).decode("utf-8")
    return img_str

def topics_per_document(model, corpus, start=0, end=1):
    corpus_sel = corpus[start:end]
    dominant_topics = []
    topic_percentages = []
    for i, corp in enumerate(corpus_sel):
        topic_percs, wordid_topics, wordid_phivalues = model[corp]
        dominant_topic = sorted(topic_percs, key = lambda x: x[1], reverse=True)[0][0]
        dominant_topics.append((i, dominant_topic))
        topic_percentages.append(topic_percs)
    return(dominant_topics, topic_percentages)

def plot_sentence_topic_coloring(lda_model,corpus, start = 0, end = 13):
    corp = corpus[start:end]
    mycolors = [color for name, color in mcolors.TABLEAU_COLORS.items()]

    fig, axes = plt.subplots(end - start, 1, figsize=(20, (end - start) * 0.95), dpi=160)
    axes[0].axis('off')
    for i, ax in enumerate(axes):
        if i > 0:
            corp_cur = corp[i - 1]
            topic_percs, wordid_topics, wordid_phivalues = lda_model[corp_cur]
            word_dominanttopic = [(lda_model.id2word[wd], topic[0]) for wd, topic in wordid_topics]
            ax.text(0.01, 0.5, "Doc " + str(i - 1) + ": ", verticalalignment='center',
                    fontsize=16, color='black', transform=ax.transAxes, fontweight=700)

            # Draw Rectange
            topic_percs_sorted = sorted(topic_percs, key=lambda x: (x[1]), reverse=True)
            ax.add_patch(Rectangle((0.0, 0.05), 0.99, 0.90, fill=None, alpha=1,
                                   color=mycolors[topic_percs_sorted[0][0]], linewidth=2))

            word_pos = 0.06
            for j, (word, topics) in enumerate(word_dominanttopic):
                if j < 14:
                    ax.text(word_pos, 0.5, word,
                            horizontalalignment='left',
                            verticalalignment='center',
                            fontsize=16, color=mycolors[topics],
                            transform=ax.transAxes, fontweight=700)
                    word_pos += .009 * len(word)  # to move the word for the next iter
                    ax.axis('off')
            ax.text(word_pos, 0.5, '. . .',
                    horizontalalignment='left',
                    verticalalignment='center',
                    fontsize=16, color='black',
                    transform=ax.transAxes)

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.suptitle('Sentence Topic Coloring for Documents: ' + str(start) + ' to ' + str(end - 2), fontsize=22, y=0.95,
                 fontweight=700)
    plt.tight_layout()
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    img_str = base64.b64encode(image_png).decode("utf-8")
    return img_str

def plot_no_of_documents(dominant_topics, topic_percentages,lda_model):
    df = pd.DataFrame(dominant_topics, columns=['Document_Id', 'Dominant_Topic'])
    dominant_topic_in_each_doc = df.groupby('Dominant_Topic').size()
    df_dominant_topic_in_each_doc = dominant_topic_in_each_doc.to_frame(name='count').reset_index()

    # Total Topic Distribution by actual weight
    topic_weightage_by_doc = pd.DataFrame([dict(t) for t in topic_percentages])
    df_topic_weightage_by_doc = topic_weightage_by_doc.sum().to_frame(name='count').reset_index()

    # Top 3 Keywords for each Topic
    topic_top3words = [(i, topic) for i, topics in lda_model.show_topics(formatted=False)
                       for j, (topic, wt) in enumerate(topics) if j < 3]

    df_top3words_stacked = pd.DataFrame(topic_top3words, columns=['topic_id', 'words'])
    df_top3words = df_top3words_stacked.groupby('topic_id').agg(', \n'.join)
    df_top3words.reset_index(level=0, inplace=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), dpi=120, sharey=True)
    # Topic Distribution by Dominant Topics
    ax1.bar(x='Dominant_Topic', height='count', data=df_dominant_topic_in_each_doc, width=.5, color='firebrick')
    ax1.set_xticks(range(df_dominant_topic_in_each_doc.Dominant_Topic.unique().__len__()))
    tick_formatter = FuncFormatter(
        lambda x, pos: 'Topic ' + str(x) + '\n' + df_top3words.loc[df_top3words.topic_id == x, 'words'].values[0])
    ax1.xaxis.set_major_formatter(tick_formatter)
    ax1.set_title('Number of Documents by Dominant Topic', fontdict=dict(size=10))
    ax1.set_ylabel('Number of Documents')
    ax1.set_ylim(0, 1000)

    # Topic Distribution by Topic Weights
    ax2.bar(x='index', height='count', data=df_topic_weightage_by_doc, width=.5, color='steelblue')
    ax2.set_xticks(range(df_topic_weightage_by_doc.index.unique().__len__()))
    ax2.xaxis.set_major_formatter(tick_formatter)
    ax2.set_title('Number of Documents by Topic Weightage', fontdict=dict(size=10))
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    img_str = base64.b64encode(image_png).decode("utf-8")
    return img_str

def insight_visualization(data1,user_param2):
    global bigram_mod
    global trigram_mod
    global data_ready
    def sent_to_words(sentences):
        for sent in sentences:
            sent = re.sub('\S*@\S*\s?', '', sent)  # remove emails
            sent = re.sub('\s+', ' ', sent)  # remove newline chars
            sent = re.sub("\'", "", sent)  # remove single quotes
            sent = gensim.utils.simple_preprocess(str(sent), deacc=True)
            yield (sent)
    data = data1.Feature.values.tolist()
    data_words = list(sent_to_words(data))
    bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100)  # higher threshold fewer phrases.
    trigram = gensim.models.Phrases(bigram[data_words], threshold=100)
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)
    data_ready = process_words(data_words)  # processed Text Data!
    id2word = corpora.Dictionary(data_ready)
    corpus = [id2word.doc2bow(text) for text in data_ready]
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,id2word=id2word,num_topics=4,random_state=100,update_every=1,
                    chunksize=10,passes=10,alpha='symmetric',iterations=100,per_word_topics=True)
    df_topic_sents_keywords = format_topics_sentences(lda_model, corpus, data_ready)
    df_dominant_topic = df_topic_sents_keywords.reset_index()
    df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']
    if(user_param2 == "Distribution of Document Word Counts by Dominant Topic"):
        result =  plot_word_count_distribution(df_dominant_topic)
    elif (user_param2 == "Word Clouds of Top N Keywords in Each Topic"):
        result = plot_word_cloud_topics(lda_model)
    elif (user_param2 == "Word Count and Importance of Topic Keywords"):
        result = plot_word_importance(lda_model,data_ready)
    elif (user_param2 == "Sentence Topic Coloring for Documents"):
        result = plot_sentence_topic_coloring(lda_model,corpus)
    elif (user_param2 == "Number of Documents by Topic Weightage"):
        dominant_topics, topic_percentages = topics_per_document(model=lda_model, corpus=corpus, end=-1)
        result = plot_no_of_documents(dominant_topics, topic_percentages,lda_model)
    else:
        result="Invalid UserInput"
    return result


@app.route('/')
def index():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload():
    user_param1 = request.form['dropdown1']
    #filename = request.files['file']
    filename= request.form['dataset']
    user_param2 = request.form['dropdown2']
    if filename == "zoom":
        filepath = 'zoom_dataset.xlsx'
    elif filename == "webex":
        filepath = 'webex_dataset.xlsx'
    df=analyze_text(filepath)
    old_data = df[df['Month'] <= 4]
    new_data = df[df['Month'] > 4]
    if user_param1 == "old":
        result = insight_visualization(old_data,user_param2)
    else:
        result = insight_visualization(new_data,user_param2)
    return render_template('upload.html', result=result, user_param1=user_param1, filename=filename,user_param2=user_param2)


@app.route('/deploy', methods=['POST'])
def deploy():
    os._exit(0)


def run_flask():
    # run flask app with debug mode and port 8000 with host 0.0.0.0
    app.run(debug=True, port=8000, host='0.0.0.0')


if __name__ == '__main__':
    run_flask()
