import plotly
import pandas as pd
from flask import Flask, render_template, request, jsonify
from plotly.graph_objs import Bar, Histogram
from Recommender import Recommender
import json

app = Flask(__name__)

#load data
df = pd.read_csv('static/data/user_item_interactions.csv')
df_content = pd.read_csv('static/data/articles_community.csv')
del df['Unnamed: 0']
del df_content['Unnamed: 0']

#load model
recommendation_eng = Recommender(df, df_content)
df = recommendation_eng.df
@app.route('/')
@app.route('/index')
def index():
    #what to do on homepage
    user_interaction_counts = df.groupby('user_id').count()['title']
    content_duplicate_counts = [df_content.duplicated(col).sum() for col in df_content.columns]
    article_counts = df.item_id.value_counts()
    popular_articles = article_counts.sort_values(ascending=False) #op 30 articles
    
    #create visuals
    graphs=[
        {
            'data':[
                Histogram(x=user_interaction_counts, xbins=dict(start=1, end=85, size=5))
            ],
            'layout':{
                'title':'User Activity Distribution',
                'subtitle':'Outliers Removed',
                'yaxis':{
                    'title':'Occurences',
                    'showspikes':"true"
                    },
                'xaxis':{
                    'title':'# Interactions'
                }
            }
        },
        {
            'data':[
                Histogram(x=article_counts)
            ],
            'layout':{
                'title':'Distribution of Article Interactions',
                'yaxis':{
                    'title':'# of Articles',
                    'showspikes':"true"
                    },
                'xaxis':{
                    'title':'# Interactions'
                }
            }
        },
        {
            'data':[
                Bar(
                    x=popular_articles.index,
                    y=popular_articles.values
                )
            ],
            'layout':{
                'title':'Most Popular Articles',
                'yaxis':{'title':'# Interactions'},
                'xaxis':{'title':'Article ID'}
            }
        },
        {
            'data':[
                Bar(
                    x=df_content.columns,
                    y=content_duplicate_counts
                )
            ],
            'layout':{
                'title':'Duplicates In Article Dataset',
                'yaxis':{'title':'article section'},
                'xaxis':{'title':'# duplicates'}
            }
        }
    ]
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    return render_template('index.html', ids=ids, graphJSON=graphJSON)

@app.route('/recommendations')
def recommendations():
    num_recs = request.args.get('num_recs', default=10, type=int)
    user_id = request.args.get('user_id', default=None, type=int)
    title = None
    if(user_id): #do collaborative filtering if user_id is given
        rec_ids, rec_titles = recommendation_eng.user_user_recs(user_id, num_recs)
        title = 'Collaborative Recommendations'
    else: #do rank based recs
        rec_titles = recommendation_eng.get_top_items(num_recs)
        rec_ids = recommendation_eng.get_top_item_ids(num_recs)
        title = 'Rank Based Recommendations'

    recommendations=dict(zip(rec_ids, rec_titles))
    return render_template('recommendations.html', user_id=user_id, num_recs=num_recs,
             title=title,recommendations=recommendations)