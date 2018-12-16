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
                Histogram(y=user_interaction_counts, xbins=dict(start=1, end=85, size=5))
            ],
            'layout':{
                'title':'User Interaction Distribution',
                'subtitle':'Outliers Removed',
                'yaxis':{
                    'title':'Interactions',
                    'showspikes':"true"
                    },
                'xaxis':{
                    'title':'# Articles'
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
        },
    ]
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    return render_template('index.html', ids=ids, graphJSON=graphJSON)

'''@app.route('/recommendations')
def recommendations():
    query = request.args.get('query', '')
    query_type = ''


    return render_template('recommendations.html', query=query, recommendations=recommendations)
'''
def main():
    app.run(host='0.0.0.0', port=3001, debug=True)

if __name__ == '__main__':
    main()