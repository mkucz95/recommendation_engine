import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('abc')
nltk.download('webtext')
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.corpus import nps_chat

class ContentBasedRec(Recommender):
    def __init__(self):
        self.data=[]
    
    stop_words = (set(stopwords.words('english')) | set(nltk.corpus.webtext.words()) | set(nltk.corpus.abc.words()))
    tokens = word_tokenize(x) #split each message into individual words
    lemmatizer = WordNetLemmatizer()
    clean_tokens=[]
    for token in tokens:
        #clean each token from whitespace and punctuation, and conver to
        #root of word ie walking->walk
        clean_token = lemmatizer.lemmatize(token).lower().strip()
        clean_tokens.append(clean_token)
        
    filtered = [word for word in clean_tokens if word not in stop_words and word.isalpha()]
    return filtered


    def make_content_recs(data_id, user_id=True, m=10, df=df):
        '''
        INPUT:
            data_id (str) - id of either user or item
            user_id (bool) - if true, make recs based on user
            m (int) - number of recommendations to give based on term
        OUTPUT:
            recs (list) - list of item ids that are recommended
            rec_names (list) - list of item names that are recommended
            
        Description:
        This content based recommender looks at the items the user has interacted with.
        It goes through each item and using he NLTK library, finds the most common words
        (related to content) throughout all the items.
        
        Based on these most common words, the recommender looks at the sums of words in
        the content of each item, and based on the number of matches as well as the
        general popularity of the item it gives back the best recommendations.
        '''

        if(user_id):
            user_id = data_id
            try:
                #get already read items
                item_ids, _ = get_user_items(user_id)
            except KeyError: #user does not exist
                print('User Doesnt Exist, Recommending Top items')
                recs = get_top_item_ids(m)
                return recs, get_item_names(recs)
        
        else:
            item_ids = data_id
            
        content = df_content[df_content['item_id'].isin(list(map(float, item_ids)))]
        print(content)
        words=[]
        for col in ['doc_full_name', 'doc_description', 'doc_body']:
            tokenized = tokenize(content[col].str.cat(sep=' '))
            words.extend(tokenized)
            
        common_words = pd.value_counts(words).sort_values(ascending=False)[:5].index

        counts = []
        for word in common_words:
            counts.append((df_content.doc_body.str.count(word).fillna(0)+ \
                                df_content.doc_full_name.str.count(word).fillna(0)+ \
                                df_content.doc_body.str.count(word).fillna(0)))
            
        top_matches = pd.DataFrame({'top_matches':pd.concat(counts)})
        
        top_matches['item_id'] = df_content.item_id.astype(float)
        item_occurences = pd.DataFrame({'occurences':df.item_id.value_counts()})

        top_matches = top_matches.merge(item_occurences, left_on='item_id', right_index=True)
        top_matches.sort_values(['top_matches', 'occurences'], ascending=False, inplace=True)    

        recs = top_matches.item_id[:m].values.astype(str)
        rec_names = get_item_names(recs)
        
        return recs, rec_names