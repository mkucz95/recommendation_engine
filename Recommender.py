import pandas as pd
import numpy as np

class Recommender(object):
    def __init__(self, df, df_content):
        '''
        Create Recommender Object

        Inputs
            df (pandas DataFrame) - data containing user item interaction information
            df_content (pandas DataFrame) - contains information regarding the items
        '''

        self.df = df
        self.df_content = df_content
        email_encoded = self.email_mapper()
        del self.df['email']
        self.df['user_id'] = email_encoded
        self.df.rename({'article_id':'item_id'}, axis=1, inplace=True) #expand this for more functionality
        self.user_item = self.create_user_item_matrix(df)

    #helper function
    def email_mapper(self):
        coded_dict = dict()
        cter = 1
        email_encoded = []
        
        for val in self.df['email']:
            if val not in coded_dict:
                coded_dict[val] = cter
                cter+=1
            
            email_encoded.append(coded_dict[val])
        return email_encoded

    #RANK BASED RECOMMENDATIONS
    def get_top_items(self, n, df=None):
        '''
        INPUT:
        n - (int) the number of top items to return
        df - (pandas dataframe) df as defined at the top of the notebook 
        
        OUTPUT:
        top_items - (list) A list of the top 'n' item titles 
        
        '''
        if(df is None): df = self.df
        ids = df.item_id.value_counts().sort_values(ascending=False).head(n).index
        top_items = df.loc[~df.item_id.duplicated() & df.item_id.isin(ids)].title.values

        return top_items # Return the top item titles from df (not df_content)

    def get_top_item_ids(self, n, df=None):
        '''
        INPUT:
        n - (int) the number of top items to return
        df - (pandas dataframe) df as defined at the top of the notebook 
        
        OUTPUT:
        top_items (str)- (list) A list of the top 'n' item ids
        
        '''
        if(df is None): df = self.df
        top_items = list(map(str, df.item_id.value_counts().sort_values(ascending=False).head(n).index))
        return top_items # Return the top item ids

    #COLLABORATIVE FILTERING (binary)
    def create_user_item_matrix(self, df):
        '''
        INPUT:
            df - pandas dataframe with item_id, title, user_id columns
        
        OUTPUT:
            user_item - user item matrix 
        
        Description:
        Return a matrix with user ids as rows and item ids on the columns with 1 values where a user interacted with 
        an item and a 0 otherwise
        '''
        mapping = {True:1, False:0}
        
        user_item_df = df.groupby(['user_id', 'item_id'])['title'].max().unstack()
        user_item_df = ~user_item_df.isnull() #True is not empty, false is empty
        user_item = user_item_df.applymap(lambda x: mapping[x])
        
        return user_item # return the user_item matrix 

    def find_similar_users(self, user_id, user_item=None):
        '''
        INPUT:
        user_id - (int) a user_id
        user_item - (pandas dataframe) matrix of users by items: 
                    1's when a user has interacted with an item, 0 otherwise
        
        OUTPUT:
        similar_users - (list) an ordered list where the closest users (largest dot product users)
                        are listed first
        
        Description:
        Computes the similarity of every pair of users based on the dot product
        
        NB: We could also compute the similarity based on kendall tau or
        other distance based similarity measures. However, this recommendation engine has only binary
        interactions, for which dot-product based similarity is sufficient
        
        '''
        if(user_item is None): user_item = self.user_item
        # compute similarity of each user to the provided user
        similarity = user_item.dot(user_item.loc[user_id])
        # sort by similarity
        similarity = similarity.sort_values(ascending=False)
        # remove the own user's id
        similarity.drop(user_id, inplace=True)
        # create list of just the ids
        most_similar_users = list(similarity.index)
        
        return most_similar_users # return a list of the users in order from most to least similar

    def get_item_names(self, item_ids, df=None):
        '''
        INPUT:
        item_ids - (list) a list of item ids
        df - (pandas dataframe) df as defined at the top of the notebook
        
        OUTPUT:
        item_names - (list) a list of item names associated with the list of item ids 
                        (this is identified by the title column)
        '''

        if(df is None): df = self.df
        #first filter only the associated item_ids
        #, then get only the first occurence of each item_id, title and make lists into title.
        #is sorted in same order as item_ids
        item_names = list(df[df.item_id.isin(item_ids)].groupby('item_id')['title'].first())
        
        return item_names # Return the item names associated with list of item ids

    def get_user_items(self, user_id, user_item=None):
        '''
        INPUT:
        user_id - (int) a user id
        user_item - (pandas dataframe) matrix of users by items: 
                    1's when a user has interacted with an item, 0 otherwise
        
        OUTPUT:
        item_ids - (list) a list of the item ids seen by the user
        item_names - (list) a list of item names associated with the list of item ids 
                        (this is identified by the doc_full_name column in df_content)
        
        Description:
        Provides a list of the item_ids and item titles that have been seen by a user
        '''

        if(user_item is None): user_item = self.user_item
        #find the places in which the user_item mtx ==1 for that user
        #look up the columns to get actual item_id
        user_id = int(float(user_id))
        item_ids = list(user_item.columns[np.where(user_item.loc[user_id]==1)])
        #get item name from item id
        item_names = self.get_item_names(item_ids)
        return [str(a_id) for a_id in item_ids], item_names # return the ids and names


    def get_top_sorted_users(self, user_id, df=None, user_item=None):
        '''
        INPUT:
        user_id - (int)
        df - (pandas dataframe) df as defined at the top of the notebook 
        user_item - (pandas dataframe) matrix of users by items: 
                1's when a user has interacted with an item, 0 otherwise
        
                
        OUTPUT:
        neighbors_df - (pandas dataframe) a dataframe with:
                        neighbor_id - is a neighbor user_id
                        similarity - measure of the similarity of each user to the provided user_id
                        num_interactions - the number of items viewed by the user - if a u
                        
        Other Details - sort the neighbors_df by the similarity and then by number of interactions where 
                        highest of each is higher in the dataframe
        
        '''
        if(df is None): df = self.df
        if(user_item is None): user_item = self.user_item

        #find user similarity w/ dot product
        similarity = user_item.dot(user_item.loc[user_id])
        
        # sort by similarity
        similarity = similarity.sort_values(ascending=False).drop(user_id).to_frame(name='similarity').reset_index()

        #get number of interactions for each user
        num_interactions = df.user_id.value_counts().to_frame('num_interactions')
        
        #combine the value counts with similarity
        neighbors_df = similarity.merge(num_interactions, left_on='user_id', 
                            right_index=True).rename(columns={'user_id':'neighbor_id'})

        neighbors_df.sort_values(by=['similarity', 'num_interactions'], ascending=False, inplace=True)
        return neighbors_df # Return the dataframe specified in the doc_string

    def user_user_recs(self, user_id, m):
        '''
        INPUT:
        user_id - (int) a user id
        m - (int) the number of recommendations you want for the user
        
        OUTPUT:
        recs - (list) a list of recommendations for the user by item id
        rec_names - (list) a list of recommendations for the user by item title
        
        Description:
        Loops through the users based on closeness to the input user_id
        For each user - finds items the user hasn't seen before and provides them as recs
        Does this until m recommendations are found
        
        Notes:
        * Choose the users that have the most total item interactions 
        before choosing those with fewer item interactions.

        * Choose items with the items with the most total interactions 
        before choosing those with fewer total interactions. 
    
        '''
        try:
            #get already read items
            user_item_ids, _ = self.get_user_items(user_id)
        except KeyError: #user does not exist
            recs = self.get_top_item_ids(m)
            return recs, get_item_names(recs)
        #get neighbors sorted by similarity (descending)
        neighbours = self.get_top_sorted_users(user_id).neighbor_id.values
        
        #get top 400 items (their ids), if outside of top 400 we dont want to recommend
        all_items_sorted = self.get_top_item_ids(300)
        
        recs = []
        
        for user in neighbours:
            neighbour_item_ids, _ = self.get_user_items(user)
            not_seen = list(set(neighbour_item_ids)-(set(user_item_ids)&set(neighbour_item_ids)))
            
            #sort by highest ranked items, add to list
            not_seen_sorted = list(set(all_items_sorted) &set(not_seen))
            
            recs.extend(not_seen)
            if(len(recs)>=m):
                recs = recs[:m]
                break; #do not add any more
        
        return recs, self.get_item_names(recs)