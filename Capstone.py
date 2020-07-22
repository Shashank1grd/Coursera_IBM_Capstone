#!/usr/bin/env python
# coding: utf-8

# # First of all we are importing the libraries

# In[2]:


import pandas as pd
import requests
import numpy as np 
import io
from IPython.display import display_html
import folium
#from geopy.geocoders import Nominatim
from sklearn.cluster import KMeans
from pandas.io.json import json_normalize
import matplotlib.cm as cm
import matplotlib.colors as colors


# # Now we define the url and the read the html page and get the data.

# In[3]:


url = "https://en.wikipedia.org/wiki/List_of_postal_codes_of_Canada:_M"


# In[4]:


dfs = pd.read_html(url, header=0)
dfs[0].head(3)


# # After that convert the data in to Dataframe which we get in to a list.

# In[5]:


df = pd.DataFrame(dfs[0])


# In[6]:


df.head(5)


# ### Then the next step is to clean the data in which we get only that data where Borough is assigned. And ignore the rows where Borough is not assigned.

# In[7]:


refined_df = df[df.Borough != 'Not assigned']


# In[8]:



refined_df.head()


# ### Next step is to check that either all Neighbourhoods are assigned or not?

# In[9]:


refined_df[refined_df.Neighbourhood == 'Not assigned']


# ### Now we have to check the shape of our data.

# In[10]:


refined_df.shape


# You can see that our data has 103 rows and 3 columns.

# # You can check one more method of scraping and reading data with bautifull Soup. This is just to check, we already have read data, and clean it.

# In[11]:


from bs4 import BeautifulSoup


# In[12]:


content = requests.get(url).text


# In[13]:


soup = BeautifulSoup(content, 'lxml')


# In[14]:


# print(soup.prettify())


# In[15]:


table = str(soup.table)


# In[16]:


# display_html(table, raw=True)


# # Now you can follow the same procedure of read_html to read table and after that convert it into data frame.

# In[17]:


url_lat_lon = 'http://cocl.us/Geospatial_data'


# In[18]:


lat_lon = pd.read_csv(url_lat_lon)


# In[19]:


lat_lon


# # Now we are merging the two dataframes.
# ## Before we merging the two data frames, we have to clearify that on which attribute we are going to merge the data. So in this case we are merging the data on the basis of Post code, so first we rename the column, and then merge the dataframes.

# In[21]:


refined_df.columns


# In[22]:


lat_lon.columns


# In[23]:


lat_lon.rename(columns={'Post Code' : 'Postal Code'}, inplace=True)


# In[24]:


lat_lon.columns


# In[25]:


dataframe = pd.merge(refined_df,lat_lon, on= 'Postal Code')


# In[26]:


dataframe.head(5)


# # Now we have done that, we have created the dataframe which we reqiured.

# In[27]:


# dataframe.to_csv('CompleteData')


# ## we have saved the data in to a csv file and we can retrieve it at any time.
# ## From here we are going to do the Clustring and plotting the neighbourhood of Canada which Contain Toronto in their neighbourhood

# In[28]:


dataframe['Borough'].value_counts()


# ## Getting all the rows containing Toronto in their their Boroughs.

# In[29]:


neighbourhoods = dataframe[dataframe.Borough.str.contains('Toronto')]


# In[30]:


neighbourhoods.head()


# In[31]:


neighbourhoods['Borough'].value_counts()


# In[32]:


# address = 'Toronto, Ontario'

# geolocator = Nominatim(user_agent="ny_explorer")
# location = geolocator.geocode(address)
# latitude = location.latitude
# longitude = location.longitude
# print('The geograpical coordinate of New York City are {}, {}.'.format(latitude, longitude))


# In[33]:


latitude = 43.651070
longitude = -79.347015


# In[34]:


Toronto_map = folium.Map(location=[latitude, longitude], zoom_start=10)
for lat, lon, borough, neighbourhood in zip(neighbourhoods['Latitude'], neighbourhoods['Longitude'], neighbourhoods['Borough'], neighbourhoods['Neighbourhood']):
    label = '{}, {}'.format(neighbourhood,borough)
    label = folium.Popup(label, parse_html=True)
    folium.CircleMarker([lat,lon],
                       radius=5,
                       popup=label,
                       color='blue',
                       fill=True,
                       fill_color='#3186cc',
                       fill_opacity=0.7,
                       parse_html=False).add_to(Toronto_map)
Toronto_map


# # Using K-Means Clustring for clustring the neighbourhoods of Toronto.

# In[35]:


n = neighbourhoods


# In[36]:


n.head(3)


# In[37]:


toronto_clustring = n.drop(['Postal Code','Borough', 'Neighbourhood'], axis=1)


# In[38]:


toronto_clustring.head(3)


# In[39]:


k=5
toronto_clustering = neighbourhoods.drop(['Postal Code','Borough','Neighbourhood'],axis=1)
kmeans = KMeans(n_clusters = k,random_state=0).fit(toronto_clustering)
kmeans.labels_
neighbourhoods.insert(0, 'Cluster Labels', kmeans.labels_)


# In[40]:


neighbourhoods.head(5)


# In[41]:


map_cluster = folium.Map(location=[latitude, longitude], zoom_start=10)

x = np.arange(k)
ys = [i + x + (i*x)**2 for i in range(k)]
colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))
rainbow = [colors.rgb2hex(i) for i in colors_array]

markers_colors = []
for lat, lon, neighbourhood, cluster in zip(neighbourhoods['Latitude'], neighbourhoods['Longitude'], neighbourhoods['Neighbourhood'], neighbourhoods['Cluster Labels']):
    label = folium.Popup(' Cluster ' + str(cluster), parse_html=True)
    folium.CircleMarker(
        [lat, lon],
        radius=5,
        popup=label,
        color=rainbow[cluster-1],
        fill=True,
        fill_color=rainbow[cluster-1],
        fill_opacity=0.7).add_to(map_cluster)
       
map_cluster


# In[ ]:




