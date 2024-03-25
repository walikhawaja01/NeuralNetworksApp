
import pandas as pd
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from collections import Counter
import re

models_path = 'REPLACE-WITH-YOUR-MODEL'
video_data_combined = pd.DataFrame()
category_dict = {}

# Loop through the files in the 'models' directory
for filename in os.listdir(models_path):
    if filename.endswith('.csv'):
        csv_file_path = os.path.join(models_path, filename)
        video_data = pd.read_csv(csv_file_path, encoding='ISO-8859-1')
        video_data_combined = pd.concat([video_data_combined, video_data], ignore_index=True)
    elif filename.endswith('.json'):
        json_file_path = os.path.join(models_path, filename)
        with open(json_file_path, 'r') as file:
            category_data = json.load(file)
            category_dict.update({int(item['id']): item['snippet']['title'] for item in category_data['items']})

# Map categories
if category_dict:
    video_data_combined['category_name'] = video_data_combined['category_id'].map(category_dict)

# Data Cleaning and Preprocessing (Example: fill missing values)
video_data_combined.fillna('Unknown', inplace=True)

# Exploratory Data Analysis
# Analyzing the most popular videos
top_videos = video_data_combined.sort_values(by='views', ascending=False).head(20)

# Analyzing the most common words in video titles
title_words = pd.Series(' '.join(video_data_combined['title']).lower().split()).value_counts().head(10)

# Displaying results
print("Top 20 Most Popular Videos:\n", top_videos[['title', 'views']])
print("\nMost Common Words in Video Titles:\n", title_words)

category_analysis = video_data_combined.groupby('category_name').agg({'views': 'sum', 'likes': 'sum', 'dislikes': 'sum', 'comment_count': 'sum'}).sort_values(by='views', ascending=False)
print("\nCategory-wise Analysis:\n", category_analysis)

# Convert 'trending_date' to datetime
video_data_combined['trending_date'] = pd.to_datetime(video_data_combined['trending_date'], format='%y.%d.%m')

# Group by month and calculate the average views
date_analysis = video_data_combined.groupby(video_data_combined['trending_date'].dt.month).agg({'views': 'mean'})

# Improve the display by adding month names and formatting the views
date_analysis.index = pd.to_datetime(date_analysis.index, format='%m').month_name()
date_analysis['views'] = date_analysis['views'].apply(lambda x: "{:.2f}".format(x))

print("\naverage number of views that videos received in each month: \n", date_analysis)

print("\n-----------------------------------------------------")

# Combine all tags into a single list
all_tags = video_data_combined['tags'].str.cat(sep='|').split('|')

# Clean and count tags
tag_counts = Counter([re.sub(r'\s+', ' ', tag.strip().lower()) for tag in all_tags])

# Display most common tags
most_common_tags = tag_counts.most_common(20)
print("\nMost Common Tags:\n", most_common_tags)


print("\n----------------------------------------------------")

# Sort by views or likes
top_titles_by_views = video_data_combined.sort_values(by='views', ascending=False)[['title', 'views']].head(20)
top_titles_by_likes = video_data_combined.sort_values(by='likes', ascending=False)[['title', 'likes']].head(20)

# Display top titles
print("\nTop 20 Video Titles by Views:\n", top_titles_by_views)
print("\nTop 20 Video Titles by Likes:\n", top_titles_by_likes)
