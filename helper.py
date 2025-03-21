#Helper
from urlextract import URLExtract
from wordcloud import WordCloud
import pandas as pd
from collections import Counter
import emoji
from textblob import TextBlob  # For sentiment analysis
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer#machine earning algo
import pyLDAvis
import matplotlib.pyplot as plt
from wordcloud import WordCloud

extractor = URLExtract()

def fetch_stat(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    num_messages = df.shape[0]
    words = []
    for message in df['message']:
        words.extend(message.split())
    num_media_messages = df[df['message'] == '<Media omitted>\n'].shape[0]
    links = []
    for message in df['message']:
        links.extend(extractor.find_urls(message))
    return num_messages, len(words), num_media_messages, len(links)

def most_busy_user(df):
    x = df['user'].value_counts().head()
    df = round((df['user'].value_counts()/df.shape[0])*100, 2).reset_index().rename(columns={'user': 'name', 'count': 'percent'})
    return x, df

def create_wordcloud(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    with open('stop_hinglish.txt', 'r') as f:
        stop_words = f.read().splitlines()
    temp = df[df['user'] != 'group_notification']
    temp = temp[temp['message'] != '<Media omitted>\n']
    def remove_stop_words(message):
        return " ".join(word for word in message.lower().split() if word not in stop_words)
    wc = WordCloud(width=500, height=500, min_font_size=10, background_color='white')
    temp['message'] = temp['message'].apply(remove_stop_words)
    df_wc = wc.generate(temp['message'].str.cat(sep=" "))
    return df_wc

def most_common_words(selected_user, df):
    with open('stop_hinglish.txt', 'r') as f:
        stop_words = f.read().splitlines()
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    temp = df[df['user'] != 'group_notification']
    temp = temp[temp['message'] != '<Media omitted>\n']
    words = []
    for message in temp['message']:
        for word in message.lower().split():
            if word not in stop_words:
                words.append(word)
    return_df = pd.DataFrame(Counter(words).most_common(20))
    return return_df

def emoji_helper(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    emojis = [c for message in df['message'] for c in message if emoji.is_emoji(c)]
    emoji_df = pd.DataFrame(Counter(emojis).most_common(len(Counter(emojis))))
    return emoji_df

def monthly_timeline(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    timeline = df.groupby(['year', 'month_num', 'month']).count()['message'].reset_index()
    timeline['time'] = timeline['month'].astype(str) + "-" + timeline['year'].astype(str)
    return timeline

def daily_timeline(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    daily_timeline = df.groupby('only_date').count()['message'].reset_index()
    return daily_timeline

def week_activity_map(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    return df['day_name'].value_counts()

def month_activity_map(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    return df['month'].value_counts()

def user_heatmap(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    activity_heatmap = df.pivot_table(index='day_name', columns='period', values='message', aggfunc='count').fillna(0)
    return activity_heatmap

def sentiment_analysis(daily_timeline, df):
    sentiment_data = daily_timeline.copy()
    sentiment_data['positive'] = 0
    sentiment_data['negative'] = 0
    sentiment_data['neutral'] = 0  # Add neutral column

    if 'only_date' not in sentiment_data.columns:
        raise KeyError("Column 'only_date' not found in daily_timeline DataFrame")
    if 'message' not in df.columns:
        raise KeyError("Column 'message' not found in df DataFrame")

    for i, row in sentiment_data.iterrows():
        date = row['only_date']
        messages = df[df['only_date'] == date]['message']
        sentiment_scores = [TextBlob(message).sentiment.polarity for message in messages]
        sentiment_data.at[i, 'positive'] = sum(score > 0 for score in sentiment_scores)
        sentiment_data.at[i, 'negative'] = sum(score < 0 for score in sentiment_scores)
        sentiment_data.at[i, 'neutral'] = sum(score == 0 for score in sentiment_scores)

    return sentiment_data

# Function to perform topic modeling using LDA with visualization
def perform_topic_modeling_visualization(chat_data, n_topics=5, method="lda"):
    # Vectorize the chat data
    vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
    chat_data_vectorized = vectorizer.fit_transform(chat_data)

    # Apply LDA
    if method == "lda":
        model = LatentDirichletAllocation(n_components=n_topics, random_state=42)
        model.fit(chat_data_vectorized)
    else:
        raise ValueError("Invalid method: Choose 'lda'")

    # Visualize with pyLDAvis
    lda_vis_data = pyLDAvis.sklearn.prepare(model, chat_data_vectorized, vectorizer)  # Updated this line
    pyLDAvis.save_html(lda_vis_data, 'lda_topics.html')  # Save to HTML

    # Visualize with word clouds for each topic
    visualize_wordclouds(model, vectorizer, n_topics)

    # Return the model and topics
    return model, lda_vis_data


# Word cloud visualization function
def visualize_wordclouds(model, vectorizer, n_topics=5):
    terms = vectorizer.get_feature_names_out()

    for topic_idx, topic in enumerate(model.components_):
        # Generate a word cloud for each topic
        word_freq = {terms[i]: topic[i] for i in topic.argsort()[:-21:-1]}  # Top 20 words
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq)

        # Plot the word cloud
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.title(f'Topic {topic_idx + 1}', fontsize=16)
        plt.axis("off")
        plt.show()
