import re
import pandas as pd
import nltk
from nltk.corpus import stopwords


def preprocess(data):
    pattern = r'\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}\s(?:am|pm)\s-\s'
    messages = re.split(pattern, data)[1:]  # Split the chat data into messages
    dates = re.findall(pattern, data)  # Extract the date pattern

    # Create a DataFrame
    df = pd.DataFrame({'user_messages': messages, 'message_date': dates})

    # Convert message_date to datetime
    df['message_date'] = pd.to_datetime(df['message_date'], format='%d/%m/%y, %I:%M %p - ')
    df.rename(columns={'message_date': 'date'}, inplace=True) #clarity

    # Separate user and messages
    users = []
    messages = []
    for message in df['user_messages']:
        entry = re.split('([\w\W]+?):\s', message)
        if entry[1:]:  # Check if there's a username
            users.append(entry[1])
            messages.append(entry[2])
        else:
            users.append('group_notification')  # For messages without a user
            messages.append(entry[0])

    df['user'] = users
    df['message'] = messages
    df.drop(columns=['user_messages'], inplace=True)

    # Extract date components
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month_name()
    df['month_num'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['hour'] = df['date'].dt.hour
    df['minute'] = df['date'].dt.minute
    df['only_date'] = df['date'].dt.date
    df['day_name'] = df['date'].dt.day_name()

    # Create periods
    period = []
    for hour in df['hour']:
        if hour == 23:
            period.append(f"{hour}-{00}")
        else:
            period.append(f"{hour}-{hour + 1}")
    df['period'] = period

    return df


# Download NLTK stopwords
nltk.download('stopwords')


def preprocess_chat_data(chat_data):
    stop_words = set(stopwords.words('english'))

    # Load Hinglish stopwords
    with open('stop_hinglish.txt', 'r') as file:
        hinglish_stopwords = set(file.read().splitlines())

    stop_words.update(hinglish_stopwords)

    # Clean and tokenize chat data
    cleaned_chat = []
    for message in chat_data:
        # Check if the message is a string
        if isinstance(message, str):
            # Remove non-alphanumeric characters
            message = re.sub(r'\W+', ' ', message)
            tokens = message.lower().split()
            tokens = [word for word in tokens if word not in stop_words and len(word) > 1]
            cleaned_chat.append(' '.join(tokens))
        else:
            print(f"Warning: Skipped a non-string message: {message}")  # Log non-string messages

    return cleaned_chat
