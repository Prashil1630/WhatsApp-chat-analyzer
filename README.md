# Chat-Analyzer
# WhatsApp Chat Analyzer

## Introduction
The **WhatsApp Chat Analyzer** is a data analysis tool designed to process and visualize WhatsApp chat data. It allows users to extract meaningful insights, such as the most active participants, sentiment analysis, daily message trends, and more. The application is built using **Streamlit** for an interactive user interface, along with various Python libraries for data processing and visualization.

## Features
- **Chat Data Parsing**: Extracts messages, timestamps, and sender details from WhatsApp chat exports.
- **Sentiment Analysis**: Analyzes the tone of messages using Natural Language Processing (NLP).
- **Message Statistics**: Displays total messages, word count, and active users.
- **Daily & Monthly Trends**: Visualizes message trends over time.
- **Most Active Users**: Identifies top contributors in the conversation.
- **Word Cloud**: Generates a word cloud of frequently used words.
- **Emoji Analysis**: Displays the most used emojis in the chat.
- **Media & Link Tracking**: Counts shared images, videos, and links.
- **Activity Heatmap**: Shows user activity distribution across different times of the day.
- **Topic Modeling**: Uses LDA to analyze chat topics and visualize them interactively.

## Technologies Used
- **Python**: For data processing and analysis.
- **Streamlit**: For the interactive web application.
- **Pandas**: For data manipulation.
- **Matplotlib & Seaborn**: For data visualization.
- **Plotly**: For interactive graphs.
- **NLTK & TextBlob**: For sentiment analysis.
- **WordCloud**: For generating word clouds.
- **Latent Dirichlet Allocation (LDA)**: For topic modeling.
- **PyLDAvis**: For interactive topic visualization.

## How It Works
1. **Upload Chat File**: Export and upload a WhatsApp chat file (`.txt` format).
2. **Data Preprocessing**: The `preprocessor.py` module extracts messages, timestamps, and user details.
3. **Visualization & Insights**: Various statistical and visual analyses are performed using `helper.py`.
4. **Sentiment Analysis**: Determines message polarity (positive, negative, neutral).
5. **Topic Modeling**: Identifies key conversation topics using LDA.

## Installation & Usage
### Prerequisites
Ensure you have Python installed along with the required dependencies:
```bash
pip install -r libraries.txt.txt
```
### Running the Application
```bash
streamlit run app.py
```

## File Structure
- **`app.py`**: Main Streamlit application that integrates all features.
- **`helper.py`**: Contains utility functions for data analysis, sentiment analysis, and visualization.
- **`preprocessor.py`**: Handles chat data parsing and preprocessing.
- **`stop_hinglish.txt`**: List of common stop words used for filtering text data.
- **`libraries.txt.txt`**: List of required Python packages.

## Future Enhancements
- Support for multi-language sentiment analysis.
- Advanced NLP techniques for better context understanding.
- Integration with a database for long-term chat storage.
- User authentication for personalized analysis.

## Conclusion
The **WhatsApp Chat Analyzer** provides an intuitive way to explore and gain insights from WhatsApp conversations. It simplifies chat data analysis and helps users understand messaging patterns, trends, and emotions effectively.

