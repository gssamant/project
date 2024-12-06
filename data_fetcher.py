import praw
import json
from bs4 import BeautifulSoup
import requests
from config import (
    REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT,
    TOP_POST_LIMIT, COMMENT_UPVOTE_THRESHOLD
)

# Initialize Reddit API
reddit = praw.Reddit(
    client_id=REDDIT_CLIENT_ID,
    client_secret=REDDIT_CLIENT_SECRET,
    user_agent=REDDIT_USER_AGENT
)

def save_data_to_file(data, filename="reddit_data.json"):
    """
    Save fetched data to a local JSON file.
    """
    with open(filename, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

def load_data_from_file(filename="reddit_data.json"):
    """
    Load data from a local JSON file.
    """
    with open(filename, "r", encoding="utf-8") as file:
        return json.load(file)

def fetch_reddit_post_data(url=None):
    """
    Fetch Reddit data from a URL or the top posts in the technology subreddit.
    """
    return fetch_post_from_url(url) if url else fetch_top_posts()

def fetch_post_from_url(url):
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')

    title = soup.find('h1').text if soup.find('h1') else 'No title found'
    comments = [
        comment.text for comment in soup.find_all('div', {'data-testid': 'comment-body'})
    ]

    return {"title": title, "comments": comments}

def fetch_top_posts():
    subreddit = reddit.subreddit("technology")
    top_posts = subreddit.top(limit=TOP_POST_LIMIT)
    data = []

    for post in top_posts:
        post_data = {
            "post_title": post.title,
            "post_body": post.selftext,
            "comments": []
        }
        post.comments.replace_more(limit=0)
        for comment in post.comments:
            if comment.score >= COMMENT_UPVOTE_THRESHOLD:
                post_data["comments"].append(comment.body)
        data.append(post_data)

    save_data_to_file(data)  # Save data locally
    return data
