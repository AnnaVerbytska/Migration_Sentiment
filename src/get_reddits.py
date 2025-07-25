# Install libraries
import pandas as pd  # Data manipulation and saving results to CSV
import praw  # Python Reddit API Wrapper for interacting with Reddit
import datetime  # To handle date and time objects for filtering posts by date
import time  # To manage pacing and pauses between API requests

import sys
import os
# Add the parent directory to the Python path to allow imports from parent folders if needed
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

# Load environment variables from a .env file to keep credentials securefrom dotenv 
import load_dotenv  # pip install python-dotenv
load_dotenv()  # Load environment variables from .env file


# Create Reddit client using environment variables
# These credentials are obtained by creating a Reddit app at https://www.reddit.com/prefs/apps
reddit = praw.Reddit(
    client_id=os.getenv("CLIENT_ID"),
    client_secret=os.getenv("CLIENT_SECRET"),
    user_agent=os.getenv("USER_AGENT")
)
# List of keywords to search for posts related to migration topics, including Ukraine and Gaza/Israel contexts
keywords = [
    # Migration-related
    "migrant", "refugee", "asylum seeker", "displaced", "migration",
    "border crossing", "immigrant"

    # Ukraine-focused additions
    "Ukrainian refugee", "refugee in Europe", "eastern Ukraine",
    "Donbas displaced", "refugee", "internal displacement Ukraine",

    # Gaza/Israel-focused additions
    "Gaza refugees", "Palestinian refugee", "Israel war migrant",
]

# Subreddits to search within for relevant posts
subreddits = [
    "worldnews", "news", "europe", "ukraine",
    "middleeast", "geopolitics", "politics",
    "IsraelPalestine", "russia"
]

# Date range: May 1 – July 31, 2025
start_date = datetime.datetime(2025, 5, 1)
end_date = datetime.datetime(2025, 7, 31, 23, 59, 59)

# Configurable per-subreddit cap to balance the number of posts
MAX_POSTS_PER_SUBREDDIT = 50

results = []

# Loop through subreddits
for subreddit_name in subreddits:
    subreddit = reddit.subreddit(subreddit_name)
    print(f"\n🔍 Searching in r/{subreddit_name}")

    post_count = 0

    for keyword in keywords:
        if post_count >= MAX_POSTS_PER_SUBREDDIT:
            print(f"   ✅ Reached limit of {MAX_POSTS_PER_SUBREDDIT} posts for r/{subreddit_name}")
            break

        try:
            print(f"   ➤ Keyword: {keyword}")
            # Search subreddit for posts containing the keyword, sorted by newest first, max 100 results per query
            for submission in subreddit.search(query=keyword, sort='new', limit=100):
                # Convert post creation time from Unix timestamp to datetime object
                post_date = datetime.datetime.fromtimestamp(submission.created_utc)


                # Filter posts based on these conditions:
                # - Post is a text/self post (not a link or image)
                # - Post text exists and is longer than 200 characters for substantial content
                # - Post has more than 5 upvotes to ensure some engagement
                # - Post date falls within the specified date range

                if (
                    submission.is_self and
                    submission.selftext.strip() and
                    len(submission.selftext.strip()) > 200 and
                    submission.score > 5 and
                    start_date <= post_date <= end_date
                ):
                     # Append relevant post information to results list as a dictionary
                    results.append({
                        "subreddit": subreddit_name,
                        "title": submission.title,
                        "text": submission.selftext,
                        "url": submission.url,
                        "created": post_date.strftime("%Y-%m-%d %H:%M:%S"),
                        "keyword": keyword,
                        "score": submission.score
                    })
                    post_count += 1
                    
                    # Stop collecting posts for this subreddit if the max limit is reached
                    if post_count >= MAX_POSTS_PER_SUBREDDIT:
                        break

            time.sleep(2)  # API pacing
        except Exception as e:
            # Print any errors encountered during API requests but continue processing other keywords/subreddits
            print(f"     [Error] {e}")
            continue

# Print the results
df = pd.DataFrame(results)

# Save the DataFrame to a CSV file in the data directory
df.to_csv("../data/reddit_raw.csv", index=False)
print(f"\n✅ Saved {len(df)} posts to '../data/reddit_raw.csv'")
