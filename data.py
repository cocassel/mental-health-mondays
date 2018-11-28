import praw
import csv

reddit = praw.Reddit(client_id='nCs3G9SYaRBn4Q',
                     client_secret='LBILMTx4fvoJlS9LfSNOFMlVqjw',
                     user_agent='my user agent')

reddit_comments = []

for submission in reddit.subreddit('uwaterloo').new(limit=1000):
    if submission.title == "Mental Health Monday":
        submission.comment_sort = 'new'
        post_comments = submission.comments.list()
        print post_comments
        for comment in post_comments:
            if comment.body:
                reddit_comments.append(comment.body.encode('utf-8').strip())

print(reddit_comments)
with open('comments.csv', 'wb') as csv_file:
    writer = csv.writer(csv_file)
    for comment in reddit_comments:
        writer.writerow([comment])
