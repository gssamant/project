from data_fetcher import fetch_reddit_post_data
from model import load_model, generate_summary

def main():
    print("Welcome to the Reddit Summarizer!")
    url = input("Enter a Reddit post URL (or press Enter to fetch top posts): ").strip()

    data = fetch_reddit_post_data(url)
    model, tokenizer = load_model()

    if isinstance(data, list):  # Top posts
        for idx, post in enumerate(data, start=1):
            print(f"\nSummarizing Post {idx}...")
            text = " ".join([post["post_title"], post["post_body"]] + post["comments"])
            summary = generate_summary(model, tokenizer, text)
            print(f"Post Title: {post['post_title']}")
            print(f"Summary: {summary}")
    else:
        print("\nSummarizing the provided post...")
        text = " ".join([data["title"]] + data["comments"])
        summary = generate_summary(model, tokenizer, text)
        print(f"Post Title: {data['title']}")
        print(f"Summary: {summary}")

if __name__ == "__main__":
    main()
