from flask import Flask, render_template, request, jsonify
from data_fetcher import fetch_reddit_post_data
from model import load_model, generate_summary

app = Flask(__name__)

# Load model and tokenizer once at the beginning
model, tokenizer = load_model()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize():
    url = request.form.get('url', '').strip()

    if url:
        data = fetch_reddit_post_data(url)
    else:
        return jsonify({"error": "No URL provided"}), 400

    summaries = []
    if isinstance(data, list):  # Multiple posts
        for post in data:
            text = " ".join([post["post_title"], post["post_body"]] + post["comments"])
            summary = generate_summary(model, tokenizer, text)
            summaries.append({"title": post["post_title"], "summary": summary})
    else:  # Single post
        text = " ".join([data["title"]] + data["comments"])
        summary = generate_summary(model, tokenizer, text)
        summaries.append({"title": data["title"], "summary": summary})

    return jsonify({"summaries": summaries})

if __name__ == "__main__":
    app.run(debug=True)
