from flask import Flask, request, jsonify, render_template
from sentiment import analyze_sentiment  # This uses your full pipeline
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static'

@app.route('/')
def home():
    return render_template('index.html')  # Make sure this exists in templates/

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    video_id = data.get('videoId')

    if not video_id:
        return jsonify({'error': 'No videoId provided'}), 400

    try:
        result = analyze_sentiment(video_id)
        return jsonify({
            'thumbnail_url': f"https://img.youtube.com/vi/{video_id}/0.jpg",
            #'pie_chart_url': '/static/pie_chart.png',
            'positive': result['positive'],
            'neutral':  result['neutral'],
            'negative': result['negative']
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500 

if __name__ == '__main__':
    app.run(debug=True)

