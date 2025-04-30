# FILE: app.py

from flask import Flask, render_template, jsonify
import json

app = Flask(__name__)

# === Route: Initial Analysis ===
@app.route('/initial-analysis/')
def initial_analysis_view():
    return render_template('initial-analysis.html')

@app.route('/data-initial-analysis')
def initial_analysis_data():
    with open('initial_analysis.json', 'r') as f:
        graph_data = json.load(f)
    return jsonify(graph_data)

if __name__ == '__main__':
    app.run(debug=True)