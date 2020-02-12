#test_app.py

from flask import Flask, escape, request, jsonify
import sqlite3
import pandas as pd

conn = sqlite3.connect("sampleDB.db")

app = Flask(__name__)

@app.route("/", methods = ['GET','POST'])
def fun():
    name = request.args.get("sampleid", 1)
    cursor = conn.execute("SELECT * FROM sample WHERE sample.id = 1")
    rows = cursor.fetchall()
    alldata = []
    content = {}
    for row in rows:
        content = {'id' = row[0], 'date' = row[1], 'sequencer_id' = row[2], 'tissue' = row[3], 'species' = row[4], 'seed_id' = row[5], 'collection_location' = row[6], 'experiment' = row[7], 'data_size' = row[8]}
        alldata.append(content)
        content = {}
    return jsonify(content)
