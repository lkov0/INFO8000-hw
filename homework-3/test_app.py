#test_app.py

from flask import Flask, escape, request, jsonify
from flask_api import status
import sqlite3

app = Flask(__name__)
key = '2f279b10631d72ec3c9e82f3e59c6d83'

@app.route("/", methods = ['GET'])
def fun():
    conn = sqlite3.connect("sampleDB.db")
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    name = request.args.get("sampleid")
    if name is not None:
        c.execute('SELECT * FROM sample WHERE sample.id = ?', [name])
        rows = c.fetchall()
        results = [dict(row) for row in rows]
        return jsonify(results)
    else:
        c.execute('SELECT * FROM sample')
        rows = c.fetchall()
        results = [dict(row) for row in rows]
        return jsonify(results)

@app.route("/", methods = ['POST'])
def receivePost():
    
    clientkey = request.args.get("clientkey")
    
    if clientkey != key:
        return "uh", status.HTTP_401_UNAUTHORIZED
    else:
        conn = sqlite3.connect("sampleDB.db")
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        content = request.json
    
        c.execute('INSERT INTO sample VALUES (?,?,?,?,?,?,?,?,?)', (content['id'], content['date'], content['sequencer_id'], content['tissue'], content['species'], content['seed_id'], content['collection_location'], content['experiment'], content['data_size'])) 
        conn.commit()
    
        c.execute('SELECT * FROM sample')
        rows = c.fetchall()
        results = [dict(row) for row in rows]
        return jsonify(results)

    
