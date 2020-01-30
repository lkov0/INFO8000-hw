#test_app.py

from flask import Flask, escape, request, jsonify

app = Flask(__name__)

@app.route("/", methods = ['GET','POST'])
def fun():
    name = request.args.get("name", "World")
    response = {"name":name}
    return jsonify(response)
