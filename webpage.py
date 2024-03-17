from flask import Flask

app = Flase(__name__)

@app.route('/')
def index():
    return None

if __name__ == "__main__":
    app.run(debug=True)