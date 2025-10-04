from flask import Flask, render_template

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/ecg")
def ecg():
    return render_template("ecg.html", signal="ecg", color="rgba(30,58,138,0.8)")

@app.route("/emg")
def emg():
    return render_template("emg.html", signal="emg", color="rgba(220,38,38,0.8)")

@app.route("/ppg")
def ppg():
    return render_template("ppg.html", signal="ppg", color="rgba(16,185,129,0.8)")

@app.route("/resp")
def resp():
    return render_template("resp.html", signal="resp", color="rgba(59,130,246,0.8)")

if __name__ == "__main__":
    app.run(debug=True)
