from flask import Flask, render_template, request

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html",
                         title="MedMonitor",
                         header="Система мониторинга медицинских сигналов",
                         description="Выберите тип анализа")

@app.route("/ecg")
def ecg():
    return render_template("signal.html",
                         signal="ecg",
                         header="ЭКГ",
                         description="Электрокардиограмма - мониторинг сердечной активности",
                         color="rgba(30,58,138,0.8)")

@app.route("/emg")
def emg():
    return render_template("signal.html",
                         signal="emg",
                         header="ЭМГ",
                         description="Электромиография - мониторинг мышечной активности",
                         color="rgba(220,38,38,0.8)")

@app.route("/ppg")
def ppg():
    return render_template("signal.html",
                         signal="ppg",
                         header="ФПГ",
                         description="Фотоплетизмография - оптический метод измерения",
                         color="rgba(16,185,129,0.8)")

@app.route("/resp")
def resp():
    return render_template("signal.html",
                         signal="resp",
                         header="Дыхание",
                         description="Респираторный сигнал - мониторинг дыхания",
                         color="rgba(59,130,246,0.8)")

@app.context_processor
def inject_request():
    return dict(request=request)

if __name__ == "__main__":
    app.run(debug=True, port=5000)