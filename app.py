from flask import Flask, request, redirect, render_template
import os
from dotenv import load_dotenv
from process_pdf import process_pdf
from rag_pipeline import search_and_ask

# 載入 .env 檔案
load_dotenv()

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def upload_pdf():
    if request.method == 'POST':
        if 'pdf' not in request.files:
            return redirect(request.url)

        file = request.files['pdf']
        if file.filename == '':
            return redirect(request.url)

        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            process_pdf(filepath)
            return redirect('/ask')

    return render_template('upload.html')

@app.route('/ask', methods=['GET', 'POST'])
def ask():
    if request.method == 'POST':
        question = request.form['question']
        try:
            answer = search_and_ask(question)

            # 寫入查詢紀錄
            with open("query_log.txt", "a", encoding="utf-8") as log:
                log.write(f"Q: {question}\nA: {answer}\n\n")

        except Exception as e:
            answer = f"⚠️ 發生錯誤：{str(e)}"

        return render_template('result.html', question=question, answer=answer)

    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
