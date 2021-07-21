from flask import Flask, render_template,request
import pickle
import numpy as np
from iris import accuracy, con_mat, set_matrix

model = pickle.load(open('model.pkl','rb'))
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def pred():
    data1 = request.form['SL']
    data2 = request.form['SW']
    data3 = request.form['PL']
    data4 = request.form['PW']
    arr = np.array([[data1,data2,data3,data4]])
    output = model.predict(arr)
    print(output[0])
    return render_template('output.html',data=output[0])

@app.route('/info')
def info():
    return render_template('info.html', acc= round(accuracy), confusion= con_mat, set_info = set_matrix)

if __name__ == '__main__':
    app.run(debug=True)
