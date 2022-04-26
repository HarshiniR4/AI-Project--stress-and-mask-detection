from flask import Flask, Response, render_template
from stressmasktest import VideoCamera

app = Flask(__name__, template_folder='D:/AI-Project--stress-and-mask-detection\code/templates')


@app.route('/')
def home():
    return render_template('index.html')

def gen(stressmask):
    """Video streaming generator function."""
    while True:
        frame= stressmask.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        print(frame)
       
@app.route("/predict", methods=['POST', 'GET'])
def predict():  
    return Response(gen(VideoCamera()), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=False,threaded=False)