from keras.preprocessing.sequence import pad_sequences
from seqeval.metrics import classification_report,accuracy_score,f1_score
import math
import torch
import os
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

from pytorch_pretrained_bert import BertTokenizer, BertConfig
from pytorch_pretrained_bert import BertForTokenClassification, BertAdam
import pandas as pd
import numpy as np
import re
from tqdm import tqdm_notebook as tqdm
from tqdm import tqdm,trange
from keras.preprocessing.text import Tokenizer 
import docx2txt
from pdfminer.converter import TextConverter
from pdfminer.pdfinterp import PDFPageInterpreter
from pdfminer.pdfinterp import PDFResourceManager
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
import io
import json
# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
import pickle
# Define a flask app
app = Flask(__name__)


PATH = (r'models\saved_model.pb')
# Load your trained model
model = BertForTokenClassification.from_pretrained("bert-base-uncased",num_labels=12)

model.load_state_dict(torch.load(PATH, map_location='cpu'))
model.eval()

print('Model loaded. Check http://127.0.0.1:5000/')

def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as fh:
        # iterate over all pages of PDF document
        for page in PDFPage.get_pages(fh, caching=True, check_extractable=True):
            # creating a resoure manager
            resource_manager = PDFResourceManager()
            
            # create a file handle
            fake_file_handle = io.StringIO()
            
            # creating a text converter object
            converter = TextConverter(
                                resource_manager, 
                                fake_file_handle, 
                                codec='utf-8', 
                                laparams=LAParams()
                        )

            # creating a page interpreter
            page_interpreter = PDFPageInterpreter(
                                resource_manager, 
                                converter
                            )

            # process current page
            page_interpreter.process_page(page)
            
            # extract text
            text = fake_file_handle.getvalue()
            yield text

            # close open handles
            converter.close()
            fake_file_handle.close()


def extract_text_from_doc(doc_path):
    text = docx2txt.process(doc_path)
    return text

def predict(text):
    
    text=text.replace("\n"," ")
    tokenizer = Tokenizer(num_words=20000)
    tokenizer.fit_on_texts(text)
    tokenized_texts = tokenizer.texts_to_sequences(text)
    input_ids = pad_sequences(tokenized_texts,maxlen=10000, dtype="long", truncating="post", padding="post")
    text1= torch.tensor(input_ids)
    device = torch.device("cuda")
    logits = model(text1,token_type_ids=None)
    logits = logits.detach().cpu().numpy()
    predictions=[]
    tags_vals = ["UNKNOWN", "Name", "Degree","Skills","College Name","Email Address","Designation","Companies worked at","Empty","Graduation Year","Years of Experience","Location"]
    predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
    pred_tags = [tags_vals[p_i] for p in predictions for p_i in p]
    result=""
    for i in range(len(pred_tags)):
        result+=str(pred_tags[i])
    return str(result),text


def prepare_data(path):
    name, extension = os.path.splitext(path)
    text=""
    if extension==".pdf":
        for page in extract_text_from_pdf(path):
            text += ' ' + page
            return text
    elif extension==".docx":
        extract_text_from_doc(path)
        return text
    elif extension==".json":
        text=json.load(path)
        return text
    else:
        return "File not supported"

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        text = prepare_data(file_path)
        result=predict(text)
        
        return result
    return None


if __name__ == '__main__':
    app.run(debug=True)

