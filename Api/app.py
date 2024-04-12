from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
from flask_cors import CORS, cross_origin  # Thêm dòng này
import torch
import numpy as np
import re
from transformers import AutoModel, AutoTokenizer # Thư viện BERT
import underthesea # Thư viện tách từ
import os
from sklearn.model_selection import train_test_split # Thư viện chia tách dữ liệu
import unicodedata
import json
import warnings
import joblib

# Tắt tất cả các cảnh báo
warnings.filterwarnings("ignore")
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Load mô hình từ tệp PKL
with open('save_model.pickle', 'rb') as file:
    model = joblib.load(file)

@app.route('/')
def home():
    return jsonify({})

def remove_text_after_at_symbol(text):
    # Sử dụng biểu thức chính quy để tìm và thay thế phần cần xóa
    cleaned_text = re.sub(r'@[^ ]+', '@', text)
    return cleaned_text
def remove_period_comma(text):
    # Sử dụng phương thức replace để thay thế dấu chấm và dấu phẩy bằng chuỗi rỗng
    cleaned_text = text.replace('.', '').replace(',', '')
    return cleaned_text    
def encode_to_utf8(text):
    return text.encode('utf-8').decode('utf-8')

def lowercase_sentences(text):
    return text.lower()

def delete_redundant_space(text):
    return re.sub(' +', ' ', text)

def delete_links(text):
    return re.sub(r'http\S+', '', text)

def normalize_unicode(text):
    return unicodedata.normalize('NFKC', text)

def replace_k_with_khong(input_text):
    # Thay thế "k" thành "không" khi nó đứng một mình
    result = re.sub(r'\bk\b', 'không', input_text, flags=re.IGNORECASE)
    return result

def replace_h_with_gio(input_text):
    # Thay thế "h" thành "giờ" khi nó đứng một mình
    result = re.sub(r'\bh\b', 'giờ', input_text, flags=re.IGNORECASE)
    return result

def replace_cc_with_ccc(input_text):
    # Sử dụng phương thức replace để thay thế từ "cc" bằng cụm từ "con chim"
    result = input_text.replace('cc', 'con cặc')
    return result

# ở đây đặt hàm thay k => không để tránh trường hợp kk,kkk,... (Kaka) sẽ thành "khôngkhông" nên đặt hàm thay K trước hàm xóa các ký tự lặp
def delete_redundant_characters(text):
    return re.sub(r'(.)\1+', r'\1', text)

def remove_emojis(text):
    cleaned_text = re.sub(r'[^\w\s,.?!]', '', text)
    return cleaned_text

def standardize_data(row):
    # Xóa dấu chấm, phẩy, hỏi ở cuối câu
    row = re.sub(r"[\.,\?]+$-", "", row)
    # Xóa tất cả dấu chấm, phẩy, chấm phẩy, chấm thang, ... trong câu
    row = row.replace(",", " ").replace(".", " ") \
        .replace(";", " ").replace("“", " ") \
        .replace(":", " ").replace("”", " ") \
        .replace('"', " ").replace("'", " ") \
        .replace("!", " ").replace("?", " ") \
        .replace("-", " ").replace("?", " ")
    row = row.strip().lower()
    return row

@app.route('/predictsentiment', methods=['POST'])
@cross_origin()
def predictsentiment():
        data = request.json
        print(data)
        # Định nghĩa hàm để trích xuất nội dung từ chuỗi JSON
        def extract_content(json_data):
            # Truy cập giá trị của khóa "value"
            value = json_data['value']

            # Tách nội dung sau dấu ":"
            content = value.split(':', 1)[1].strip()

            return content
        
        content = extract_content(data)
        print("Nội dung bình luận: ",content)
        print("-----------------------------------")

        content = remove_text_after_at_symbol(content)
        content = remove_period_comma(content)
        content = encode_to_utf8(content)
        content = lowercase_sentences(content)
        content = delete_redundant_space(content)
        content = delete_links(content)
        content = normalize_unicode(content)
        content = replace_k_with_khong(content)
        content = replace_h_with_gio(content)
        content = replace_cc_with_ccc(content)
        content = delete_redundant_characters(content)
        content = remove_emojis(content)
        content = standardize_data(content)

        print("Xử lý sơ bộ: ",content)
 
        print("-----------------------------------")

        
        anhxa_df = pd.read_csv('anhxa_teencode.csv')
        conversion_dict = dict(zip(anhxa_df['teencode'], anhxa_df['fullword']))

        def map_teencode_to_full_word(text, conversion_dict):
                words = text.split()
                mapped_words = [conversion_dict.get(word, word) for word in words]
                mapped_text = ' '.join(mapped_words)
                return mapped_text
        
        text_processed = map_teencode_to_full_word(content, conversion_dict)
        print("Nghĩa đầy đủ: ", text_processed)

        print("-----------------------------------")

        def load_bert():
                v_phobert = AutoModel.from_pretrained("vinai/phobert-base")
                v_tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", use_fast=False)
                return v_phobert, v_tokenizer
        
        def load_stopwords():
                sw = []
                with open("vietnamese-stopwords.txt", encoding='utf-8') as f:
                        lines = f.readlines()
                for line in lines:
                        sw.append(line.replace("\n",""))
                return sw



        def make_bert_features(v_text):
                global phobert, tokenizer
                phobert, tokenizer = load_bert()
                v_tokenized = []
                sw = load_stopwords()
                max_len = 100 # Mỗi câu dài tối đa 100 từ

                if isinstance(v_text, str):
                    v_text = [v_text]


                for i_text in v_text:
                        print("Đang xử lý line = ", i_text)
                        # Phân thành từng từ
                        line = underthesea.word_tokenize(i_text)
                        # Lọc các từ vô nghĩa
                        filtered_sentence = [w for w in line if not w in sw]
                        # Ghép lại thành câu như cũ sau khi lọc
                        line = " ".join(filtered_sentence)
                        # line = underthesea.word_tokenize(line, format="text")
                        # Tokenize bởi BERT
                        line = tokenizer.encode(line, max_length=max_len, pad_to_max_length=True, truncation=True)
                        v_tokenized.append(line)

                # Chuyển đổi danh sách v_tokenized thành mảng NumPy
                padded_array = np.array(v_tokenized)
                print('padded:', padded_array[0])
                print('len padded:', padded_array.shape)

                # Chuyển mảng NumPy thành tensor
                padded_tensor = torch.tensor(padded_array).to(torch.long)
                print("Padded tensor shape:", padded_tensor.size())

                # Lấy features đầu ra từ BERT
                with torch.no_grad():
                        last_hidden_states = phobert(input_ids=padded_tensor)

                # Trích xuất features từ output của BERT
                v_features = last_hidden_states[0][:, 0, :].    numpy()
                # print("Features shape:", v_features.shape)

                return v_features

        new_features = make_bert_features(text_processed)
        # Dự đoán cảm xúc của bình luận bằng mô hình Machine Learning

        prediction = model.predict(new_features)

        print('Nhãn mà mô hình phát hiện:', str(prediction[0]))

        # Trả về dự đoán (0 cho tích cực, 1 cho tiêu cực)
        return str(prediction[0])

    
if __name__ == '__main__':
    app.run(debug=True)
