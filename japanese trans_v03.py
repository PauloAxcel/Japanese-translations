# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 09:03:47 2023

@author: Paulo Gomes
"""
import cv2
import pytesseract
from PIL import ImageGrab
from googletrans import Translator
import numpy as np
import mss
import pykakasi
import matplotlib.pyplot as plt

def acquire_screenshot(monitor_number):
    with mss.mss() as sct:
        mon = sct.monitors[monitor_number]
        screenshot = sct.grab(mon)
        screenshot = np.array(screenshot)
        # screenshot2 = screenshot[screen_region[0]:screen_region[1], screen_region[2]:screen_region[3]]
    
    return screenshot

def preprocess_image(image):
    # Apply image preprocessing techniques like resizing, noise reduction, and thresholding
    # Use OpenCV for these tasks
    # For example, convert the image to grayscale and apply thresholding
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, threshold_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return threshold_image

def perform_ocr(image):
    # Perform OCR using pytesseract on the preprocessed image
    # Set the correct language for Japanese characters
    # extracted_text = pytesseract.image_to_string(image, lang='jpn')
    extracted_text_data = pytesseract.image_to_data(image, lang='jpn', output_type=pytesseract.Output.DICT)
    return extracted_text_data

def translate_text(cleaned_extracted_text_data):
    # Use a translation API to translate the Japanese text
    # For example, using the googletrans library:
    translator = Translator()
    cleaned_extracted_text_data_trans = cleaned_extracted_text_data
    for cnt, matrix in enumerate(cleaned_extracted_text_data):
        text = matrix[0]
        translated_text = translator.translate(text, dest='en').text
        cleaned_extracted_text_data_trans[cnt].append(translated_text)
        
    return cleaned_extracted_text_data_trans

import re

def clean_japanese_text(extracted_text_data):
    
    text = ('\n').join(extracted_text_data['text'])
    japanese_pattern = re.compile(r'[^\u3000-\u303F\u3040-\u309F\u30A0-\u30FF\uFF66-\uFF9F\u4E00-\u9FFF\n]+')
    
    # Use the regex pattern to remove non-Japanese characters from the text
    cleaned_text = japanese_pattern.sub('', text)
    
    # Remove the single consecutive newline character
    cleaned_text = re.sub(r'(?<=\S)\n(?=\S)', '', cleaned_text)

    # Replace multiple consecutive newline characters with a single newline character
    cleaned_text = re.sub(r'\n{2,}', '\n', cleaned_text)

    return cleaned_text

def convert_space_inside(lst):
    updated_list = []
    for element in lst:
        if element == ' ' or element == '  ':
            updated_list.append('')
        else:
            updated_list.append(element)
    return updated_list

def separate_sentences(text1,left,top,width,height):
    sentences = []
    tops = []
    lefts = []
    widths = []
    heights = []
    
    current_sentence = ""
    current_top = []
    current_left = []
    current_width = []
    current_height = []
    
    sentence_count = 1
    for cnt, (te,l,t,w,h) in enumerate(zip(text1,left,top,width,height)):
        if te != "":
            # Belongs to a sentence
            current_sentence += te + " "
            current_top.append(t)
            current_left.append(l)
            current_width.append(w)
            current_height.append(h)
        else:
            # Empty string, sentence ended
            if current_sentence.strip() != "":
                sentences.append(current_sentence.strip())
                tops.append([int(np.min(current_top)),int(np.max(current_top))])
                lefts.append([int(np.min(current_left)),int(np.max(current_left))])
                widths.append([int(np.min(current_width)),int(np.max(current_width))])
                heights.append([int(np.min(current_height)),int(np.max(current_height))])
                                        

                
                sentence_count += 1
                
                current_sentence = ""
                current_top = []
                current_left = []
                current_width = []
                current_height = []
    # Add the last sentence if it exists
    if current_sentence.strip() != "":
        sentences.append(current_sentence.strip())
        tops.append([int(np.min(current_top)),int(np.max(current_top))])
        lefts.append([int(np.min(current_left)),int(np.max(current_left))])
        widths.append([int(np.min(current_width)),int(np.max(current_width))])
        heights.append([int(np.min(current_height)),int(np.max(current_height))])
        
    return sentences, tops, lefts, widths, heights

import japanize_matplotlib
def clean_extracted_data(extracted_text_data, cleaned_text, image_np):
    
    text0 = extracted_text_data['text']
    text1 = convert_space_inside(text0)
    
    
    left = extracted_text_data['left']
    top = extracted_text_data['top']
    width = extracted_text_data['width']
    height = extracted_text_data['height']
    
    sentences, tops, lefts, widths, heights = separate_sentences(text1,left,top,width,height)

    japanese_pattern = re.compile(r'[^\u3000-\u303F\u3040-\u309F\u30A0-\u30FF\uFF66-\uFF9F\u4E00-\u9FFF0-9\n]+')
    
    cleaned_text = [japanese_pattern.sub('', a) for a in sentences]
    
    spaces2 = [a != '' for a in cleaned_text]
    
    left2 = [a for s,a in zip(spaces2,lefts) if s == True]
    top2 = [a for s,a in zip(spaces2,tops) if s == True]
    width2 = [a for s,a in zip(spaces2,widths) if s == True]
    height2 = [a for s,a in zip(spaces2,heights) if s == True]
    text2 = [a for s,a in zip(spaces2,cleaned_text) if s == True]
    
    matrix = []
    for l,t,w,h,te in zip(left2,top2,width2,height2,text2):
        # Crop the corresponding region from the image
        # try:
        #     # Calculate the coordinates of the cropping region
        delta = 15
        crop_left = max(0, l[0] - int(np.min(w)) - delta)
        crop_top = max(0, t[0] + int(np.min(h)) - delta)
        crop_right = min(image_np.shape[1] - 1, l[1] + int(np.min(w)) + delta)
        crop_bottom = min(image_np.shape[0] - 1, t[1] + int(np.min(h)) + delta)
    
        #     # Crop the corresponding region from the image
        #     cropped_region = image_np[crop_top:crop_bottom, crop_left:crop_right]
        #     plt.figure()
        #     plt.imshow(cropped_region)
        #     plt.title(te)
        matrix.append([te,crop_top ,crop_bottom, crop_left ,crop_right ])
        # except:
        #     continue
        
        
        # plt.imshow(image_np[:,left2[9][0]-int(np.min(width2[9])):left2[9][1]+int(np.min(width2[9]))])
        
        # delta = 10
        # plt.imshow(image_np[top2[9][0]+int(np.min(height2[9]))-delta:top2[9][1]+int(np.min(height2[9]))+delta,
        #                     left2[9][0]-int(np.min(width2[9]))-delta:left2[9][1]+int(np.min(width2[9]))+delta])
        

    return matrix


def kanji_to_hiragana(text):
    # Use Google Translate for converting Kanji to Hiragana
    kks = pykakasi.kakasi()
    result = kks.convert(text)
    orig = []
    hira = []
    kata = []
    for item in result:
        orig.append(item['orig'])
        hira.append(item['hira'])
        kata.append(item['kana'])
        
    hiragana_text = ('').join(hira)

    return hiragana_text

import win32api
x, y = win32api.GetCursorPos()

import nltk
nltk.download('words')
from nltk.corpus import words

word_list = set(words.words())


def is_valid_word(word):
    return word.lower() in word_list or not word.isalpha()

def clean_sentence(sentence):
    words = nltk.wordpunct_tokenize(sentence)
    valid_words = [word for word in words if is_valid_word(word)]
    return ' '.join(valid_words)

def calculate_meaning_score(sentence):
    words = nltk.wordpunct_tokenize(sentence)
    valid_word_count = sum(1 for word in words if is_valid_word(word))
    total_word_count = len(words)
    return valid_word_count / total_word_count if total_word_count > 0 else 0.0




def main():
    # Define the region of the screen to capture (coordinates may vary based on your setup)
    # xtop = 240
    # ytop = 439
    # xbottom = 1450
    # ybottom = 635
    
    # screen_region = (ytop, ybottom, xtop, xbottom)  # Example region (top-left corner: 0,0, bottom-right corner: 800,600)
    monitor_number = 1
    pytesseract.pytesseract.tesseract_cmd = r'C:\Users\Paulo Gomes\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'

    # Capture the screen
    screen = acquire_screenshot(monitor_number)
    
    # Convert the captured image to a numpy array
    image_np = np.array(screen)
    
    # Preprocess the image
    processed_image = preprocess_image(image_np)
    
    # Perform OCR
    extracted_text_data = perform_ocr(processed_image)

    # Clean the text and keep only Japanese characters
    cleaned_text = clean_japanese_text(extracted_text_data)
    
    # [te,crop_top ,crop_bottom, crop_left ,crop_right ]
    cleaned_extracted_text_data = clean_extracted_data(extracted_text_data, cleaned_text, image_np)

    # Translate the cleaned text
    cleaned_extracted_text_data_trans = translate_text(cleaned_extracted_text_data)

    # Display the original Japanese text and the translated text
    for matrix in cleaned_extracted_text_data_trans:
        japanese_text = matrix[0]
        english_text = matrix[-1]
        # print("Japanese Text:")
        # print(japanese_text)
        # Convert Kanji to Hiragana and get the romanized reading
        hiragana_text = kanji_to_hiragana(japanese_text)
        # print(hiragana_text)
        # print("Translated Text:")
        print(english_text)
        meaning_scores = calculate_meaning_score(english_text)
        print(f'Score: {meaning_scores}')
        # print('---')
        # print()  # Add an empty line to separate each pair of Japanese and English translations
        # Get bounding box information for the current text region
        
        crop_top = matrix[1]
        crop_bottom = matrix[2]
        crop_left = matrix[3]
        crop_right = matrix[4]
        
        cropped_region = image_np[crop_top:crop_bottom, crop_left:crop_right]
        plt.figure()
        plt.imshow(cropped_region)
        plt.title(japanese_text + '\n' + hiragana_text + '\n' + english_text)

if __name__ == "__main__":
    main()





# import MeCab



# import subprocess

# def parse_japanese_sentence(sentence):
#     mecab_path = r'C:/Program Files/MeCab/bin'
#     unidic_lite_path = r'C:/Users/Paulo Gomes/anaconda3/Lib/site-packages/unidic_lite'
    
#     output = subprocess.check_output([mecab_path, '-d', unidic_lite_path, '-Owakati', sentence])
#     words = output.decode().strip().split()

#     meanings = []
#     for word in words:
#         output = subprocess.check_output([mecab_path, '-d', unidic_lite_path, word])
#         output_lines = output.decode().strip().split('\n')
#         for line in output_lines:
#             surface, feature = line.split('\t')
#             features = feature.split(',')
#             word_meaning = (surface, features[0], features[-3])
#             meanings.append(word_meaning)

#     return meanings



# input_sentence = "ここは我々ミツバチの王国"
# result = parse_japanese_sentence(input_sentence)

# print("Input Sentence:", input_sentence)
# print("Table of Meaning:")
# for word, part_of_speech, meaning in result:
#     print(f"{word} {part_of_speech} {meaning}")



