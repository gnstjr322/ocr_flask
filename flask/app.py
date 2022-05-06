# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 15:23:20 2022

@author: ktnet
"""
import matplotlib.pyplot as plt
from werkzeug.utils import secure_filename
import cv2
from PIL import Image, ImageFont, ImageDraw
import ssl
import numpy as np
import imutils
import matplotlib.pyplot as plt
import argparse
import imutils
from wand.image import Image
from wand.display import display
from collections import namedtuple
import os
from flask import Flask, flash, request, redirect, url_for, render_template
import urllib.request
import pytesseract
import time

# =============================================================================
# https://tutorial101.blogspot.com/2021/04/python-flask-upload-and-display-image.html
# =============================================================================


app = Flask(__name__)
 
UPLOAD_FOLDER = 'static/uploads/' # not use this dir
TABLEZOOM_FOLDER = 'static/tablezoom/' # also
MATCHING_FOLDER = 'static/matching/'

NUMBER_FOLDER = 'static/category/number/'
BL_FOLDER = 'static/category/bl/'
CARGO_FOLDER = 'static/category/cargo/'
PAY_FOLDER = 'static/category/pay/'

app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER # not use this
app.config['TABLEZOOM_FOLDER'] = TABLEZOOM_FOLDER
app.config['MATCHING_FOLDER'] = MATCHING_FOLDER
app.config['NUMBER_FOLDER'] = NUMBER_FOLDER
app.config['BL_FOLDER'] = BL_FOLDER
app.config['CARGO_FOLDER'] = CARGO_FOLDER
app.config['PAY_FOLDER'] = PAY_FOLDER

app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
 
num_template = cv2.imread("reference/number.jpg",cv2.IMREAD_GRAYSCALE)
num_template = cv2.resize(num_template, (1000,40))

pay_template = cv2.imread("reference/payment2.jpg",cv2.IMREAD_GRAYSCALE)
pay_template = cv2.resize(pay_template, (1000,40))

real_num_template = cv2.imread("reference/real_number.jpg",cv2.IMREAD_GRAYSCALE)

bl_template = cv2.imread("reference/bl.jpg",cv2.IMREAD_GRAYSCALE)
bl_template = cv2.resize(bl_template, (1000,40))

alpha = 0.5

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def number(image, template):
    
    img2 = image.copy()
    # 입력 영상 밝기 50증가, 가우시안 잡음(sigma=10) 추가하여 약간의 변형을 줌
    noise = np.zeros(image.shape, np.int32)
    # cv2.randn은 가우시안 형태의 랜덤 넘버를 지정, 노이즈 영상에 평균이 50 시그마 10인 노이즈 추가
    cv2.randn(noise,50,10)
    
    # 노이즈를 입력 영상에 더함, 원래 영상보다 50정도 밝아지고 시그마 10정도 변형
    image = cv2.add(image, noise, dtype=cv2.CV_8UC3)
        
    res = cv2.matchTemplate(image, template,cv2.TM_CCOEFF_NORMED)
    res_norm = cv2.normalize(res, None, 0,255, cv2.NORM_MINMAX, cv2.CV_8U)
    
    # 최댓값을 찾아야하므로 minmaxloc 사용, min, max, min좌표, max좌표 반환
    _, maxv, _, maxloc = cv2.minMaxLoc(res)
    
    # 탬플릿에 해당하는 영상이 입력 영상에 없으면 고만고만한 값에서 가장 큰 값을 도출.
    # 그래서 maxv를 임계값 0.7 or 0.6을 설정하여 템플릿 영상이 입력 영상에 존재하는지 파악
    #print('maxv : ', maxv)
    #print('maxloc : ', maxloc) 
    
    # 매칭 결과를 빨간색 사각형으로 표시
    # maxv가 어느 값 이상이여야지 잘 찾았다고 간주할 수 있다.
    th, tw = template.shape[:2]
    dst = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    
    #cv2.rectangle(dst, maxloc, (maxloc[0] + tw, maxloc[1] + th), (0, 0, 255), 2)
    dst_number = img2[ :maxloc[1] + th + 3, 20: 203 ]
    
    
    # =============================================================================
    # 한글 신고번호 제외하기    
    # =============================================================================
      
    res = cv2.matchTemplate(dst_number, real_num_template, cv2.TM_CCOEFF_NORMED)
    res_norm = cv2.normalize(res, None, 0,255, cv2.NORM_MINMAX, cv2.CV_8U)
    
    _, maxv, _, maxloc = cv2.minMaxLoc(res)
    cv2.rectangle(dst, (0, 22), (180, th+10 ), (0, 0, 255), 3)
    dst_number = dst_number[22 :th + 10, 0: 180 ]
    
    #샤프닝 추가
    #dst_number = cv2.filter2D(dst_number, -1, kernel)
    
    #cv2.putText(dst, str(i), (500, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
    #dst_number = cv2.resize(dst_number, dsize=(380, 60), interpolation=cv2.INTER_CUBIC)
    
    return dst, dst_number


def bl(image, template):
    
    img2 = image.copy()
    
    # 입력 영상 밝기 50증가, 가우시안 잡음(sigma=10) 추가하여 약간의 변형을 줌
    noise = np.zeros(image.shape, np.int32)
    # cv2.randn은 가우시안 형태의 랜덤 넘버를 지정, 노이즈 영상에 평균이 50 시그마 10인 노이즈 추가
    cv2.randn(noise,50,10)
    
    # 노이즈를 입력 영상에 더함, 원래 영상보다 50정도 밝아지고 시그마 10정도 변형
    image = cv2.add(image, noise, dtype=cv2.CV_8UC3)
    
    res = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
    res_norm = cv2.normalize(res, None, 0,255, cv2.NORM_MINMAX, cv2.CV_8U)
    
    # 최댓값을 찾아야하므로 minmaxloc 사용, min, max, min좌표, max좌표 반환
    _, maxv, _, maxloc = cv2.minMaxLoc(res)
    
    dst = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    # 매칭 결과를 빨간색 사각형으로 표시
    # maxv가 어느 값 이상이여야지 잘 찾았다고 간주할 수 있다.
    th, tw = template.shape[:2]
       
    #cv2.rectangle(dst, maxloc, (maxloc[0] + tw, maxloc[1] + th), (0, 0, 255), 2)
    #cv2.rectangle(dst, (200, maxloc[1]), ( 100 , (th+10), (0, 0, 255), 2))
    
    #bl
    cv2.rectangle(dst, (50, maxloc[1]+20), ( 235 , maxloc[1] + (th)), (0, 0, 255), 3)
    #cargo
    cv2.rectangle(dst, (320, maxloc[1]+20), ( 550 , maxloc[1] + (th)), (0, 0, 255), 3)
   
    
    dst_bl = img2[maxloc[1]+20:  maxloc[1] + (th), 50 : 235]
    dst_cargo = img2[maxloc[1]+20:  maxloc[1] + (th), 320 : 550]
    
    #dst_bl = cv2.filter2D(dst_bl, -1, kernel)
    #dst_cargo = cv2.filter2D(dst_cargo, -1, kernel)
    
    #dst_bl = cv2.resize(dst_bl, dsize=(380, 60), interpolation=cv2.INTER_CUBIC)
    
    return dst, dst_bl, dst_cargo



def payment(image, template):
    
    img2 = image.copy()
    # 입력 영상 밝기 50증가, 가우시안 잡음(sigma=10) 추가하여 약간의 변형을 줌
    noise = np.zeros(image.shape, np.int32)
    # cv2.randn은 가우시안 형태의 랜덤 넘버를 지정, 노이즈 영상에 평균이 50 시그마 10인 노이즈 추가
    cv2.randn(noise,50,10)
    
    # 노이즈를 입력 영상에 더함, 원래 영상보다 50정도 밝아지고 시그마 10정도 변형
    image = cv2.add(image, noise, dtype=cv2.CV_8UC3)
    
    res = cv2.matchTemplate(image, template,cv2.TM_CCOEFF_NORMED)
    res_norm = cv2.normalize(res, None, 0,255, cv2.NORM_MINMAX, cv2.CV_8U)
    
    # 최댓값을 찾아야하므로 minmaxloc 사용, min, max, min좌표, max좌표 반환
    _, maxv, _, maxloc = cv2.minMaxLoc(res)
    
    # 탬플릿에 해당하는 영상이 입력 영상에 없으면 고만고만한 값에서 가장 큰 값을 도출.
    # 그래서 maxv를 임계값 0.7 or 0.6을 설정하여 템플릿 영상이 입력 영상에 존재하는지 파악
    #print('maxv : ', maxv)
    #print('maxloc : ', maxloc) 
    
    # 매칭 결과를 빨간색 사각형으로 표시
    # maxv가 어느 값 이상이여야지 잘 찾았다고 간주할 수 있다.
    th, tw = template.shape[:2]
    dst = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    #cv2.rectangle(dst, maxloc, (maxloc[0] + tw, maxloc[1] + th), (0, 0, 255), 2)
    #dst_pay = img2[maxloc[1]:  maxloc[1] + (th-5), 400 : 683]
    dst_pay = img2[maxloc[1]+10:  maxloc[1] + (th-6), 450 : 683]
   
    cv2.rectangle(dst, (450, maxloc[1]+10), ( 683 , maxloc[1] + (th-6)), (0, 0, 255), 3)

    #dst_pay = cv2.resize(dst_pay, dsize=(380, 50), interpolation=cv2.INTER_CUBIC)
    
    #샤프닝 추가
    #dst_pay = cv2.filter2D(dst_pay, -1, kernel)
    
    return dst, dst_pay

def template_matching(original):
    
    original = cv2.cvtColor( original, cv2.COLOR_BGR2GRAY )
    image = original.copy()
        
    w, h = image.shape
    #dst = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    #이미지 분할(정확도 상승 목적)
    top_image = image[:int(h/1.2), :]
    btm_image = image[int(h/1.2):, :]
    
    dst1, dst1_number = number(top_image, num_template)
    dst2, dst2_bl, dst2_cargo = bl(top_image, bl_template)
    dst3, dst3_pay = payment(btm_image, pay_template)
     
    #병합
    dst1 = cv2.addWeighted(dst1, alpha, dst2, (1-alpha), 0)
    dst = cv2.vconcat([dst1, dst3])
    
    '''  
    dst1_number = Image.from_array(dst1_number.astype('uint8'))
    dst1_number.unsharp_mask(radius=10,
                 sigma=4,
                 amount=1,
                 threshold=0)
    dst1_number = np.array(dst1_number)
    
    '''
    return dst, dst1_number, dst2_bl, dst2_cargo, dst3_pay
        
    
#테이블 줌 알고리즘 적용 
def zoom(original):
    
    img = original.clone()
    img2 = original.clone()
    
    img.deskew(0.4*img.quantum_range)
    img2.deskew(0.4*img2.quantum_range)
    
    #img.save(filename='D:/korea_docum_image/output2/5_late.jpg')
    #display(img)
    img = np.array(img)
    img2 = np.array(img2)
    
    rows, cols, ch = img.shape
    
    #이미지 분할(정확도 상승 목적)
    img = img[ : int(cols/0.89), :]
    img2 = img2[ : int(cols/0.89), :]
    
    #img = img[int(cols/9) : int(cols/0.89), :]
    
    #이미지 보간법으로 InterCubic 사용 - 이미지 확대시
    img = cv2.resize(img, (1286, 1809), interpolation = cv2.INTER_AREA)
    img2 = cv2.resize(img2, (1286, 1809), interpolation = cv2.INTER_AREA)
    
    
    # 이미지 확대
    '''
    window_name = '0216'
    cv2.namedWindow(window_name,flags = cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, img)
    cv2.waitKey(0)
    '''
    
    #샤프닝
    sharpening = np.array([[-1, -1, -1, -1, -1],
                             [-1, 2, 2, 2, -1],
                             [-1, 2, 9, 2, -1],
                             [-1, 2, 2, 2, -1],
                             [-1, -1, -1, -1, -1]]) / 9.0
           
    img = cv2.filter2D(img, -1, sharpening)
    
    
    gray_scale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, img_bin = cv2.threshold(gray_scale, 180, 225, cv2.THRESH_BINARY)
    img_bin = ~img_bin
    '''
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.imshow(window_name, img_bin)
    cv2.waitKey(0)
    '''
    line_min_width = 90
    kernal_h = np.ones((1, line_min_width) , np.uint8)
    kernal_v = np.ones((line_min_width, 1) , np.uint8)
    
    img_bin_h = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, kernal_h)
    img_bin_v = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, kernal_v)
    
    img_bin_final = img_bin_h|img_bin_v
    
    final_kernel = np.ones((3,3), np.uint8)
    img_bin_final = cv2.dilate(img_bin_final, final_kernel, iterations=1)
    
    _, labels, stats, _ = cv2.connectedComponentsWithStats(~img_bin_final,
                                                           connectivity=8)
    n1 = np.array(stats[2:])
    
    cnt = 0
    min_x = 0
    min_y = 0
    area_len = []
    for x,y,w,h,area in stats[2:]:
        
        
        if area<820:
            continue
        
        #cv2.rectangle(img, (x,y), (x+w, y+h),(0,0,255),2)
        
        if cnt == 0:
            min_x = x
            min_y = y
        
        if cnt >0:
            if x < min_x:
                min_x = x
            if y < min_y:
                min_y = y
        cnt +=1
        #print('(%d,%d), (%d,%d)' %(x,y,x+w,y+h))
        #print('('+x,',',y,')' , '(',x+w,',' ,y+h,')')
        area_len.append(area)
        
        
    #part1,5 sharpning 
    '''
    cv2.imshow(window_name, img)
    `
    cv2.waitKey(0)
    '''
    
    #print('사각형 추출개수:' , cnt)
    
    #전체표만 그리는 코드
    xw = n1[:, [0,2]].sum(axis=1).max()
    yh = n1[:, [1,3]].sum(axis=1).max()
    '''
    cv2.rectangle(img, (min_x, min_y), (xw, yh), (255,0,0), 2)
    
    #recognition
    
    cv2.imshow(window_name, img)
    cv2.waitKey(0)
    '''
    
    ##################
    #전체표 이미지 crop
    crop_img = img2[min_y: yh, min_x :  xw]
    
    ##################
    #don't give up
    
    #cv2.destroyAllWindows()
    crop_img = cv2.resize(crop_img, (1000, 1209), interpolation = cv2.INTER_CUBIC)
    #print(crop_img)
    return crop_img

@app.route('/')
def home():
    return render_template('index.html')
 
@app.route('/', methods=['POST'])
def upload_image():
    start = time.time()
    
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        print(file)
        print(filename)
        #file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        #print('upload_image filename: ' + filename)

        flash('Image successfully uploaded and displayed below')
        
        #이미지 처리
        
        #original = Image(filename='static/uploads/'+ filename)
        #업로드 이미지 형 변환: 업로드 -> numpy -> wand
        npimg = np.fromfile(file, np.uint8)
        file = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        original = Image.from_array(file)
        
        #table zoom
        crop_img = zoom(original)
        #template matching
        dst,  dst1_number, dst2_bl ,dst2_cargo, dst3_pay = template_matching(crop_img)
        
        # 후처리( wand(imageMagick))




        
        
        
        #table zoom 저장
        #cv2.imwrite(os.path.join(app.config['TABLEZOOM_FOLDER'], filename), crop_img)
        
        #전체 문서 rectangle 저장(4/20 ver)
        cv2.putText(dst, filename, (500, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
        cv2.imwrite(os.path.join(app.config['MATCHING_FOLDER'], filename), dst)
         
        #각 항목별 문서 저장(4.22)
        cv2.imwrite(os.path.join(app.config['NUMBER_FOLDER'], filename), dst1_number)
        cv2.imwrite(os.path.join(app.config['BL_FOLDER'], filename), dst2_bl)
        cv2.imwrite(os.path.join(app.config['CARGO_FOLDER'], filename), dst2_cargo)
        cv2.imwrite(os.path.join(app.config['PAY_FOLDER'], filename), dst3_pay)
        
        
        #recognition(4/25_)
        
        text1 = pytesseract.image_to_string(dst1_number, lang='eng',
                                            config="-c tessedit_char_whitelist=,-01234567890ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz --psm 6 --oem 1")
        
        text2 = pytesseract.image_to_string(dst2_bl, lang='eng', 
                                            config="-c tessedit_char_whitelist=,-01234567890ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz --psm 6 --oem 1")
       
        text3 = pytesseract.image_to_string(dst2_cargo, lang='eng', 
                                            config="-c tessedit_char_whitelist=,-01234567890ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz --psm 6 --oem 1")
       
        text4 = pytesseract.image_to_string(dst3_pay, lang='eng', 
                                            config="-c tessedit_char_whitelist=,-01234567890ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz --psm 6 --oem 1")
        
        print("time :", time.time() - start)  # 현재시각 - 시작시간 = 실행 시간
        
        return render_template('index.html', filename=filename, a = text1, b = text2, c = text3, d = text4, time = time.time() - start)
  
    else:
        flash('Allowed image types are - png, jpg, jpeg, gif')
        return redirect(request.url)


@app.route('/display/<filename>')
def display_image(filename):
    
    #print('display_image filename: ' + filename)
    filename = 'matching/' + filename
    
    print(filename)
    return redirect(url_for('static', filename=filename), code=301)


@app.route('/number/<filename>')
def display_number(filename):
    
    filename = 'category/number/' + filename
    print(filename)
    
    return redirect(url_for('static', filename=filename), code=301)

@app.route('/bl/<filename>')
def display_bl(filename):
    print(filename)
    
    filename = 'category/bl/' + filename
    return redirect(url_for('static', filename=filename), code=301)

@app.route('/cargo/<filename>')
def display_cargo(filename):
    
    filename = 'category/cargo/' + filename
    print(filename)
    
    return redirect(url_for('static', filename=filename), code=301)

@app.route('/pay/<filename>')
def display_pay(filename):
    
    filename = 'category/pay/' + filename
    print(filename)
    
    return redirect(url_for('static', filename=filename), code=301)



if __name__ == "__main__":
    app.run(host='210.181.192.104',port=8321 ,debug=True)