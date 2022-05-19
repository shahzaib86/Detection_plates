import os
import cv2 # image prcoessing
from matplotlib import pyplot as plt # for plotting or displaying the image
import numpy as np
import imutils # edge detection, sorting countours
import easyocr # text extraction from image
import sqlite3 # database
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename

app = Flask(__name__, template_folder='template')

@app.route('/',  methods=['GET', 'POST'])
def numberplate_recognition():
    if request.method == 'POST':
        file = request.files['file']
        print('fffff', file)
        filename = secure_filename(file.filename)
        print(filename)
        file.save(os.path.join(r'C:\Users\tanol\PycharmProjects\numberplate\static', 'real_file.png'))
        # matploit(filename=filename)
        return redirect(url_for('matploit', filename=filename))
    return render_template('index.html')

@app.route('/result', methods=['GET', 'POST'])
def matploit():
    filename = r'C:\Users\tanol\PycharmProjects\numberplate\static\real_file.png'
    # Convert Colored image to grayscale format
    img = cv2.imread ( filename )  # reading the image
    # plt.imshow ( img )
    gray = cv2.cvtColor ( img,cv2.COLOR_BGR2GRAY )  # converting RBG to gray image which will speed up the image processing process.
    plt.imshow(cv2.cvtColor(gray, cv2.COLOR_BGR2RGB)) # converting back to rgb from gray scale and displaying the image.
    plt.savefig(r'C:\Users\tanol\PycharmProjects\numberplate\static\gray_image.png' )

    # Applying filter and edge detection
    bfilter = cv2.bilateralFilter ( gray, 11, 17, 17 )  # noise reduction
    edged = cv2.Canny ( bfilter, 30, 200 )  # edge detection
    plt.imshow(cv2.cvtColor(edged, cv2.COLOR_BGR2RGB))
    plt.savefig(r'C:\Users\tanol\PycharmProjects\numberplate\static\edged_image.png')

    # finding contours and applying mask
    keypoints = cv2.findContours ( edged.copy (), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE )
    # print('key', keypoints)
    contours = imutils.grab_contours ( keypoints )
    # print('con', contours)
    contours = sorted ( contours, key=cv2.contourArea, reverse=True )[0: 10]

    location = None
    for contour in contours:
        approx = cv2.approxPolyDP ( contour, 10, True )
        if len ( approx ) == 4:
            location = approx
            break

    mask = np.zeros ( gray.shape, np.uint8 )
    new_image = cv2.drawContours ( mask, [location], 0, 255, -1 )
    new_image = cv2.bitwise_and ( img, img, mask=mask )

    plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))
    plt.savefig(r'C:\Users\tanol\PycharmProjects\numberplate\static\numberplate_detected.png' )

    (x, y) = np.where ( mask == 255 )
    (x1, y1) = (np.min ( x ), np.min ( y ))
    (x2, y2) = (np.max ( x ), np.max ( y ))
    cropped_image = gray[x1:x2 + 1, y1:y2 + 1]
    plt.imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
    plt.savefig(r'C:\Users\tanol\PycharmProjects\numberplate\static\cropped_image.png')

    reader = easyocr.Reader ( ['en'] )
    result = reader.readtext ( cropped_image )
    print(result[0])

    if len ( result ) == 1:
        text = result[0][-2]
    else:

        text = result[0][-2] + " " + result[2][-2]

    font = cv2.FONT_HERSHEY_SIMPLEX
    res = cv2.putText ( img, text=text, org=(approx[0][0][0], approx[1][0][1] + 60), fontFace=font, fontScale=1,
                        color=(0, 255, 0), thickness=2 )
    res = cv2.rectangle ( img, tuple ( approx[0][0] ), tuple ( approx[2][0] ), (0, 0, 0), 3 )
    k = cv2.cvtColor ( res, cv2.COLOR_BGR2RGB )
    plt.imshow ( k )

    plt.savefig( r'C:\Users\tanol\PycharmProjects\numberplate\static\hii.png' )
    plt.close ()
    datbase(value=text)
    mylist = [text]
    return render_template('result.html', mylist=mylist)

def datbase(value):
    k = value.split(' ')
    if len(k) < 3:
        if k[1] == 'ADA339':
            naam = 'Shahziab'
            num = value

        else:
            naam = 'Zaynab'
            num = value
    else:
        naam = 'Zaynab'
        num = value

    conn = sqlite3.connect('cars_info.db')

    connection = conn.cursor()
    connection.execute(""" CREATE TABLE IF NOT EXISTS INFO (
            Name VARCHAR(255) NOT NULL,
            Number VARCHAR(25) NOT NULL); """)

    sql = '''INSERT INTO INFO (Name, Number) VALUES(?,?)'''
    data = (naam, num)

    try:
        # Executing the SQL command
        connection.execute(sql, data)

        # Commit your changes in the database
        conn.commit()

    except:
        # Rolling back in case of error
        conn.rollback()

    conn.commit()


    connection.execute("SELECT * FROM INFO")

    # fetch all the matching rows
    result = connection.fetchall()

    # loop through the rows
    for row in result:
        print('row', row)
        print("\n")




# numberplate_recognition()

if __name__ == "__main__":

    app.run(debug=True)