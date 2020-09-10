import cv2
import boto3
import datetime
import requests
face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")
ds_factor=0.6

count=0

class VideoCamera(object):    
    def __init__(self):
        self.video = cv2.VideoCapture(0)
    
    def __del__(self):
        self.video.release()
    
    def get_frame(self):
        #count=0
        global count
        success, image = self.video.read()
        is_success, im_buf_arr = cv2.imencode(".jpg", image)
        image1 = im_buf_arr.tobytes()
        client=boto3.client('rekognition',
                        aws_access_key_id="ASIAY27ZBXWAUAI3JM7O",
                        aws_secret_access_key="xSuw+smplbTnd8lkKK6ZagrNaEfZL4HYw+kGXmYG",
                        aws_session_token="FwoGZXIvYXdzEG0aDGxKGfB6vPakFIUjHiLJAc1ohx8l5SG/97aAJEwSzwyLH3lz1iAt30t/6QKGt9XCQ4+TyGSFynmxoymZhIrcKLmaZUYXxKQYFk9jkjytdhGSuwe+paJlo9zmHfg2SnopuS+CHV92cMRdoQ9UTlM9t5xJxo8apFdKTCS74THVo/MtU4f1TxV9oidVuwOOX6Mgm0vEHwZJuc7I5dykD0yg80pYfOq+Ppp8NofVwNQ/gTJimMmS52b5amQGSAJSj/i5qJWoP9zFuxujR6sWpNGfKLLG2TU/sxKn/Cj5ier6BTItd0YfYagHL8jF7T7JnYvXsvcwCVXV9WRySmTaJk5FnShWalYUilgZckF0N9wy",
                        region_name='us-east-1')
        response = client.detect_custom_labels(
        ProjectVersionArn='arn:aws:rekognition:us-east-1:607723306369:project/my-masks/version/my-masks.2020-09-11T00.06.01/1599762961105',Image={
            'Bytes':image1})
        print(response['CustomLabels'])
        
        if not len(response['CustomLabels']):
            count=count+1
            date = str(datetime.datetime.now()).split(" ")[0]
            #print(date)
            url = "https://m6f7rqpk3e.execute-api.us-east-1.amazonaws.com/maskdeploy01?date="+date+"&count="+str(count)
            resp = requests.get(url)
            f = open("countfile.txt", "w")
            f.write(str(count))
            f.close()
            #print(count)

        image=cv2.resize(image,None,fx=ds_factor,fy=ds_factor,interpolation=cv2.INTER_AREA)
        
        gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        face_rects=face_cascade.detectMultiScale(gray,1.3,5)
        for (x,y,w,h) in face_rects:
        	cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
        	break
        ret, jpeg = cv2.imencode('.jpg', image)
        #cv2.putText(image, text = str(count), org=(10,40), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=(1,0,0))
        cv2.imshow('image',image)
        return jpeg.tobytes()
