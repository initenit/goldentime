import cv2
import numpy as np
import tensorflow as tf
import webbrowser

print(tf.config.list_physical_devices('GPU'))

net = cv2.dnn.readNet('yolov4-custom_final.weights', 'yolov4-custom.cfg')
classes = []
with open('object.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))
label =''

#3초 동영상 준비과정
VideoSignal = cv2.VideoCapture('rtsp://initenit:hwprint2@192.168.0.190:554/stream_ch00_0')
print('width :%d, height : %d' % (VideoSignal.get(3), VideoSignal.get(4)))
fourcc = cv2.VideoWriter_fourcc(*'XVID')
fps = 10
fallen_video_count = 0




shape = (416, 416)

while True:

    fallen_video_name = str(fallen_video_count) + '.avi'
    output = cv2.VideoWriter(fallen_video_name, fourcc, fps, (1280, 720))
    frame_ = []
    frame_2 = []
    begin = 0
    # 웹캠 신호 받기
    while True:

        ret, img = VideoSignal.read()
        height, width, channels  = img.shape
        blob = cv2.dnn.blobFromImage(img, 0.00392, shape, (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        font = cv2.FONT_HERSHEY_PLAIN
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                if label == 'fallen':
                    color = colors[0]
                else:
                    color = colors[1]
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                cv2.putText(img, label, (x, y + 30), font, 3, color, 3)
        cv2.imshow("Image", img)

        if (ret):

            if begin == 0 and len(frame_) < fps*3:
                frame_.append(img)
            else:
                frame_ = frame_[1:]
                frame_.append(img)

            if label == 'fallen' and begin == 0:
                print("넘어짐이 감지되었습니다.")
                begin = 1
                for i in range(len(frame_)):
                    output.write(frame_[i])
                    filepath = "alert.html"
                    message = """
                    <!DOCTYPE html><html lang="ko"><head><title></title><meta charset="utf-8">
                    <link rel="stylesheet" href="css/alert.css" /></head><body>
                    <div class="outer"><div class="title">
                    <p id = "title_name">EMERGENCY</p>
                    <img class="siren" src="img/siren.png"></div>
                    <div class="wapper">
                    <div class ="box">
                    <video autoplay controls loop muted class="video_box">
                    <source class ="source" src="""
                    message += fallen_video_name
                    message += """ type="video/mp4">
                    </video></div>
                    <input type="submit" value="CALL 119" class="button" id="alertButton">
                    </div></div></body></script></html>
                    """
                    with open(filepath, 'w') as f:
                        f.write(message)
                        f.close()
                    webbrowser.open_new_tab(filepath)
            if begin == 1:
                frame_2.append(img)
                if len(frame_2) == fps*3:
                    for i in range(fps*3):
                        output.write(frame_2[i])
                    begin = 0
                    fallen_video_count += 1
                    break

        k = cv2.waitKey(33)
        if k>0:    # Esc key to stop
            break

cv2.destroyAllWindows()
