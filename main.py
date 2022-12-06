# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import cv2
import numpy as np
import os
# from imageai.Detection import ObjectDetection
from skimage import data, filters


def extractFrames(name):
    # Opens the Video file
    if not os.path.exists(name):
        os.makedirs(name)
    else:
        return

    cap = cv2.VideoCapture(name + '.mp4')
    i = 0
    print(cap.isOpened())
    while cap.isOpened():
        ret, frame = cap.read()
        print(ret)
        if not ret:
            break
        cv2.imwrite(name + '/' + name + str(i) + '.jpg', frame)
        i += 1

    cap.release()
    cv2.destroyAllWindows()


def displayVideo():
    cap = cv2.VideoCapture('Stairs.mp4')
    # Check if camera opened successfully
    if cap.isOpened() == False:
        print("Error opening video stream or file")

    # Read until video is completed
    while cap.isOpened():
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret:

            # Display the resulting frame
            cv2.imshow('Frame', frame)

            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        # Break the loop
        else:
            break

    # When everything done, release the video capture object
    cap.release()

    # Closes all the frames
    cv2.destroyAllWindows()


def createPano(name):
    extractFrames(name)
    imgs = []
    i = 0
    # 403 test1
    # 630 test2
    # 594 test3
    while i < 594:
        imgs.append(cv2.imread(name + '/' + name + str(i) + ".jpg"))
        # cv2.imshow(str(i), imgs[i])
        print(i)
        i = i + 10

    stitchy = cv2.Stitcher.create()
    (dummy, output) = stitchy.stitch(imgs)
    if dummy != cv2.STITCHER_OK:
        # checking if the stitching procedure is successful
        # .stitch() function returns a true value if stitching is
        # done successfully
        print("stitching ain't successful")
    else:
        print('Your Panorama is ready!!!')

    # final output
    cv2.imshow('final result', output)
    cv2.imwrite('panorama_' + name + '_with_front.jpg', output)

    cv2.waitKey(0)


# not really working
def backgroundRemoval():
    # Open Video
    cap = cv2.VideoCapture('Stairs.mp4')

    # Randomly select 25 frames
    frameIds = cap.get(cv2.CAP_PROP_FRAME_COUNT) * np.random.uniform(size=25)

    # Store selected frames in an array
    frames = []
    for fid in frameIds:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
        ret, frame = cap.read()
        frames.append(frame)

    # Calculate the median along the time axis
    medianFrame = np.median(frames, axis=0).astype(dtype=np.uint8)

    # Display median frame
    cv2.imshow('frame', medianFrame)
    cv2.waitKey(0)


# not really used either
def calculateMotionVector1():
    # https://learnopencv.com/optical-flow-in-opencv/
    cap = cv2.VideoCapture('Stairs.mp4')
    # params for ShiTomasi corner detection
    feature_params = dict(maxCorners=100,
                          qualityLevel=0.3,
                          minDistance=7,
                          blockSize=7)
    # Parameters for lucas kanade optical flow
    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    # Create some random colors
    color = np.random.randint(0, 255, (100, 3))
    # Take first frame and find corners in it
    ret, old_frame = cap.read()
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
    # Create a mask image for drawing purposes
    mask = np.zeros_like(old_frame)
    while (1):
        ret, frame = cap.read()
        if not ret:
            print('No frames grabbed!')
            break
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
        # Select good points
        if p1 is not None:
            good_new = p1[st == 1]
            good_old = p0[st == 1]
        # draw the tracks
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
            frame = cv2.circle(frame, (int(a), int(b)), 5, color[i].tolist(), -1)
        img = cv2.add(frame, mask)
        cv2.imshow('frame', img)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
        # Now update the previous frame and previous points
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)
    cv2.destroyAllWindows()


# The one really used
# colored videos
def calculateMotionVector2(name, grayOut=False):
    # https://learnopencv.com/optical-flow-in-opencv/
    # https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_video/py_lucas_kanade/py_lucas_kanade.html
    cap = cv2.VideoCapture(name + '.mp4')
    videoSize = (640, 480)
    ret, frame1 = cap.read()
    frame1 = cv2.resize(frame1, videoSize)
    prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

    hsv = np.zeros_like(frame1)
    hsv[..., 1] = 255

    out = cv2.VideoWriter(
        'colored' + name + '.mp4',
        cv2.VideoWriter_fourcc(*'mp4v'),
        20,
        videoSize)

    while 1:
        ret, frame2 = cap.read()
        if not ret:
            print('No frames grabbed!')
            break
        frame2 = cv2.resize(frame2, videoSize)
        next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        # for i in range(0, hsv.shape[0]):
        #     for j in range(0, hsv.shape[1]):
        #         # val = hsv[i][j]
        #         # if val[1] < 43 and 46 < val[2] < 220:
        #         hsv[i][j][0] = 255
        #         hsv[i][j][1] = 255
        #         hsv[i][j][2] = 255

        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        if grayOut:
            bgr = cv2.cvtColor(hsv, cv2.COLOR_BGR2GRAY)
        # out.write(bgr)
        out.write(bgr.astype('uint8'))

        cv2.imshow('frame2', bgr)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
        # elif k == ord('s'):
        # cv2.imshow('opticalfb.png', frame2)
        # cv2.imshow('opticalhsv.png', bgr)
        prvs = next
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    extractFrames('colored' + name)


# Not really used
# The result isn't really useful
# but can be presented
def edgeDetection():
    # Read the original image
    img = cv2.imread('coloredwalking/coloredwalking20.jpg')
    # Display original image
    cv2.imshow('Original', img)
    cv2.waitKey(0)

    # Convert to graycsale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Blur the image for better edge detection
    img_blur = cv2.GaussianBlur(img_gray, (3, 3), 0)

    # Sobel Edge Detection
    sobelx = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5)  # Sobel Edge Detection on the X axis
    sobely = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5)  # Sobel Edge Detection on the Y axis
    sobelxy = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5)  # Combined X and Y Sobel Edge Detection
    # Display Sobel Edge Detection Images
    name = 'walking'
    cv2.imshow('Sobel X', sobelx)
    cv2.imwrite('edgeDetection_step1_colored_' + name + '.jpg', sobelx)
    cv2.waitKey(0)
    cv2.imshow('Sobel Y', sobely)
    cv2.imwrite('edgeDetection_step2_colored_' + name + '.jpg', sobelx)
    cv2.waitKey(0)
    cv2.imshow('Sobel X Y using Sobel() function', sobelxy)
    cv2.imwrite('edgeDetection_step3_colored_' + name + '.jpg', sobelx)
    cv2.waitKey(0)

    # Canny Edge Detection
    edges = cv2.Canny(image=img_blur, threshold1=100, threshold2=200)  # Canny Edge Detection
    # Display Canny Edge Detection Image
    cv2.imshow('Canny Edge Detection', edges)
    cv2.imwrite('edgeDetection_step4_colored_' + name + '.jpg', sobelx)
    cv2.waitKey(0)

    cv2.destroyAllWindows()


# not complete
def objectDetection():
    execution_path = os.getcwd()

    detector = ObjectDetection()
    detector.setModelTypeAsRetinaNet()
    detector.setModelPath(os.path.join(execution_path, "resnet50_coco_best_v2.1.0.h5"))
    detector.loadModel()
    detections = detector.detectObjectsFromImage(input_image=os.path.join(execution_path, "stairs/stairs0.jpg"),
                                                 output_image_path=os.path.join(execution_path, "stairsAI/stairs0.jpg"))

    for eachObject in detections:
        print(eachObject["name"], " : ", eachObject["percentage_probability"])


# Use HOG to detect people
# Reset to size 480*640 for speed purpose
def objectDetection2(name):
    # initialize the HOG descriptor/person detector
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    cv2.startWindowThread()

    # open webcam video stream
    cap = cv2.VideoCapture(name + '.mp4')
    videoSize = (640, 480)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # the output will be written to output.avi
    out = cv2.VideoWriter(
        'opencvAI' + name + '.mp4',
        cv2.VideoWriter_fourcc(*'mp4v'),
        20,
        videoSize)
    frameCount = 0
    while (True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        frame = cv2.resize(frame, videoSize)
        if not ret:
            print('No frames grabbed!')
            break
        # resizing for faster detection
        # frame = cv2.resize(frame, (width, height))
        # using a greyscale picture, also for faster detection
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        # detect people in the image
        # returns the bounding boxes for the detected objects
        boxes, weights = hog.detectMultiScale(frame, winStride=(8, 8))

        boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])
        # https://stackoverflow.com/questions/15341538/numpy-opencv-2-how-do-i-crop-non-rectangular-region
        if not os.path.exists(name + 'AI'):
            os.makedirs(name + 'AI')
        for (xA, yA, xB, yB) in boxes:
            # display the detected boxes in the colour picture
            for i in range(0, videoSize[1]):
                for j in range(0, videoSize[0]):
                    if not (xA < j < xB and yA < i < yB):
                        frame[i, j] = (255, 255, 255)

            # cropped_image = frame[yA:yB, xA:xB]
            # cv2.rectangle(frame, (xA, yA), (xB, yB),
            #               (0, 255, 0), 2)
            # mask = np.zeros(frame.shape, dtype=np.uint8)
            # roi_corners = np.array([[(xA, yA), (xB, yA), (xA, yB), (xB, yB)]], dtype=np.int32)
            # channel_count = frame.shape[2]  # i.e. 3 or 4 depending on your image
            # ignore_mask_color = (255,) * channel_count
            # cv2.fillPoly(mask, roi_corners, ignore_mask_color)
            # # from Masterfool: use cv2.fillConvexPoly if you know it's convex
            #
            # # apply the mask
            # masked_image = cv2.bitwise_and(frame, mask)

            # save the result

        cv2.imwrite(name + 'AI/' + name + 'AI' + str(frameCount) + '.jpg', frame)
        # cv2.imwrite('stairsAI/stairsAI'+str(i)+'.png', masked_image)

        # Write the output video
        out.write(frame.astype('uint8'))
        # Display the resulting frame
        cv2.imshow('frame', frame)
        frameCount += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    # and release the output
    out.release()
    # finally, close the window
    cv2.destroyAllWindows()
    cv2.waitKey(1)


def runFrontRemovalBasedOnMotionVector(name):
    if not os.path.exists(name + 'ColorFilter'):
        os.makedirs(name + 'ColorFilter')
    cap = cv2.VideoCapture(name + '.mp4')
    videoSize = (640, 480)
    ret, frame1 = cap.read()
    frame1 = cv2.resize(frame1, videoSize)
    prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

    hsv = np.zeros_like(frame1)
    hsv[..., 1] = 255

    out = cv2.VideoWriter(
        'filtered_' + name + '.mp4',
        cv2.VideoWriter_fourcc(*'mp4v'),
        20,
        videoSize)
    out.write(frame1)
    frameCount = 0
    while 1:
        ret, frame2 = cap.read()
        ret, frame2 = cap.read()
        ret, frame2 = cap.read()
        if not ret:
            print('No frames grabbed!')
            break
        frame2 = cv2.resize(frame2, videoSize)
        next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        for i in range(0, bgr.shape[0]):
            for j in range(0, bgr.shape[1]):
                # for walking
                # if not (bgr[i][j][0] < 10) or (bgr[i][j][0] < 5 and bgr[i][j][1] < 5 and bgr[i][j][2] < 5):
                # for test 1
                # if (not (bgr[i][j][0] < 100 or bgr[i][j][1] < 100)) or (bgr[i][j][2] > 100):
                # for test 2
                # if not ((bgr[i][j][0] < 10 and bgr[i][j][1] < 10) or (bgr[i][j][2] < 10 and bgr[i][j][1] > 60 and bgr[i][j][0] > 60) or (bgr[i][j][1] < 30 and 40 < bgr[i][j][2] < 70) ):
                # test 3
                if not ((bgr[i][j][0] < 10 and bgr[i][j][1] < 10 and bgr[i][j][1] < 10) or (bgr[i][j][1] > 80)):
                    frame2[i][j][0] = 255
                    frame2[i][j][1] = 255
                    frame2[i][j][2] = 255

        out.write(frame2)
        cv2.imwrite(name + 'ColorFilter/' + name + 'ColorFilter' + str(frameCount) + '.jpg', frame2)

        cv2.imshow('frame2', frame2)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
        prvs = next
        frameCount += 3
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    extractFrames('colored' + name)


def runTestData():
    createPano('walking')


def main():
    # runFrontRemovalBasedOnMotionVector('test1')
    # runFrontRemovalBasedOnMotionVector('test2')
    # runFrontRemovalBasedOnMotionVector('test3')
    createPano('test3')


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
