import torch
import numpy as np
import cv2
import time
import mss
from paddleocr import PaddleOCR

class ObjectDetection:
    """
    Class implements Yolo5 model to make inferences on a youtube video using OpenCV .
    """
    
    def __init__(self):
        """
        Initializes the class with youtube url and output file.
        :param url: Has to be as youtube URL,on which prediction is made.
        :param out_file: A valid output file name.
        """
        self.model = self.load_model()
        self.classes = self.model.names
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("\n\nDevice Used:",self.device)



    def load_model(self):
        return torch.hub.load('yolov5', model='custom', path='LicensPlateModel2.pt', source='local', force_reload=True)



    def score_frame(self, frame):
        """
        Takes a single frame as input, and scores the frame using yolo5 model.
        :param frame: input frame in numpy/list/tuple format.
        :return: Labels and Coordinates of objects detected by model in the frame.
        """
        self.model.to(self.device)
        frame = [frame]
        results = self.model(frame)
     
        labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
        return labels, cord


    def class_to_label(self, x):
        """
        For a given label value, return corresponding string label.
        :param x: numeric label
        :return: corresponding string label
        """
        return self.classes[int(x)]


    def plot_boxes(self, results, frame):
 
        labels, cord = results
        n = len(labels)
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        bboxes = []
        for i in range(n):
            row = cord[i]
            if row[4] >= 0.7:
                x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
                bboxes.append((x1, y1, x2, y2))
                bgr = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
                cv2.putText(frame, self.class_to_label(labels[i]), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)

        return frame, bboxes
    
    def OCR(self, frame):
        ocr = PaddleOCR(use_angle_cls=True, lang='ar') # need to run only once to download and load model into memory
        img_path = frame
        result = ocr.ocr(img_path, cls=True)
        for idx in range(len(result)):
            res = result[idx]
            for line in res:
                print(f'car number is {line[1][0]}')
        return result


    def __call__(self):
        """
        This function is called when class is executed, it runs the loop to read the video frame by frame,
        and write the output into a new file.
        :return: void
        """
        # with mss.mss() as sct:
        #     monitor = {"top": 0, "left": 0, "width": 1920, "height": 1080}
        #     while True:
        #         last_time = time.time()
        #         img = np.array(sct.grab(monitor))
        #         img_resize = cv2.resize(img, (640, 640))
        #         results = detection.score_frame(img_resize)

        #         img, bboxes = detection.plot_boxes(results, img_resize)

        #         # Display the ROIs of each bounding box
        #         for bbox in bboxes:
        #             roi = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        #             cv2.imshow("ROI", roi)

        #         print("FPS: {}".format(1 / (time.time() - last_time)))
        #         cv2.imshow("OpenCV/Numpy normal", img)
        #         if cv2.waitKey(25) & 0xFF == ord("q"):
        #             cv2.destroyAllWindows()
        #             break



        # read pictures
        img = cv2.imread('pic/9.jpg')
        img_resize = cv2.resize(img, (640, 640))
        results = detection.score_frame(img_resize)
        img , bboxes = detection.plot_boxes(results, img_resize)
        for bbox in bboxes:
            roi = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
            # sharpen the image
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])                      
            roi = cv2.filter2D(roi, -1, kernel)
            
            cv2.imshow("ROI", roi)
            cv2.imwrite('roi.jpg', roi)
            self.OCR(roi) # OCR

            
        
        cv2.imshow("OpenCV/Numpy normal", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()



# Create a new object and execute.
detection = ObjectDetection()
detection()
