from imutils.video import VideoStream
from imutils.video import FPS
from imutils.object_detection import non_max_suppression
import numpy as np
import argparse
import imutils
import time
import cv2
import pytesseract
from project_crnn import CRNN
import tensorflow as tf
import func
import project_classes
import string
import multiprocessing
from signal import signal, SIGPIPE, SIG_DFL
signal(SIGPIPE, SIG_DFL)

char_list = string.ascii_letters+string.digits

layerNames = [
	"feature_fusion/Conv_7/Sigmoid",
	"feature_fusion/concat_3"]


def image_processing(frame, rW, rH, net, child_conn):


	model = tf.keras.models.load_model("my_model.h5")
	ret_values = []
	if best_years["stopped"]==True:

		orig = frame.copy()
		frame = cv2.resize(frame, (320, 320))
		blob = cv2.dnn.blobFromImage(frame, 1.0, (320, 320),
			(123.68, 116.78, 103.94), swapRB=True, crop=False)
		net.setInput(blob)
		(scores, geometry) = net.forward(layerNames)
		(rects, confidences) = func.decode_predictions(scores, geometry)
		boxes = non_max_suppression(np.array(rects), probs=confidences)

		for (startX, startY, endX, endY) in boxes:
			print("inside boxes")
			op=""
			startX = int(startX * rW)
			startY = int(startY * rH)
			endX = int(endX * rW)
			endY = int(endY * rH)
		 
			r = orig[startY:startY+32, startX:endX]
			r = np.mean(r, -1)
			r = np.expand_dims(r, axis=0)
			print("r ki preprocessing done")
			prediction = model.predict(r, batch_size=None)
			print("prediction done")

			out = tf.keras.backend.get_value(tf.keras.backend.ctc_decode(prediction, input_length=np.ones(prediction.shape[0])*prediction.shape[1],
			                         greedy=True)[0][0])
			print("got output")

			for x in out:  
				for p in x:  
				    if int(p) != -1:
				        op += char_list[int(p)]

			print("We have reached the return statement")
			ret_values = [ startX, startY, endX, endY, op]
			
		if ret_values is not None:
			child_conn.send(ret_values)
		print("If condition not fulfilled with ret_values{}".format(ret_values) )




ap = argparse.ArgumentParser()
ap.add_argument("-east", "--east", type=str, required=True,
	help="path to input EAST text detector")
ap.add_argument("-c", "--min-confidence", type=float, default=0.5,
	help="minimum probability required to inspect a region")
args = vars(ap.parse_args())


(W, H) = (None, None)
(rW, rH) = (None, None)
 

net = cv2.dnn.readNet(args["east"])


print("[INFO] starting video stream...")
vs = VideoStream(src=-1).start()
time.sleep(1.0)

fps = project_classes.FPS().start()
 
processes = []

manager = multiprocessing.Manager()
ret_values = manager.dict()
best_years = manager.dict()
thin_white_lies = manager.dict()

thin_white_lies["killingme"] = False
best_years["stopped"] = True

i = 0
parent_conn, child_conn = multiprocessing.Pipe()

while True:

	frame = vs.read()
 
	if frame is None:
		print("No input found")
		break
 
	frame = imutils.resize(frame, width=1000)

	if W is None or H is None:
		(H, W) = frame.shape[:2]
		rW = W / float(320)
		rH = H / float(320)


	if (i%128==0):


		p = multiprocessing.Process(target=image_processing, args=[frame, rW, rH, net, child_conn])
		processes.append(p)
		p.start()



	if parent_conn.poll():
		ret_values = parent_conn.recv()

	try:
		print(ret_values)
		cv2.putText(frame, ret_values[4], (ret_values[0], ret_values[1] + 32),
		cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0,0, 255), 2)
		cv2.rectangle(frame, (ret_values[0], ret_values[1]), 
			(ret_values[2], ret_values[1]+32), (0, 255, 0), 2)

	except Exception as ex:
		pass
		print("Caught:{}".format(type(ex).__name__))


	fps.update()
	cv2.imshow("Text Detection", frame)
	key = cv2.waitKey(50) & 0xFF
	i = (i+1)%128

	if key == ord("q"):
		break
	 
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
 
vs.stop()
cv2.destroyAllWindows()

for proc in processes:
	print("Finishing")
	proc.join()
	print("Finished")
print("ret_values: {}".format(ret_values))
