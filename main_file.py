# Core Pkgs
import streamlit as st 
import cv2
from PIL import Image,ImageEnhance
import numpy as np 
import os

# Load the model
net = cv2.dnn.readNetFromTensorflow("dnn/frozen_inference_graph_coco.pb",
									"dnn/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt")

# Store Coco Names in a list
classesFile = "coco.names"
classNames = open(classesFile).read().strip().split('\n')
print(classNames)


def detect_Object(our_image):
	new_img = np.array(our_image.convert('RGB'))
	height, width, _ = new_img.shape
	img = cv2.cvtColor(new_img,1)
	gray = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)

	# Create black image
	blank_mask = np.zeros((height, width, 3), np.uint8)
	blank_mask[:] = (0, 0, 0)
	
	# Create blob from the image
	blob = cv2.dnn.blobFromImage(new_img, swapRB=True)
	# Detect objects
	net.setInput(blob)
	boxes, masks = net.forward(["detection_out_final", "detection_masks"])
	detection_count = boxes.shape[2]
	print(detection_count)
	count = 0

	# Draw rectangle around the faces
	for i in range(detection_count):
            # Extract information from detection
            box = boxes[0, 0, i]
            class_id = int(box[1])
            score = box[2]
            # print(class_id, score)
            if score < 0.6:
                continue
            # print(class_id)
            class_name = (classNames[class_id])
            # print(class_name, score)
            x = int(box[3] * width)
            y = int(box[4] * height)
            x2 = int(box[5] * width)
            y2 = int(box[6] * height)
            roi = blank_mask[y: y2, x: x2]
            roi_height, roi_width, _ = roi.shape
            # Get the mask
            mask = masks[i, int(class_id)]
            mask = cv2.resize(mask, (roi_width, roi_height))
            _, mask = cv2.threshold(mask, 0.5, 255, cv2.THRESH_BINARY)
            # cv2.imshow("mask"+str(count), mask)
            count+=1
            # Find contours of the mask
            contours, _ = cv2.findContours(np.array(mask, np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            color = np.random.randint(0, 255, 3, dtype='uint8')
            color = [int(c) for c in color]
            # fill some color in segmented area
            for cnt in contours:
                cv2.fillPoly(roi, [cnt], (int(color[0]), int(color[1]), int(color[2])))
                # cv2.imshow("roi", roi)
            # Draw bounding box
            cv2.rectangle(img, (x, y), (x2, y2), color, 2)
            cv2.putText(img, class_name + " " + str(score), (x, y-5), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)

        
        
	return img 

def main():
	"""Object Detection App"""

	
	activities = ["Detection","About"]
	choice = st.sidebar.selectbox("Select Activty",activities)

	if choice == 'Detection':
		st.subheader("Object Detection")
		st.title("Object Detection App")
		img = Image.open("C://Users//Welcome//Desktop//project//object ditection//C&S.jpg")
		st.image(img)

		image_file = st.file_uploader("Upload Image",type=['jpg','png','jpeg'])

		show_image_file = st.empty()

		if not image_file:
                    show_image_file.info("Please upload a photo of type: " + ", ".join(["jfif", "png", "jpg"]))
                    return

		if image_file is not None:
			our_image = Image.open(image_file)
			st.text("Original Image")
			# st.write(type(our_image))
			st.image(our_image)

		enhance_type = st.sidebar.radio("Enhance Type",["Original","Gray-Scale","Contrast","Brightness","Blurring"])
		if enhance_type == 'Gray-Scale':
			new_img = np.array(our_image.convert('RGB'))
			img = cv2.cvtColor(new_img,1)
			gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
			# st.write(new_img)
			st.image(gray)
		elif enhance_type == 'Contrast':
			c_rate = st.sidebar.slider("Contrast",0.5,3.5)
			enhancer = ImageEnhance.Contrast(our_image)
			img_output = enhancer.enhance(c_rate)
			st.image(img_output)

		elif enhance_type == 'Brightness':
			c_rate = st.sidebar.slider("Brightness",0.5,3.5)
			enhancer = ImageEnhance.Brightness(our_image)
			img_output = enhancer.enhance(c_rate)
			st.image(img_output)

		elif enhance_type == 'Blurring':
			new_img = np.array(our_image.convert('RGB'))
			blur_rate = st.sidebar.slider("Brightness",0.5,3.5)
			img = cv2.cvtColor(new_img,1)
			blur_img = cv2.GaussianBlur(img,(11,11),blur_rate)
			st.image(blur_img)
           
		elif enhance_type == 'Original':
			st.image(our_image,width=300)
		else:
			st.image(our_image,width=300)



		# Face Detection
		task = ["Object"]
		feature_choice = st.sidebar.selectbox("Find Features",task)
		if st.button("Process"):

			if feature_choice == 'Object':
				result_img = detect_Object(our_image)
				st.text("object detected Image")
				st.image(result_img)

				
			
	elif choice == 'About':
		st.subheader("About Object Detection App")
		st.markdown("made with LOVE")
		st.text("Raj R Pawar4")
		st.success("Raj")



if __name__ == '__main__':
		main()	
