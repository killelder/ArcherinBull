import cv2
import numpy as np  

def displayIMG(img , winname):
	cv2.namedWindow(winname, cv2.WINDOW_NORMAL)
	cv2.resizeWindow(winname, 600, 600)
	cv2.imshow(winname, img)
	cv2.waitKey(0)

def get_specific_region(img, low_threshold, high_threshold):
	"""
		filter the region with the specific color
	"""
	imgHSV = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

	lower = np.array(low_threshold)
	upper = np.array(high_threshold)
	mask = cv2.inRange(imgHSV,lower,upper)
	imgResult = cv2.bitwise_and(img,img,mask=mask)
	return imgResult

def find_circles(img, pltimg, cen, rad):
	"""
		find the circles
	"""
	#data preprocessing
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15))
	img = cv2.dilate(img,kernel)
	img = cv2.erode(img, kernel)

	#find the circles
	circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,300,
								param1=50,param2=30,minRadius=20,maxRadius=200)

	circles = circles[0, :, :]
	circles = np.uint16(np.around(circles))
	
	for i in circles[:]:
		if rad != -1:
			x = np.asarray((i[0], i[1]))
			y = np.asarray(cen)
			distance=np.sqrt(np.sum(np.square(x-y)))
			#if the circle center is not in the previous circle, then skip this candidate
			if distance > rad:
				continue
		center = (i[0], i[1])
		radius = i[2]
		cv2.circle(pltimg, (i[0], i[1]), i[2], (0, 255, 0), 3)

	print(center, radius)
	return pltimg, center, radius

def find_score_region(pic_path):
	img = cv2.imread(pic_path)

	img_blue = get_specific_region(img, [60, 118, 130], [120, 255,255])
	img_red = get_specific_region(img, [160, 65, 104], [255,255,255])	
	img_yellow = get_specific_region(img, [3,110,100], [32,255,255])
	
	img, cen_b, rad_b = find_circles(img_blue, img, (-1,-1), -1)
	img, cen_r, rad_r = find_circles(img_red, img, cen_b, rad_b)
	img, cen_y, rad_y = find_circles(img_yellow, img, cen_r, rad_r)
	#According to the red and blue to draw the black and white region
	cv2.circle(img, cen_b, rad_b*2-rad_r, (0, 255, 0), 3)
	cv2.circle(img, cen_b, rad_b*3-rad_r*2, (0, 255, 0), 3)
	 
	displayIMG(img, "res")
	cv2.destroyAllWindows()

if __name__ == "__main__":
	pic_path = "target.jpg"
	find_score_region(pic_path)