# import the necessary packages
from skimage.metrics import structural_similarity as compare_ssim
import imutils
import cv2
import sentry_sdk
sentry_sdk.init(
    dsn="https://da5575591b4e4828bd9d1304bd3512c2@o1411533.ingest.sentry.io/6750081",

    # Set traces_sample_rate to 1.0 to capture 100%
    # of transactions for performance monitoring.
    # We recommend adjusting this value in production.
    traces_sample_rate=1.0

)

# load our input image
image=cv2.imread('bird.jpg')
cv2.imshow('original', image)

image1=cv2.imread('bird1.jpg')
cv2.imshow('edit', image1)

#we use cvtcolor, to convert to greyscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow('grayscale', gray_image)

gray_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
cv2.imshow('grayscale1', gray_image1)


# compute the Structural Similarity Index (SSIM) between the two
# images, ensuring that the difference image is returned
(score, diff) = compare_ssim(gray_image, gray_image1, full=True)
diff = (diff * 255).astype("uint8")
print("SSIM: {}".format(score))

# threshold the difference image, followed by finding contours to
# obtain the regions of the two input images that differ
thresh = cv2.threshold(diff, 0, 255,
	cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

# loop over the contours
for c in cnts:
	# compute the bounding box of the contour and then draw the
	# bounding box on both input images to represent where the two
	# images differ
	(x, y, w, h) = cv2.boundingRect(c)
	cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
	cv2.rectangle(image1, (x, y), (x + w, y + h), (0, 0, 255), 2)
# show the output images
cv2.imshow("Origina", image)
cv2.imshow("Modified", image1)
cv2.imshow("Diff", diff)
cv2.imshow("Thresh", thresh)
cv2.waitKey(0)
