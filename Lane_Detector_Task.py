import cv2
import numpy as np
import matplotlib.pyplot as plt


# "Improving the result"

CANNY_LOW, CANNY_HIGH = 50, 150          # A. try (30,100), (70,150), (100,200)
ROI_TOP_FRAC          = 0.55             # B. polygon apex (0.50, 0.55, 0.60, 0.65)
ROI_SIDE_INNER        = 0.05             # B. inner side of trapezoid base
ROI_SIDE_OUTER        = 0.95             # B. outer side of trapezoid base

HOUGH_RHO             = 2                # C. try 1, 2, 3
HOUGH_THETA           = np.pi / 180      # C. try pi/180, pi/90, pi/360
HOUGH_THRESHOLD       = 30               # C. try 20, 30, 50, 100
HOUGH_MIN_LINE_LEN    = 20               # C. try 20, 40, 60
HOUGH_MAX_LINE_GAP    = 20               # C. try 5, 10, 20


# Load and display

img = cv2.imread('road.jpg')
if img is None:
    raise FileNotFoundError(
        "road.jpg not found in the working directory. "
        "Download a driver-perspective road image and save it as road.jpg."
    )

cv2.imshow('Image', img)
cv2.waitKey(0)

height, width = img.shape[:2]
print(f"Image size: width={width}, height={height}")


# Convert to grayscale 

copy_img = np.copy(img)
gray_img = cv2.cvtColor(copy_img, cv2.COLOR_RGB2GRAY)

plt.figure(figsize=(10, 6))
plt.imshow(gray_img, cmap='gray')
plt.title('Grayscale image')
plt.axis('off')
plt.show()


# Gaussian blur 

blur_img = cv2.GaussianBlur(gray_img, (5, 5), 20)


# Canny edge detection 

canny_img = cv2.Canny(blur_img, CANNY_LOW, CANNY_HIGH)

plt.imshow(canny_img)
plt.title('Canny edges')
plt.show()


# Region of Interest 

def region_of_interest(img):
    """Slide 19, but with the trapezoid generated from tunables above."""
    h = img.shape[0]
    w = img.shape[1]
    points = np.array([
        [int(w * ROI_SIDE_INNER),  h],
        [int(w * 0.40),            int(h * ROI_TOP_FRAC)],
        [int(w * 0.60),            int(h * ROI_TOP_FRAC)],
        [int(w * ROI_SIDE_OUTER),  h],
    ], dtype=np.int32)
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, pts=[points], color=255)
    return mask


mask = region_of_interest(canny_img)

masked_image = cv2.bitwise_and(canny_img, mask)

plt.imshow(masked_image, cmap='gray')
plt.title('ROI-masked edges')
plt.show()


# Hough Transform  

lines = cv2.HoughLinesP(
    masked_image,
    rho=HOUGH_RHO,
    theta=HOUGH_THETA,
    threshold=HOUGH_THRESHOLD,
    minLineLength=HOUGH_MIN_LINE_LEN,
    maxLineGap=HOUGH_MAX_LINE_GAP,
)

print(f"Detected {0 if lines is None else len(lines)} line segments")
print(lines)


# Display lines and combine with the original image

def display_lines(img, lines):
    line_img = np.zeros_like(img)
    if lines is not None:

        for [x1, y1, x2, y2] in lines:
            cv2.line(line_img, (x1, y1), (x2, y2), (255, 0, 0), 5)
    return line_img


line_image = display_lines(img, lines)

plt.imshow(line_image)
plt.title('Detected line segments')
plt.show()

combo_image = cv2.addWeighted(img, 0.8, line_image, 1, 1)

plt.imshow(cv2.cvtColor(combo_image, cv2.COLOR_BGR2RGB))
plt.title('Original + Hough lines')
plt.show()

cv2.imshow('Lane Detection', combo_image)
cv2.waitKey(0)
cv2.destroyAllWindows()