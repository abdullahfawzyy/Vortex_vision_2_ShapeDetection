# ==========================================
# CALIBRATION NOTES:
# After debugging with the sample image, I had to tune the parameters below.
# 1. area limits: the original code was picking up small noise dots and the 
#   large paper border. I added a min area (500) and max area (10000) to 
#   isolate only the target shapes.
#2. Tolerance: The default approximation was too loose (0.04), causing 
#   circles to be detected as triangles. I lowered the tolerance to 2% (0.02)
#   to ensure circles retain enough vertices to be correctly classified.
# ==========================================
import cv2 as cv
import numpy as np
import os
# a dic to store the count of the shapes 
counts = {
    "triangle": 0, "square": 0, "rectangle": 0, "circle": 0
}

def count_shapes(imagepath):
    img = cv.imread(imagepath)
    #after loading the image make sure that it is present 
    if img is None:
        print("can not load image")
        return None  # Stop function if no image
    #convert to gray scale to highlight the geometrical shapes as contrast matter not colors and to be able to threshold later
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    #blurring the image do reduce the noise
    blurred = cv.GaussianBlur(gray, (7, 7), 0)
    #threshold and ingnore the first return value "the threshold value" and inv makes the background black and the shape white not like normal binary image 
    _, binaryimg = cv.threshold(blurred, 127, 255, cv.THRESH_BINARY_INV)
    #find conntour edges and ignore the second return value "hierarchy list that we dont need"
    contours, _ = cv.findContours(binaryimg, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv.contourArea(contour)
        #filter out too small contours that are mostly noise and too large as they are the photo edges 
        if area < 500 or area > 10000:
            continue
        #find the perimeter using arc length             
        perimeter = cv.arcLength(contour, True)
        #approxPloyDP reduces the number of vertices to simplify the shape within the tolerance value of 2% of perimeter
        approximate = cv.approxPolyDP(contour, 0.02 * perimeter, True)
        #this counts the number of straight lines that is the same as the number of vertices 
        vertices = len(approximate)
        # bounding rect creates the smallest rectangle that encloses the shape to get the width and height
        x, y, w, h = cv.boundingRect(approximate) 
        #caluclates the radio between the sides to diffrentiate between the square and the rectangle 
        sidesratio = float(w) / float(h)
        #classification based on the vertices count 
        if vertices == 3:
            counts["triangle"] += 1
            cv.drawContours(img, [approximate], 0, (255, 0, 0), 2)
        elif vertices == 4:
            if 0.95 <= sidesratio <= 1.05:
                counts["square"] += 1
                cv.drawContours(img, [approximate], 0, (0, 255, 0), 2)
            else:
                counts["rectangle"] += 1
                cv.drawContours(img, [approximate], 0, (0, 0, 255), 2)
        #anything else is a circles as it has more than 5 sides this changes according to the tolerance value of the approxpolyDP
        else:
            counts["circle"] += 1
            cv.drawContours(img, [approximate], 0, (0, 0, 0), 2)

    return img

result_img = count_shapes("assets/input_images/Shapes.jpg") 
#save a copy of the results in the output images folder 
output_path = os.path.join("assets", "output_images", "detected_shapes.jpg")
cv.imwrite(output_path, result_img)
#display the results 
if result_img is not None:
    print("The Shape counts are:")
    for shape, count in counts.items():
        print(f"{shape}: {count}")
    cv.imshow("detected shapes", result_img)
    cv.waitKey(0)
    cv.destroyAllWindows()