import easyocr
from io import BytesIO
import cv2
import numpy as np
from utility import TextObject

def is_noisy(image, threshold=30):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    mean, stddev = cv2.meanStdDev(gray)
    return stddev[0][0] > threshold

def choose_contrast_method(img_gray):
    hist = cv2.calcHist([img_gray], [0], None, [256], [0, 256])
    hist = hist.ravel()
    contrast_range = np.percentile(hist, 95) - np.percentile(hist, 5)
    return 'clahe' if contrast_range < 40 else 'hist'

def recommend_preprocessing(features):
    recommendations = []

    # Adaptive contrast thresholds based on stroke thickness
    stroke = features['stroke']
    contrast = features['contrast']
    noise = features['noise']
    edges = features['edges']
    components = features['components']

    if contrast < 15:
        recommendations.append('hist_eq')
    elif contrast < 40:
        recommendations.append('clahe')
    elif contrast > 70:
        recommendations.append('none')

    # If stroke is thin, dilate to enhance text thickness
    if stroke < 1.7 and contrast < 40 and edges < 0.10 and stroke >= 2.5:
        recommendations.append('dilate')
    elif stroke > 8:
        recommendations.append('erode')

    # Noise thresholds, with margin for adaptive h in denoising
    if noise > 20 and stroke > 2:
        recommendations.append('denoise')

    # Sharpen if low edge density and moderate contrast
    if (features['edges'] < 0.03 and contrast < 50 ) or stroke < 2.5:
        recommendations.append('sharpen')

    # Morph close to merge small gaps if many components
    if features['components'] > 150:
        recommendations.append('morph_close')

    return list(set(recommendations))



def analyze_roi_properties(roi):
    gray = roi if len(roi.shape) == 2 else cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # 1. Contrast: RMS (Root Mean Square) contrast
    # Measures the standard deviation of pixel intensities around the mean — good indicator of perceptual contrast.
    mean_intensity = np.mean(gray)
    rms_contrast = np.sqrt(np.mean((gray - mean_intensity) ** 2))

    # 2. Stroke thickness: via distance transform on binarized image
    # Distance transform gives radius from edges — x2 for full thickness estimate
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Invert if background is dark (i.e. more white pixels = background)
    if np.sum(binary == 255) > np.sum(binary == 0):
        binary = 255 - binary

    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 3)
    stroke_thickness = np.median(dist[dist > 0]) * 2 if np.any(dist > 0) else 0

    # 3. Noise: median of local patch standard deviations (robust to edges)
    # This detects local texture variance — good proxy for grain/noise
    patch_size = 16
    patches = [
        gray[i:i+patch_size, j:j+patch_size]
        for i in range(0, gray.shape[0] - patch_size + 1, patch_size)
        for j in range(0, gray.shape[1] - patch_size + 1, patch_size)
    ]
    local_stds = [np.std(p) for p in patches if p.size == patch_size * patch_size]
    noise_level = np.median(local_stds) if local_stds else 0

    # 4. Edge density: Canny edge pixels divided by total area
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / (gray.shape[0] * gray.shape[1])

    # 5. Connected components: count of distinct blobs in binarized image
    num_labels, _, _, _ = cv2.connectedComponentsWithStats(binary)

    return {
        'contrast': rms_contrast,
        'stroke': stroke_thickness,
        'noise': noise_level,
        'edges': edge_density,
        'components': num_labels
    }



def preProcess_roi(roi):
    props = analyze_roi_properties(roi)
    actions = recommend_preprocessing(props)
    gray = roi if len(roi.shape) == 2 else cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    print(f'props: {props}, actions: {actions}')

    # Denoise — tuned strength to avoid smearing small text
    if 'denoise' in actions:
        h = min(max(int(props['noise'] * 0.4), 3), 7)
        gray = cv2.fastNlMeansDenoising(gray, h=h)

    # CLAHE — mild enhancement for moderate contrast + small tiles for fine details
    if 'clahe' in actions:
        clipLimit = 1.5 if props['stroke'] < 2.5 else 1.2
        tileGridSize = (4, 4) if gray.shape[0] < 64 else (6, 6)
        clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
        gray = clahe.apply(gray)

    # Histogram Equalization — only for very flat contrast
    if 'hist_eq' in actions:
        gray = cv2.equalizeHist(gray)

    # Sharpen — soft kernel to reduce over-boosting artifacts
    if 'sharpen' in actions:
        kernel = np.array([[0, -1, 0], [-1, 6, -1], [0, -1, 0]])
        gray = cv2.filter2D(gray, -1, kernel)

    # Dilate — scale kernel size to weak strokes but keep minimal expansion
    if 'dilate' in actions:
        kernel_size = max(1, int(props['stroke'] / 2.5))
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        gray = cv2.dilate(gray, kernel, iterations=1)

    # Erode — same logic for thick strokes
    if 'erode' in actions:
        kernel_size = max(1, int(props['stroke'] / 3))
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        gray = cv2.erode(gray, kernel, iterations=1)

    # Morph close — closes small holes between connected strokes
    if 'morph_close' in actions:
        kernel_size = 1 if gray.shape[0] < 64 else 2
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)

    return gray





def imgPreProcess(img):
    if is_noisy(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        denoised = cv2.fastNlMeansDenoising(gray, h=10)
    else:
        denoised = img

    sharpened = cv2.filter2D(denoised, -1, np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]))

    gray = sharpened if len(sharpened.shape) == 2 else cv2.cvtColor(sharpened, cv2.COLOR_BGR2GRAY)
    method = choose_contrast_method(gray)

    if method == 'clahe':
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
    else:
        enhanced = cv2.equalizeHist(gray)

    return enhanced




def bboxSizes(bboxes):
    textObjs = []
    for bbox in bboxes:
        bboxInt = [int(val) for val in bbox]
        xMin ,xMax, yMin, yMax = bboxInt
        width = xMax - xMin
        height = yMax - yMin
        textObjs.append(TextObject(bbox = bboxInt, width = width, height = height))

    return textObjs

def bboxDraw(img, bboxes):
    
    for bbox in bboxes:
        xMin ,xMax, yMin, yMax = bbox
        cv2.rectangle(img, (xMin,yMin), (xMax,yMax), (0,0,255), 2)
    return img


def detect_bboxes_easyOCR(img, reader):

    # Convert image to grayscale for better OCR results
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)    
    # Parameters
    scale = 3
    min_size = 1
    text_threshold = 0.05
    low_text = 0.2
    link_threshold = 0.7

    # Preprocess
    enhanced = imgPreProcess(gray_image)

    # Detect on enhanced image
    scaled_enhanced = cv2.resize(enhanced, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    horizontal_list_enhanced, _ = reader.detect(
        scaled_enhanced,
        min_size=min_size,
        text_threshold=text_threshold,
        low_text=low_text,
        link_threshold=link_threshold
    )
    bboxes_enhanced = [(int(x_min / scale), int(x_max / scale), int(y_min / scale), int(y_max / scale))
                       for x_min, x_max, y_min, y_max in horizontal_list_enhanced[0]]

    # Detect on raw gray image
    scaled_gray = cv2.resize(gray_image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    horizontal_list_gray, _ = reader.detect(
        scaled_gray,
        min_size=min_size,
        text_threshold=text_threshold,
        low_text=low_text,
        link_threshold=link_threshold
    )
    bboxes_gray = [(int(x_min / scale), int(x_max / scale), int(y_min / scale), int(y_max / scale))
                   for x_min, x_max, y_min, y_max in horizontal_list_gray[0]]

    # Combine both bbox lists
    combined_bboxes = bboxes_enhanced + bboxes_gray

    # Filter combined bboxes by size
    filtered_bboxes = [b for b in combined_bboxes if (b[1] - b[0]) > 10 and 8 < (b[3] - b[2]) < 100]

    # Merge overlapping boxes
    merged_bboxes = []
    while filtered_bboxes:
        base = filtered_bboxes.pop(0)
        to_merge = [base]
        i = 0
        while i < len(filtered_bboxes):
            xA = max(base[0], filtered_bboxes[i][0])
            yA = max(base[2], filtered_bboxes[i][2])
            xB = min(base[1], filtered_bboxes[i][1])
            yB = min(base[3], filtered_bboxes[i][3])
            interArea = max(0, xB - xA) * max(0, yB - yA)
            boxAArea = (base[1] - base[0]) * (base[3] - base[2])
            boxBArea = (filtered_bboxes[i][1] - filtered_bboxes[i][0]) * (filtered_bboxes[i][3] - filtered_bboxes[i][2])
            iou = interArea / float(boxAArea + boxBArea - interArea) if (boxAArea + boxBArea - interArea) > 0 else 0

            if iou > 0.3:
                to_merge.append(filtered_bboxes.pop(i))
                # update base to the union of all boxes merged so far
                base = (
                    min(b[0] for b in to_merge),
                    max(b[1] for b in to_merge),
                    min(b[2] for b in to_merge),
                    max(b[3] for b in to_merge)
                )
                i = 0
            else:
                i += 1
        merged_bboxes.append(base)

    #boxedImg = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    boxedImg = bboxDraw(img.copy(), merged_bboxes)
    
    #cv2.imshow("BBoxes easyOCR", boxedImg)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    return merged_bboxes


def bboxRead(img, textObjs, reader):

    # Convert image to grayscale for better OCR results
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  

    best_i = None
    bestAvgProb = 0
    bestAvgHeight = 0
    results = []
   
    for i in np.arange(20, 30, 1):
        
        updated_count = 0
        totalProb = 0
        totalHeight = 0
        for obj in textObjs:
            xMin ,xMax, yMin, yMax = obj.bbox
            margin = 2
            yMinM = max(yMin - margin, 0)
            yMaxM = min(yMax + margin, gray_image.shape[0])
            xMinM = max(xMin - margin, 0)
            xMaxM = min(xMax + margin, gray_image.shape[1])
            croppedImage = gray_image[yMinM:yMaxM, xMinM:xMaxM]
            scaleFactor = 1
            if obj.height < 25:
                scaleFactor = i/obj.height
                croppedImage = cv2.resize(croppedImage, None, fx = scaleFactor, fy = scaleFactor, interpolation= cv2.INTER_CUBIC)

            optimizedImage = preProcess_roi(croppedImage)
            res = reader.readtext(optimizedImage, min_size = 1)
            for bbox, text, prob in res:
                #print(f'scaleFactor: {scaleFactor}, text: {text}, prob: {prob}, obj.confidence: {obj.confidence}')
                if prob > obj.confidence:
                    obj.text = text
                    obj.confidence = prob
                    obj.bestScale = scaleFactor
                    updated_count += 1
                    totalProb += prob
                    currentHeight = (bbox[2][1] -bbox[1][1])
                    totalHeight += currentHeight
                    #cv2.imshow("optimizedImage", optimizedImage)
                    print(obj.text)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()

            #print(f'i: {i} , scaleFactor: {scaleFactor}, count: {count}, currentHeight: {currentHeight}, currentProb: {prob} Average probability: {totalProb/count}, averageHeight: {totalHeight/count}')
        

        avg_prob = totalProb / updated_count if updated_count > 0 else 0
        avg_height = totalHeight / updated_count if updated_count > 0 else 0

        print(f'i: {i} , count: {updated_count}, Average probability: {avg_prob}, Average height: {avg_height}')
        results.append((i, avg_prob, avg_height))

        # Update best `i` if current probability is better
        if avg_prob > bestAvgProb:
            bestAvgProb = avg_prob
            best_i = i
            bestAvgHeight = avg_height


            #print(f"Enlarged object: \nText: {obj.text}, confidence: {obj.confidence:.2f}, bbox: {obj.bbox} \n")
    print(f'\nBest i value: {best_i} with average probability {bestAvgProb}, average height: {bestAvgHeight}')

    return textObjs

def process_image(image_data: BytesIO):
    # Convert image bytes to numpy array
    image_bytes = image_data.getvalue()
    nparr = np.frombuffer(image_bytes, np.uint8)
    
    # Decode the image
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode image")
    
    # Initialize EasyOCR reader
    reader = easyocr.Reader(['en']) 

                                            
    
    bboxes = detect_bboxes_easyOCR(img, reader)
    

    #detect_bboxes_opencv(img)



    # Getting bbox sizes and creating TextObject array with the bbox values + width & height
    textObjs = bboxSizes(bboxes) 

    textObjs = bboxRead(img,textObjs,reader)
   
    return textObjs
