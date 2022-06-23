import cv2 as cv
import numpy as np
from PIL import Image
import base64
import io
# import draw_contours
from scipy import ndimage
from skimage import measure
import seaborn as sns
import pandas as pd

# local import
from .config import PROTXT_FILE, CAFFEMODEL_FILE, OUTPUT_FRAG_IMG, OUTPUT_FRAG_PIE_CHART, OUTPUT_FRAG_GRAPH, OUTPUT_FRAG_RESULTS_CV_CSV


# Take in base64 string and return cv image
def base64_to_image(base64_string):
    """Convert base64 string to image

    Args:
        base64_string (string): base64 format of an image

    Returns:
        numpy.ndarray: converted image
    """
    imgdata = base64.b64decode(str(base64_string))
    image = Image.open(io.BytesIO(imgdata))
    return image

def image_to_base64(image_path):
    """Take an image path and convert that image to base64 string

    Args:
        image_path (numpy.ndarray): Path of the image that needs to be converted into base64

    Returns:
        base64_string: base64 format of an image
    """
    with open(image_path, "rb") as image_file:
        data = base64.b64encode(image_file.read())
        return data.decode("utf-8")

def image_to_base64_without_path(image):
    """Take a numpy array (CV Image) and convert it into base64 string

    Args:
        image (str): Image that needs to be converted into base64

    Returns:
        base64_string: base64 format of an image
    """
    data = base64.b64encode(image)
    return data.decode("utf-8")

class CropLayer(object):
    """Holistically Nested Edge Detection Class"""

    def __init__(self, params, blobs):
        self.xstart = 0
        self.xend = 0
        self.ystart = 0
        self.yend = 0

    def getMemoryShapes(self, inputs):
        """Our layer receives two inputs. We need to crop the first input blob
        to match a shape of the second one (keeping batch size and number of channels)
        """
        inputShape, targetShape = inputs[0], inputs[1]
        batchSize, numChannels = inputShape[0], inputShape[1]
        height, width = targetShape[2], targetShape[3]

        self.ystart = int((inputShape[2] - targetShape[2]) / 2)
        self.xstart = int((inputShape[3] - targetShape[3]) / 2)
        self.yend = self.ystart + height
        self.xend = self.xstart + width

        return [[batchSize, numChannels, height, width]]

    def forward(self, inputs):
        """Simple feedforward function"""
        return [inputs[0][:,:,self.ystart:self.yend,self.xstart:self.xend]]


def processing(img_name, low_intensity=100, high_intensity=200, external_contours=None):
    """Main Function to perform fragmentation. It performs so by first finding the edges of rocks in a picture.
    A Holistically Nested Edge Detection Algoriithm (HED) is used for finding edges with better accuracy.
    After that image is preprocessed and thresholded to find contours in the image where contours are rock boundaries.
    After that area of contours / rocks are found. The area can be further visualized if needed with a matplotlib plot


    Args:
        img_name (str): image path
        low_intensity (int, optional): Low intensity of thershold. Defaults to 100.
        high_intensity (int, optional): High intensity of thershold. Defaults to 200.
        external_contours (_type_, optional): A flag that if enabled lets you draw external contours. Defaults to None.

    Returns:
        output_list (list): output image name, rock area data and black and white version of image
    """

    # Load the model and protxt file.
    net = cv.dnn.readNetFromCaffe(PROTXT_FILE, CAFFEMODEL_FILE)
    cv.dnn_registerLayer('Crop', CropLayer)
    
    # Read Image
    image=cv.imread(img_name)

    # Pre-process image
    inp = cv.dnn.blobFromImage(image, scalefactor=1.0, size=(image.shape[0],image.shape[1]),
                            mean=(104.00698793, 116.66876762, 122.67891434),
                            swapRB=False, crop=False)
    
    # Start Forward pass and perform HED
    net.setInput(inp)
    out = net.forward()

    # Post-process image
    out = out[0, 0]
    out = cv.resize(out, (image.shape[1], image.shape[0]))
    print("Size of Image is: " + str(out.shape))
    out=cv.cvtColor(out,cv.COLOR_GRAY2BGR)
    out = 255 * out
    out = out.astype(np.uint8)

    # Converting image to binary using low and high threshold
    out[np.where(out>high_intensity)] = 255
    out[np.where(out<low_intensity)] = 0

    ########################## Find and Draw Contours ###################################

    # Error can happen due to different cv2 version
    try:
        _, cnts,_ = cv.findContours(cv.cvtColor(out, cv.COLOR_BGR2GRAY), mode=cv.RETR_TREE, method=cv.CHAIN_APPROX_SIMPLE)
    except:
        cnts,_ = cv.findContours(cv.cvtColor(out, cv.COLOR_BGR2GRAY), mode=cv.RETR_TREE, method=cv.CHAIN_APPROX_SIMPLE)

    # Draw external contours if it exists
    if external_contours != None:
        temp = []
        temp = cnts
        cnts = []
        cnts.extend(temp)
        cnts.extend(external_contours)
    
    # draw contours
    cv.drawContours(image, cnts, -1, (255, 0, 0), 2)

    ########################## Draw External Contours (Test Function Used for debugging only) ###################################
    # out = draw_contours.draw_pixels(image)

    ########################## Find Area of Contour Blobs ###################################

    # If we want the scipy function to extract area in terms of pixels
    # scipy_contour_area(cnts, out)

    # If we want to use opencv function to extract area in terms of pixels
    ret_rock_data = cv_contour_area(cnts, out, img_name)
    
    ########################## Draw Graph and Pie Chart and save figures ###################################

    # Change the the OUTPUT_FRAG_RESULTS_CSV to OUTPUT_FRAG_RESULTS_CV_CSV if we want to use csv from opencv 
    # and uncomment the above function accordingly
    # draw_graph_pie(img_name)
    
    # Concatenate original image with processed image and save the image
    # con=np.concatenate((image,out),axis=1)
    con = image.copy()
    cv.imwrite(img_name[:9] + OUTPUT_FRAG_IMG,con)

    # saving black and white image
    bandw_img_path = img_name[:9] + "bandw.png" 
    cv.imwrite(bandw_img_path, out)
    # Unregister cv layer otherwise it will give error next time
    cv.dnn_unregisterLayer('Crop')

    return [img_name[:9] + OUTPUT_FRAG_IMG, ret_rock_data, bandw_img_path]

def cv_contour_area(cnts, image, filename):
    """Finding area of rock contours using opencv

    Args:
        cnts (numpy.ndarray): contours
        image (numpy.ndarray): image / frame whose contours are provided
        filename (str): CSV file of the output of rock area data

    Returns:
        rock_data (list): list of bucketed rock size 
    """
    out = image.copy()
    total = 0
    count = 0

    # open file for saving data
    f = open(filename[:9] + OUTPUT_FRAG_RESULTS_CV_CSV, "w")
    f.write(',' + ",".join(["Area"]) + '\n')

    ret = {}
    
    for c in cnts:
        x,y,w,h = cv.boundingRect(c)

        # calculating the mask of image
        mask = np.zeros(out.shape, dtype=np.uint8)
        cv.fillPoly(mask, [c], [255,255,255])
        mask = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)

        # counting all non zero blobs
        pixels = cv.countNonZero(mask)
        total += pixels
        # cv.putText(out, '{}'.format(pixels), (x,y), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        # writing the data to csv
        f.write(str(count) + ", " + str(pixels) + "\n")

        # sanity check to ignore rocks whose size are less than 100 pixels
        if pixels > 100:
            ret[count] = pixels
        count += 1

    f.close()
    # print(ret, file=sys.stderr)
    return bucket_data_v2(ret)

def bucket_data(data):
    """Bucket the rock area data into linear size areas

    Args:
        data (dict): dictionary containing count of rock and its area in terms of pixels

    Returns:
        rock_data (list): list of bucketed rock size 
    """
    ret = []
    max_val = max(data.values())
    temp = {max_val/10: 0, (max_val * 2)/10: 0, (max_val * 3)/10: 0, (max_val * 4)/10: 0, (max_val * 5)/10: 0, (max_val * 6)/10: 0, (max_val * 7)/10: 0, (max_val * 8)/10: 0, (max_val * 9)/10: 0, max_val: 0}
    for key, value in data.items():
        absolute_difference_function = lambda list_value : abs(list_value - value)
        closest_value = min(temp.keys(), key=absolute_difference_function)
        temp[closest_value] += 1
    for key, value in temp.items():
        ret.append([key, value])
    return ret

def bucket_data_v2(data):
    """Bucket the rock area data into linear size areas in a better way using pandas

    Args:
        data (dict): dictionary containing count of rock and its area in terms of pixels

    Returns:
        rock_data (list): list of bucketed rock size 
    """
    ret = pd.DataFrame({'test':pd.Series(data)})
    test = ret.value_counts().reset_index()
    _, edges = pd.cut(test['test'], bins=1000, retbins=True)
    edges = [int(x) for x in edges]
    test['range'] = pd.cut(test['test'],bins=edges).astype(str)
    test2 = test.groupby('range')['test'].count().reset_index()

    test2['range_stripped'] = test2['range'].map(lambda x:x.split(',')[1].rstrip(']').lstrip(' '))
    
    test2.drop(columns=['range'],inplace=True)
    ret_list = []
    for _,v in test2.astype(int).sort_values('range_stripped',ascending=True).iterrows():
        ret_list.append([str(v[1]),int(v[0])])

    return ret_list

def scipy_contour_area(cnts, image):
    """Finding area of rock contours using scipy

    Args:
        cnts (numpy.ndarray): contours
        image (numpy.ndarray): image / frame whose contours are provided
        filename (str): CSV file of the output of rock area data

    Returns:
        rock_data (list): list of bucketed rock size 
    """
    out = image.copy()
    kernel = np.ones((3, 3), np.uint8)
    eroded = cv.erode(out, kernel, iterations=1)
    dilated = cv.dilate(eroded, kernel, iterations=1)

    mask = dilated == 255

    labeled_mask, num_labels = ndimage.label(mask)
    print("Number of rocks detected are: " + str(num_labels))

    # area of clusters
    clusters = measure.regionprops(labeled_mask)
    propList = ['Area']

    output_file = open('output/fragmentation_result.csv', 'w')
    output_file.write(',' + ",".join(propList) + '\n')

    for cluster_props in clusters:
        # output cluster properties to the excel file
        output_file.write(str(cluster_props['Label']))
        for i, prop in enumerate(propList):
            if (prop == 'Area'):
                to_print = cluster_props[prop]  # Convert pixel square to um square
            else:
                to_print = cluster_props[prop]
            output_file.write(',' + str(to_print))
        output_file.write('\n')
    output_file.close()

def draw_graph_pie(img_name):
    """Function to visualize the bucketed data

    Args:
        img_name (str): unique string received from main function
    """
    df = pd.read_csv(img_name[:9] + OUTPUT_FRAG_RESULTS_CV_CSV)
    
    # Make a copy
    df_copy=df

    # Make an empty list to store counts of rocks 
    l=[]

    # Append the numbers as per logic given below
    for i in range(df.Area.shape[0]):
        if df.Area[i] < 1000:
            l.append(0)
        elif df.Area[i] < 3000:
            l.append(1)
        elif df.Area[i] < 6000:
            l.append(2)
        elif df.Area[i] < 9000:
            l.append(3)
        else:
            l.append(4)

    # Drop if area is too small or too large
    for i in range(df.shape[0]):
        # print(df.shape[0])
        if df.Area[i] < 100 or df.Area[i] > 50000:
            df = df.drop([i])

    # process and save graph
    df = df.reset_index(drop=True)
    graph_pic = sns.displot(df, x="Area", kde=True)
    graph_pic.savefig(img_name[:9] + OUTPUT_FRAG_GRAPH)
    
    # process and save pie chart
    df_copy['new_column'] = l
    sns.displot(df_copy, x="Area", kde=True, col="new_column")
    data = df_copy.groupby("new_column").count()
    plot = data.plot.pie(y='Area', figsize=(10, 8))
    fig = plot.get_figure()
    fig.savefig(img_name[:9] + OUTPUT_FRAG_PIE_CHART)