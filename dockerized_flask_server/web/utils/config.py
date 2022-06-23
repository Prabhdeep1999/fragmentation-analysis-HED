from configparser import ConfigParser

# important constants
PROTXT_FILE = "./req_files/deploy.prototxt"
CAFFEMODEL_FILE = "./req_files/hed_pretrained_bsds.caffemodel"
OUTPUT_FRAG_IMG = "demo_out.jpg"
OUTPUT_FRAG_PIE_CHART = "pie_fig.jpg"
OUTPUT_FRAG_GRAPH = "out_gf.jpg"
OUTPUT_FRAG_RESULTS_CSV = "fragmentation_result.csv" 
OUTPUT_FRAG_RESULTS_CV_CSV = "fragmentation_result_cv.csv"
OUTPUT_DRAW_CONTOURS = "contour_draw.jpg"

def config(filename='./req_files/database.ini', section='postgresql'):
    """Config Parser for the database.ini config file 

    Args:
        filename (str, optional): Path to .ini file (config file). Defaults to 'req_files/database.ini'.
        section (str, optional): section in .ini file to extract. Defaults to 'postgresql'.

    Raises:
        Exception: If the section is not in .ini file an exception is raised

    Returns:
        db (dict): Dictionary of all the relevant information needed to connect to Postgresql database
    """
    # create a parser
    parser = ConfigParser()
    # read config file
    parser.read(filename)

    # get section, default to postgresql
    db = {}
    if parser.has_section(section):
        params = parser.items(section)
        for param in params:
            db[param[0]] = param[1]
    else:
        raise Exception('Section {0} not found in the {1} file'.format(section, filename))

    return db