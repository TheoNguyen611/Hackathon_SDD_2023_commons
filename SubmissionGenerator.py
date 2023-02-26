import numpy as np
EVALUATION_DATASET_LENGTH=1559
DV_MAX=1.5
MANEUVER_TIME_MAX=48*3600

def create_submission(predictions, output_file, normalize=True):
    """Create the subimssion in the correct format
            Parameters
            ----------
            predictions : np.ndarray
                The predictions. It should be a numpy array of shape (1559,3). The rows are the samples of the evaluation
                dataset.
                The first column is the detection either 0 or 1, the second column is the estimation of the dv between -1.5 and 1.5, the third column is
                 the estimation of the date of the maneuver in second from the start of the observation window between
                 0 and 48*3600.
            output_file : str
                path of the outputfile.
            normalize:bool, optional (default is True)
                If the estimations are already normalized between 0 and 1, you should set this parameter to False.
            """
    check_argument(predictions)
    if ".csv" not in output_file:
        output_file+=".csv"
    with open(output_file,"w+") as prediction_file:
        prediction_file.write(f"detection;dv;date\n")
        for prediction in predictions:
            classification,dv,time=prediction
            check_line(classification,dv,time)
            if normalize:
                dv=(DV_MAX+dv)/(2*DV_MAX)
                time/=MANEUVER_TIME_MAX
            prediction_file.write(f"{int(classification)};{dv};{time}\n")
def check_line(classification, dv, time):
    assert isinstance(classification,float), f"got {type(classification)} while expected float"
    assert isinstance(dv,float),f"got {type(dv)} while expected float"
    assert isinstance(time,float),f"got {type(time)} while expected float"

def check_argument(prediction):
    assert isinstance(prediction, np.ndarray),"prediction should be a numpy array"
    assert prediction.shape==(EVALUATION_DATASET_LENGTH,3),f"prediction shape is {prediction.shape} " \
                                                           f"while ({EVALUATION_DATASET_LENGTH},3) is expected"
    assert prediction.shape == (EVALUATION_DATASET_LENGTH, 3), f"prediction shape is {prediction.shape} " \
                                                               f"while ({EVALUATION_DATASET_LENGTH},3) is expected"
