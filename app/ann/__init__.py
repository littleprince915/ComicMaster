import Pre_Processing as pp
import ANN_Model as annm
import ANN_Functions as annf
import pickle

picklefile = open("model.pickle", "rb")
parameters = pickle.load(picklefile)
picklefile.close()


def predict_ann(image):
    data = pp.preprocess_data(image, 100)

    return annf.predict_one(data, parameters)
