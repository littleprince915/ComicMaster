
import Pre_Processing as pp
import ANN_Model as annm
import ANN_Functions as annf
import pickle

global parameters


picklefile = open("model.pickle", "rb")
parameters = pickle.load(picklefile)
picklefile.close()

def create_ann_dataset():
    return
    global parameters
    side_length = 100
    np_training_set_x, np_training_set_y = pp.load_dataset('training', 'jpg', side_length)

    np_training_set_x = annf.standardize(np_training_set_x)

    parameters = annm.L_layer_model(np_training_set_x, np_training_set_y,
                                    num_iterations=10, learning_rate=0.1, print_cost=True,
                                    layers_dims=[side_length * side_length * 3, 100, 80, 60, 40, 20, 10, 1])

    picklefile = open("model.pickle", "wb")
    pickle.dump(parameters, picklefile)
    picklefile.close()

def predict_ann(image):
    data, resized_image = pp.preprocess_data(image, 100)

    

    return annf.predict_one(data, parameters), resized_image