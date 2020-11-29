import importlib
import os
from tqdm import tqdm
import time


def generate_outputs(pyfile_path, input_paths, output_dir):
    predict_py = pyfile_path+".model_predict"
    define_py = pyfile_path+".model_define"
    init_model = getattr(importlib.import_module(define_py), "init_model")
    predict = getattr(importlib.import_module(predict_py), "predict")

    model = init_model()
    start_time = time.time()
    for input_path in tqdm(input_paths):
        predict(model, input_path, output_dir)
    print("using time: %.2fs" %(time.time()-start_time))
    
if __name__ == '__main__':
    data_dir = "../datasets/naicrs/datasetC/trainval/images"
    output_dir = "results/"
    f = open("../datasets/naicrs/txt/valC.txt", "r")
    filenames = [item.replace("\n", "") for item in f.readlines()]
    input_paths = [os.path.join(data_dir, filename) for filename in filenames]
    
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
        
    generate_outputs('code', input_paths=input_paths, output_dir=output_dir)
    