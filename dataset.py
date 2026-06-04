import pickle

def unPickle(file):
    with open(file,'rb') as f:
        dict = pickle.load(f,encoding='bytes')
    return dict

def test():
    file_path = "dataset/data_batch_1"
    dict = unPickle(file_path)
    print(dict)

if __name__ == "__main__":
    test()