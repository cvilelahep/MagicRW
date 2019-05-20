import sys
import pickle
import matplotlib.pyplot as plt

def main(fName) :

    d = pickle.load(open(fName, "r"))
    
    plt.plot(d["test"]["rmse"])
    plt.plot(d["train"]["rmse"])
    
    plt.show()
    raw_input()

if __name__ == "__main__" :
    fName_ = sys.argv[1]
    main(fName_)
