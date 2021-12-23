from CNN_EMNIST import Character_Recognition as CR
import matplotlib.pyplot as plt
from emnist import extract_test_samples
import torch as T

if __name__ == '__main__': 

    #Uncomment to test CNN with EMNIST balanced test dataset---------------------------
    #test_images_original, test_labels_original = extract_test_samples('balanced')
    #test_images  = T.from_numpy((test_images_original.copy()/255.0).reshape(18800, 1, 28, 28))
    #test_labels  = T.from_numpy((test_labels_original.copy().astype(int)))
    #CR.test(test_images, test_labels)
    #-----------------------------------------------------------------------------------

    #Uncomment to test CNN with local image---------------------------------------------
    input = plt.imread("example_image.png")
    prediction = CR.classify(input)
    character = CR.decode(prediction)
    print("Character classified as: {}".format(character))
    #-----------------------------------------------------------------------------------
