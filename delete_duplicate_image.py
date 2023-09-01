# -*- coding: utf-8 -*-
# Commented out IPython magic to ensure Python compatibility.
import hashlib
import cv2
import matplotlib.pyplot as plt
import os


if __name__ == "__main__":
    os.getcwd()
    os.chdir(r'valuation')
    file_list = os.listdir()
    print(len(file_list))

    duplicates = []
    hash_keys = dict()
    for index, filename in enumerate(os.listdir('.')):  # listdir('.') = current directory
        if os.path.isfile(filename):
            with open(filename, 'rb') as f:
                filehash = hashlib.md5(f.read()).hexdigest()
            if filehash not in hash_keys:
                hash_keys[filehash] = index
            else:
                duplicates.append((index, hash_keys[filehash]))
    print(duplicates)
    # for file_indexes in duplicates[:30]:
    #     try:
    #
    #         plt.subplot(121), plt.imshow(cv2.imread(file_list[file_indexes[1]]))
    #         plt.title(file_indexes[1]), plt.xticks([]), plt.yticks([])
    #
    #         plt.subplot(122), plt.imshow(cv2.imread(file_list[file_indexes[0]]))
    #         plt.title(str(file_indexes[0]) + ' duplicate'), plt.xticks([]), plt.yticks([])
    #         plt.show()
    #
    #     except OSError as e:
    #         continue
    """# Delete Files After Printing"""

    for index in duplicates:
        os.remove(file_list[index[0]])