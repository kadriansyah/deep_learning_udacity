# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import random
import hashlib
import json
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
from six.moves import cPickle as pickle

image_size = 28  # Pixel width and height.
pixel_depth = 255.0  # Number of levels per pixel.

def load_letter(folder, min_num_images):
    """Load the data for a single letter label."""
    image_files = os.listdir(folder)
    dataset = np.ndarray(shape=(len(image_files), image_size, image_size), dtype=np.float32)
    num_images = 0
    for image in image_files:
        image_file = os.path.join(folder, image)
        try:
            image_data = (ndimage.imread(image_file).astype(float) - pixel_depth / 2) / pixel_depth
            if image_data.shape != (image_size, image_size):
                raise Exception('Unexpected image shape: %s' % str(image_data.shape))
            dataset[num_images, :, :] = image_data
            num_images = num_images + 1
        except IOError as e:
            print('Could not read:', image_file, ':', e, '- it\'s ok, skipping.')

    dataset = dataset[0:num_images, :, :]
    if num_images < min_num_images:
        raise Exception('Many fewer images than expected: %d < %d' % (num_images, min_num_images))

    print('Full dataset tensor:', dataset.shape)
    print('Mean:', np.mean(dataset))
    print('Standard deviation:', np.std(dataset))
    return dataset

def maybe_pickle(data_folders, min_num_images_per_class, force=False):
    dataset_names = []
    for folder in data_folders:
        set_filename = folder + '.pickle'
        dataset_names.append(set_filename)
        if os.path.exists(set_filename) and not force:
            # You may override by setting force=True.
            print('%s already present - Skipping pickling.' % set_filename)
        else:
            print('Pickling %s.' % set_filename)
            dataset = load_letter(folder, min_num_images_per_class)
            try:
                with open(set_filename, 'wb') as f:
                    pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
            except Exception as e:
                print('Unable to save data to', set_filename, ':', e)

    return dataset_names

def make_arrays(nb_rows, img_size):
    if nb_rows:
        dataset = np.ndarray((nb_rows, img_size, img_size), dtype=np.float32)
        labels = np.ndarray(nb_rows, dtype=np.int32)
    else:
        dataset, labels = None, None
    return dataset, labels

def merge_datasets(pickle_files, train_size, valid_size=0):
    num_classes = len(pickle_files)
    valid_dataset, valid_labels = make_arrays(valid_size, image_size)
    train_dataset, train_labels = make_arrays(train_size, image_size)
    vsize_per_class = valid_size // num_classes
    tsize_per_class = train_size // num_classes

    start_v, start_t = 0, 0
    end_v, end_t = vsize_per_class, tsize_per_class
    end_l = vsize_per_class+tsize_per_class
    for label, pickle_file in enumerate(pickle_files):
        try:
            with open(pickle_file, 'rb') as f:
                letter_set = pickle.load(f)
                # let's shuffle the letters to have random validation and training set
                np.random.shuffle(letter_set)
                if valid_dataset is not None:
                    valid_letter = letter_set[:vsize_per_class, :, :]
                    valid_dataset[start_v:end_v, :, :] = valid_letter
                    valid_labels[start_v:end_v] = label
                    start_v += vsize_per_class
                    end_v += vsize_per_class

                train_letter = letter_set[vsize_per_class:end_l, :, :]
                train_dataset[start_t:end_t, :, :] = train_letter
                train_labels[start_t:end_t] = label
                start_t += tsize_per_class
                end_t += tsize_per_class
        except Exception as e:
            print('Unable to process data from', pickle_file, ':', e)
            raise
    return valid_dataset, valid_labels, train_dataset, train_labels

def randomize(dataset, labels):
    permutation = np.random.permutation(labels.shape[0])
    shuffled_dataset = dataset[permutation,:,:]
    shuffled_labels = labels[permutation]
    return shuffled_dataset, shuffled_labels

# begin program
root = os.path.dirname(os.path.abspath(__file__)) + '/notMNIST_large'
train_folders = [os.path.join(root, d) for d in sorted(os.listdir(root)) if os.path.isdir(os.path.join(root, d))]
# print(train_folders)

root = os.path.dirname(os.path.abspath(__file__)) + '/notMNIST_small'
test_folders = [os.path.join(root, d) for d in sorted(os.listdir(root)) if os.path.isdir(os.path.join(root, d))]
# print(test_folders)

train_datasets = maybe_pickle(train_folders, 45000)
test_datasets = maybe_pickle(test_folders, 1800)

train_size = 200000
valid_size = 10000
test_size = 10000

valid_dataset, valid_labels, train_dataset, train_labels = merge_datasets(train_datasets, train_size, valid_size)
_, _, test_dataset, test_labels = merge_datasets(test_datasets, test_size)

print('Training:', train_dataset.shape, train_labels.shape)
print('Validation:', valid_dataset.shape, valid_labels.shape)
print('Testing:', test_dataset.shape, test_labels.shape)

train_dataset, train_labels = randomize(train_dataset, train_labels)
test_dataset, test_labels = randomize(test_dataset, test_labels)
valid_dataset, valid_labels = randomize(valid_dataset, valid_labels)

# # Finally, let's save the data for later reuse:
# pickle_file = 'notMNIST.pickle'
# try:
#     f = open(pickle_file, 'wb')
#     save = {
#         'train_dataset': train_dataset,
#         'train_labels': train_labels,
#         'valid_dataset': valid_dataset,
#         'valid_labels': valid_labels,
#         'test_dataset': test_dataset,
#         'test_labels': test_labels,
#     }
#     pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
#     f.close()
# except Exception as e:
#     print('Unable to save data to', pickle_file, ':', e)
#     raise
#
# statinfo = os.stat(pickle_file)
# print('Compressed pickle size:', statinfo.st_size)

# ### Problem 1
# # Let's take a peek at some of the data to make sure it looks sensible. Each exemplar should be an image of a character A
# # through J rendered in a different font. Display a sample of the images that we just downloaded. Hint: you can use the
# # package IPython.display.
# base_dir = os.getcwd() + "/notMNIST_large/"
# letters = [chr(ord('A') + i) for i in range(0,10) ]
# for letter in letters:
#     letter_dir = base_dir + letter
#     random_image = random.choice(os.listdir(letter_dir))
#     display(Image(filename=letter_dir+ '/' + random_image))
#     print(letter)

# ### Problem 2
# # Let's verify that the data still looks good. Displaying a sample of the labels and images from the ndarray. Hint: you can
# # use matplotlib.pyplot.
# A_list = pickle.load(open("notMNIST_large/A.pickle", "rb"))
# random_letter = random.choice(A_list)
# plt.imshow(random_letter)
# plt.show()

# ### Problem 3
# # Another check: we expect the data to be balanced across classes. Verify that.
# letters = [chr(ord('A') + i) for i in range(0,10) ]
# for letter in letters:
#     letter_train_data = pickle.load(open('notMNIST_large/' + letter + '.pickle', "rb"))
#     print(letter + " train data count : " + str(len(letter_train_data)) )
#     letter_test_data = pickle.load(open('notMNIST_small/' + letter + '.pickle', "rb"))
#     print(letter + " test data count : " + str(len(letter_test_data)) )

# ### Problem 4
# fig, ax = plt.subplots(1,2)
# bins = np.arange(train_labels.min(), train_labels.max()+2)
# ax[0].hist(train_labels, bins=bins)
# ax[0].set_xticks((bins[:-1]+bins[1:])/2, [chr(k) for k in range(ord("A"), ord("J")+1)])
# ax[0].set_title("Training data")
#
# bins = np.arange(test_labels.min(), test_labels.max()+2)
# ax[1].hist(test_labels, bins=bins)
# ax[1].set_xticks((bins[:-1]+bins[1:])/2, [chr(k) for k in range(ord("A"), ord("J")+1)])
# ax[1].set_title("Test data")
# [chr(k) for k in range(ord("A"), ord("J")+1)]
#
# print((bins[:-1]+bins[1:])/2)
# print(train_labels.min(), train_labels.max())

# ### Problem 5
# all_data = pickle.load(open('notMNIST.pickle', 'rb'))
# def count_duplicates(dataset1, dataset2):
#     hashes = [hashlib.sha1(x).hexdigest() for x in dataset1]
#     dup_indices = []
#     for i in range(0, len(dataset2)):
#         if hashlib.sha1(dataset2[i]).hexdigest() in hashes:
#             dup_indices.append(i)
#     return len(dup_indices)
#
# print(count_duplicates(all_data['test_dataset'], all_data['valid_dataset']))
# print(count_duplicates(all_data['valid_dataset'], all_data['train_dataset']))
# print(count_duplicates(all_data['test_dataset'], all_data['train_dataset']))

### Problem 6
all_data = pickle.load(open('notMNIST.pickle', 'rb'))

train_dataset = all_data['train_dataset']
train_labels = all_data['train_labels']
test_dataset = all_data['test_dataset']
test_labels = all_data['test_labels']

def get_score(train_dataset, train_labels, test_dataset, test_labels):
    model = LogisticRegression()
    train_flatten_dataset = np.array([x.flatten() for x in train_dataset])
    test_flatten_dataset = np.array([x.flatten() for x in test_dataset])
    model.fit(train_flatten_dataset, train_labels)
    return model.score([x.flatten() for x in test_dataset], test_labels)

print("100 trainsamples score: " + str(get_score(train_dataset[:100], train_labels[:100], test_dataset, test_labels)))
print("1000 trainsamples score: " + str(get_score(train_dataset[:1000], train_labels[:1000], test_dataset, test_labels)))
print("5000 trainsamples score: " + str(get_score(train_dataset[:5000], train_labels[:5000], test_dataset, test_labels)))
print("10000 trainsamples score: " + str(get_score(train_dataset[:10000], train_labels[:10000], test_dataset, test_labels)))
print("all trainsamples score: " + str(get_score(train_dataset, train_labels, test_dataset, test_labels)))
