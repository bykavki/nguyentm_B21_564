from features import *
from segmentation import *
from ast import literal_eval

features=pd.read_csv('result.csv', index_col='letter')

img_bin = binarized(Image.open('letters/0.png'))

weight = get_norm_weight(img_bin)
x, y = get_relative_center(img_bin)
I_x, I_y = get_norm_inertia_moment(img_bin)

print(weight, x, y, I_x, I_y)


letter_features = features.loc['a']


distance = np.sqrt((literal_eval(letter_features['rel_center'])[0] - x)**2 + (literal_eval(letter_features['rel_center'])[1] - y)**2 +\
                           (literal_eval(letter_features['norm_inertia'])[0] - I_x)**2 + (literal_eval(letter_features['norm_inertia'])[1] - I_y)**2 +\
                           (letter_features['norm_weight'] - weight)**2)

print(distance)




