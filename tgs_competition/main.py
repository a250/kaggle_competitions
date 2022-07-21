# entry point to start process through module processing_module

import pandas as pd
from processing_module import model, get_deep, get_files_list, write_model_csv

#deep_df = get_deep()
#print(deep_frame)

qty_epoch = 1000
qty_files = 3999

file_obj = {}

file_obj['filename'] = 'models/09-27_a'
file_obj['deep_filename'] = 'depths.csv'
file_obj['train_images_path'] = 'train/images'
file_obj['train_masks_path'] = 'train/masks'

imgs, msks = get_files_list(train_images_path = file_obj['train_images_path'] ,train_masks_path = file_obj['train_masks_path'])

file_obj['train_image_files'] = imgs[:qty_files]
file_obj['train_mask_files'] = msks[:qty_files]

##print('training len', len(file_obj['train_image_files']))
#print(file_obj['train_image_files'])
input_layer_dim = 25*25+1 # add 1 extra fiture (deep)


layers_dims = [input_layer_dim, 45, 30, 15, 1]

parameters, res_cost = model(file_obj, layers_dims, beta = 0.9999, learning_rate = 0.00005, optimizer = 'adam', num_epochs= qty_epoch, continue_model = False)

print('Result',res_cost)