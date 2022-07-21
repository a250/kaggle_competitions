from processing_module import * 

def predict(Line):
    Line['prediction'] = ((forward_propagation(np.array(Line['extended_scaled_surround']).reshape(1,-1).T, parameters)[0])[0][0] >0.5) *1
    return Line

offset = 0
file_columns = ['id','rle_mask']


file_obj = {}
file_obj['filename'] = 'models/09-15_d'
parameters = load_parameters(file_obj)

test_file_obj = {}
test_file_obj['test_images_path'] = 'images'
test_file_obj['test_image_files'] = get_testfiles_list(test_file_obj['test_images_path'])
test_files = test_file_obj['test_image_files']
test_file_obj['result_file'] = 'results/res_09-19'

deep_df = get_deep()


if isfile(test_file_obj['result_file']+'.csv'):
    res_df = pd.read_csv(test_file_obj['result_file']+'.csv', names = file_columns)
    offset = len(res_df.index)-1
    print('continue file: {} from pos# {}:'.format(test_file_obj['result_file']+'.csv', offset))
else:
    res_df = pd.DataFrame(columns = ['id','rle_mask']).set_index('id')
    res_df.to_csv(test_file_obj['result_file']+'.csv', mode='w', header=True)

start_time = time.time()
iter_time = time.time()


for i,f in enumerate(test_files[offset:]):
    res_df = pd.DataFrame(columns = file_columns)
    test_item_frame = processing_single_testfile(i, test_file_obj, deep_df)
    test_predict = test_item_frame.apply(predict, axis = 1)
    test_predict = test_predict.sort_values(['px_id'])
    pic_res = np.array(test_predict['prediction']).reshape(101,101)
    item = {}
    item['id'] = f
    item['rle_mask'] = convert_mask(pic_res)
    res_df = res_df.append([item], sort = False).set_index('id')
    res_df.to_csv(test_file_obj['result_file']+'.csv', mode='a', header=False)
    
    if i%10 == 0:
        print('i: {}, i+offset: {}, iter time: {:}, total time: {:}'.format(i,i+offset,time.time()- iter_time ,time.time()- start_time))
        iter_time = time.time()