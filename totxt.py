import tf_tools

array = tf_tools.get_files('./train_data')
array = tf_tools.batches(array[0], array[1])
with open('data.txt', 'w') as file:
    file.write(str(len(array))+' '+str(len(array[0])-1)+' '+ '23\n')
    for iter in array:
        for i in iter:
            file.write(str(int(i))+' ')
        file.write('\n')
