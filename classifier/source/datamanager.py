import os, random

import numpy as np
import source.utils as utils

class DataSet(object):

    def __init__(self, dir='dataset'):

        print("\n** Prepare the Dataset")

        list_class = utils.sorted_list(path=os.path.join(dir, '*'))
        self.num_class = len(list_class)

        self.data_tr, self.data_val, self.data_te = [], [], []
        for path_class in list_class:
            list_npz = utils.sorted_list(path=os.path.join(path_class, '*.npz'))
            print(len(list_npz) // 10)
            bound = len(list_npz) // 10
            self.data_tr.extend(list_npz[:bound*8])
            self.data_val.extend(list_npz[bound*8:bound*9])
            self.data_te.extend(list_npz[bound*9:])
        self.__reset_index()

        self.num_tr, self.num_val, self.num_te = \
            len(self.data_tr), len(self.data_val), len(self.data_te)

        try:
            x, a, _, _ = self.next_batch(batch_size=1, ttv=0)
            self.dim_n, self.dim_f = x.shape[-2], x.shape[-1]
        except:
            x, a, _, _ = self.next_batch(batch_size=1, ttv=1)
            print(x.shape)
            self.dim_n, self.dim_f = x.shape[-2], x.shape[-1]

        self.__reset_index()
        print("\n* Summary")
        print("Training   : %d" %(self.num_tr))
        print("Validation : %d" %(self.num_val))
        print("Test       : %d" %(self.num_te))

    def __reset_index(self):

        self.idx_tr, self.idx_val, self.idx_te = 0, 0, 0

    def next_batch(self, batch_size=1, ttv=0):

        if(ttv == 0):
            idx_d, num_d, data = self.idx_tr, self.num_tr, self.data_tr
        elif(ttv == 1):
            idx_d, num_d, data = self.idx_te, self.num_te, self.data_te
        else:
            idx_d, num_d, data = self.idx_val, self.num_val, self.data_val

        batch_x, batch_a, batch_y, terminate = [], [], [], False
        while(True):

            try: tmp_npz = np.load(data[idx_d], allow_pickle=True)
            except:
                idx_d = 0
                terminate = True
                break
            else:
                batch_x.append(tmp_npz['feature'])
                batch_a.append(tmp_npz['adjacency'] + np.diag(np.ones(tmp_npz['adjacency'].shape[0])))
                batch_y.append(np.diag(np.ones(self.num_class))[tmp_npz['label']])
                idx_d += 1

                if(len(batch_x) >= batch_size): break

        batch_x = np.asarray(batch_x)
        batch_a = np.asarray(batch_a)
        batch_y = np.asarray(batch_y)

        if(ttv == 0):
            self.idx_tr, self.num_tr, self.data_tr = idx_d, num_d, data
        elif(ttv == 1):
            self.idx_te, self.num_te, self.data_te = idx_d, num_d, data
        else:
            self.idx_val, self.num_val, self.data_val = idx_d, num_d, data

        return batch_x.astype(np.float32), batch_a.astype(np.float32), batch_y.astype(np.float32), terminate
