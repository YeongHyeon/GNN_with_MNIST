import os

import numpy as np
import matplotlib.pyplot as plt

import source.utils as utils

def perform_from_confmat(confusion_matrix, num_class, verbose=False):

    dict_perform = {'accuracy':0, 'precision':0, 'recall':0, 'f1score':0}

    for idx_c in range(num_class):
        precision = np.nan_to_num(confusion_matrix[idx_c, idx_c] / np.sum(confusion_matrix[:, idx_c]))
        recall = np.nan_to_num(confusion_matrix[idx_c, idx_c] / np.sum(confusion_matrix[idx_c, :]))
        f1socre = np.nan_to_num(2 * (precision * recall / (precision + recall)))

        dict_perform['accuracy'] += confusion_matrix[idx_c, idx_c]
        dict_perform['precision'] += precision
        dict_perform['recall'] += recall
        dict_perform['f1score'] += f1socre

        if(verbose):
            print("Class-%d | Precision: %.5f, Recall: %.5f, F1-Score: %.5f" \
                %(idx_c, precision, recall, f1socre))

    for key in list(dict_perform.keys()):
        if('accuracy' == key): dict_perform[key] = dict_perform[key] / np.sum(confusion_matrix)
        else: dict_perform[key] = dict_perform[key] / num_class

    return dict_perform

def training(agent, dataset, batch_size, epochs):

    savedir = 'results_tr'
    utils.make_dir(path=savedir, refresh=True)

    print("\n** Training of the AE to %d epoch | Batch size: %d" %(epochs, batch_size))
    iteration = 0
    best_val = {'entropy':1e+10, 'accuracy':0, 'precision':0, 'recall':0, 'f1score':0}
    for epoch in range(epochs):

        while(True):
            x_tr, a_tr, y_tr, terminate = dataset.next_batch(batch_size=batch_size, ttv=0)
            if(len(x_tr.shape) == 1): break
            step_dict = agent.step(x=x_tr, a=a_tr, y=y_tr, iteration=iteration, training=True)
            iteration += 1
            if(terminate): break

        print("Epoch [%d / %d] | Loss: %f" %(epoch, epochs, step_dict['losses']['entropy']))

        loss_val_tmp = []
        confusion_matrix = np.zeros((dataset.num_class, dataset.num_class), np.int32)
        while(True):
            x_val, a_val, y_val, terminate = dataset.next_batch(batch_size=batch_size, ttv=2)
            if(len(x_val.shape) == 1): break
            step_dict = agent.step(x=x_val, a=a_val, y=y_val, iteration=-1, training=False)
            loss_val_tmp.append(step_dict['losses']['entropy'])
            for idx_y, _ in enumerate(y_val):
                confusion_matrix[np.argmax(y_val[idx_y]), np.argmax(step_dict['y_hat'][idx_y])] += 1
            if(terminate): break
        dict_perform = perform_from_confmat(confusion_matrix=confusion_matrix, num_class=dataset.num_class)

        for idx_k, name_key in enumerate(list(best_val.keys())):
            if(name_key == 'entropy'):
                if(best_val[name_key] > step_dict['losses']['entropy']):
                    best_val[name_key] = step_dict['losses']['entropy']
                    agent.save_params(model='model_%d_%s' %(idx_k+1, name_key), tflite=False)
            else:
                if(best_val[name_key] < dict_perform[name_key]):
                    best_val[name_key] = dict_perform[name_key]
                    agent.save_params(model='model_%d_%s' %(idx_k+1, name_key), tflite=False)

        agent.save_params(model='model_0_finepocch')

    for idx_k, name_key in enumerate(list(best_val.keys())):
        print(name_key, best_val[name_key])

def test(agent, dataset):

    savedir = 'results_te'
    utils.make_dir(path=savedir, refresh=True)

    list_model = utils.sorted_list(os.path.join('Checkpoint', 'model*'))
    for idx_model, path_model in enumerate(list_model):
        list_model[idx_model] = path_model.split('/')[-1]
    list_model.insert(0, 'base')

    f1_max = -(1e+12)
    best_dict = {'f1_name': '', 'f1score': 0}
    for idx_model, path_model in enumerate(list_model):

        print("\n** Test with %s" %(path_model))
        agent.load_params(model=path_model)
        utils.make_dir(path=os.path.join(savedir, path_model), refresh=False)

        confusion_matrix = np.zeros((dataset.num_class, dataset.num_class), np.int32)
        while(True):
            x_te, a_te, y_te, terminate = dataset.next_batch(batch_size=1, ttv=1)
            if(len(x_te.shape) == 1): break
            step_dict = agent.step(x=x_te, a=a_te, y=y_te, training=False)

            for idx_y, _ in enumerate(y_te):
                confusion_matrix[np.argmax(y_te[idx_y]), np.argmax(step_dict['y_hat'][idx_y])] += 1

            if(terminate): break

        dict_perform = perform_from_confmat(confusion_matrix=confusion_matrix, num_class=dataset.num_class, verbose=True)

        np.save(os.path.join(savedir, path_model, 'conf_mat.npy'), confusion_matrix)

        if(f1_max < dict_perform['f1score']):
            f1_max = dict_perform['f1score']
            auroc_name = path_model
            best_dict = {'f1_name': auroc_name, 'f1score': float(dict_perform['f1score'])}

    return best_dict, len(list_model)
