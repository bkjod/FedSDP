import zmq 
import sys
import time
import resnet_model
import torch
import argparse
import json
import random
from data_pre_shap import get_dataset
import os
import numpy
from collections import OrderedDict
import torchvision
import torch
from torch import nn
import traceback
import csv


# evaluate the model by data from data_loader
def eval_model(model, data_loader):
    print("Start evaluating the model!")
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    total_loss = 0.0
    correct = 0
    dataset_size = 0

    for batch_id, batch in enumerate(data_loader):
        data, target = batch
        dataset_size += data.size()[0]
        if torch.cuda.is_available():
            data = data.to(device)
            target = target.to(device)
        output = model(data)

        total_loss += torch.nn.functional.cross_entropy(
              output,
              target,
              reduction='sum'
            ).item()

        pred = output.data.max(1)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()
            
    acc = 100.0 * (float(correct) / float(dataset_size))
    total_l = total_loss / dataset_size
 
    return acc, total_l

def valid_model(model, data_loader):
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    correct = 0
    dataset_size = 0

    for batch_id, batch in enumerate(data_loader):
        data, target = batch
        dataset_size += data.size()[0]
        if torch.cuda.is_available():
            data = data.to(device)
            target = target.to(device)
        output = model(data)

        pred = output.data.max(1)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()
            
    acc = 100.0 * (float(correct) / float(dataset_size))
 
    return acc, int(correct)

def gaussian_mechanism(x, eps, delta, privacy_rate):
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    sigma = numpy.sqrt(2 * numpy.log(1.25 / delta)) / eps / 100
    noise = numpy.random.normal(loc=0, scale=sigma, size=x.shape)
    privacy_rate /= 2
    privacy_rate -= 1
    if privacy_rate < -3:
        privacy_rate = -3
    elif privacy_rate > 1:
        privacy_rate = 1
    else:
        pass
    scale = numpy.linalg.norm(noise)
    noise = torch.tensor(noise)
    noise = noise.to(device)
    return x + noise, scale

# aggregrate the model
def aggregrate_model(global_model, recieved_model, conf, e, privacy_r):
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    # bring out the first K gradient
    active_recieved = recieved_model[:conf["k"]]
    scale = 0
    # average without weight
    global_gradient = global_model.state_dict()
    for name, data in global_gradient.items():
        global_gradient[name] = torch.zeros_like(data).to(device).float()
    if conf["model_name"] == "resnet18":
        gra = resnet_model.ResNet18(num=conf["CLASS_NUM"]).to(device)
    elif conf["model_name"] == "vgg16":
        gra = resnet_model.Vgg16(num=conf["CLASS_NUM"]).to(device)
    elif conf["model_name"] == "CNN":
        gra = resnet_model.CNN(num=conf["CLASS_NUM"]).to(device)
    elif conf["model_name"] == "LSTM":
        gra = resnet_model.LSTM(num=conf["CLASS_NUM"]).to(device)
    else:
        pass

    for name, data in global_model.state_dict().items():
        for seq in range(len(active_recieved)):
            gra_way = active_recieved[seq]
            privacy_rate = privacy_r[seq]
            gra.load_state_dict(torch.load(gra_way[1]))
            gra_state = gra.state_dict()
            resource_seed = random.random()
            if resource_seed > -1:
                update_layer = (gra_state[name] / conf["k"]) 
                update_layer, s = gaussian_mechanism(update_layer, 0.1, 1/conf["batch_size"], privacy_rate)
                scale += s
            else:
                update_layer = (gra_state[name] / conf["k"]) 

            global_gradient[name] += update_layer
        
        if data.type() != global_gradient[name].type():
            global_gradient[name] = torch.round(global_gradient[name]).to(torch.int64)
        else:
            pass

        data.copy_(global_gradient[name])

    with open('sflshapdpAvg_'+conf["model_name"]+'_'+conf["type"]+'_scale_with'+'_alpha_'+str(conf["alpha"])+'.csv', mode='a+', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([e, scale])

    return global_model

# train model
def train_model(model, optimizer, data_loader, conf, seq):
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    model.train()
    gra_dict = {}
    for name, data in model.state_dict().items():
        gra_dict[name] = model.state_dict()[name].clone()

    for e in range(conf["local_epochs"]):
        for batch_id, batch in enumerate(data_loader):
            data, target = batch
            if torch.cuda.is_available():
                data = data.to(device)
                target = target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = torch.nn.functional.cross_entropy(output, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=conf["clip"])
            optimizer.step()

            if batch_id % 10 == 0:
                print("\t \t Finish ", batch_id, "/", len(data_loader), "batches.")
        
        print("\t Client", seq, " finsh ", e, " epoches train! ")

    torch.save(model.state_dict(), "./sfl/gradient_" + str(seq) + ".pt")

    return model

# main function
def main():
    # get config
    parser = argparse.ArgumentParser(description='Federated Learning')
    parser.add_argument('-c', '--conf', dest='conf')
    args = parser.parse_args()
    with open(args.conf, 'r', encoding='utf-8') as f:
        conf = json.load(f)
    eval_loader, eval_loader_list = get_dataset("../data/", conf["type"], "s", conf, -1)
    workers = conf["no_models"]
    worker_conf = {}
    for i in range(workers):
        resource = 1
        print("Client ", i, " has ", resource, " resource.")
        time.sleep(0.5)
        usual_loader, privacy_loader, train_loader, validation_loader_p, validation_loader_t = get_dataset("../data/", conf["type"], "c", conf, i)
        worker_conf[i] = [resource, usual_loader, privacy_loader, train_loader, validation_loader_p, validation_loader_t, 0, 0, "./sfl/global_model_0.pt"]
    global_epoch = 0
    have_recieved_model = []
    time_clock = 0
    uploaded_model = 0
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    if conf["model_name"] == "resnet18":
        global_model = resnet_model.ResNet18(num=conf["CLASS_NUM"]).to(device)
    elif conf["model_name"] == "vgg16":
        global_model = resnet_model.Vgg16(num=conf["CLASS_NUM"]).to(device)
    elif conf["model_name"] == "CNN":
        global_model = resnet_model.CNN(num=conf["CLASS_NUM"]).to(device)
    elif conf["model_name"] == "LSTM":
        global_model = resnet_model.LSTM(num=conf["CLASS_NUM"]).to(device)
    else:
        pass
    torch.save(global_model.state_dict(), "./sfl/global_model_0.pt")

    start_time = time.time()
    while global_epoch < conf["global_epochs"]:

        print("\nGlobal Epoch ", global_epoch, " Starts! \n")
        
        active_client = random.sample(range(conf["no_models"]), conf["k"])

        for client_seq_number in range(workers):

            if client_seq_number in active_client:
                if client_seq_number == 1:
                    # start train
                    print("\n \n Client ", client_seq_number, "start train!")
                    # load newest global model and dataloader
                    train_loader = worker_conf[client_seq_number][1]
                    using_train_model =  worker_conf[client_seq_number][-1]

                    if conf["model_name"] == "resnet18":
                        local_model = resnet_model.ResNet18(num=conf["CLASS_NUM"]).to(device)
                    elif conf["model_name"] == "vgg16":
                        local_model = resnet_model.Vgg16(num=conf["CLASS_NUM"]).to(device)
                    elif conf["model_name"] == "CNN":
                        local_model = resnet_model.CNN(num=conf["CLASS_NUM"]).to(device)
                    elif conf["model_name"] == "LSTM":
                        local_model = resnet_model.LSTM(num=conf["CLASS_NUM"]).to(device)
                    else:
                        pass
                    local_model.load_state_dict(torch.load(using_train_model))
                    privacy_rate = 0

                    # train
                    optimizer = torch.optim.SGD(local_model.parameters(), lr=conf['local_lr'], momentum=conf['local_momentum'])
                    local_model = train_model(local_model, optimizer, train_loader, conf, client_seq_number)
                    # compute the updation
                    print("Client ", client_seq_number, "finish train and upload gradient!")
                    gra =  "./sfl/gradient_" + str(client_seq_number) + ".pt"
                    have_recieved_model.append([client_seq_number, gra, privacy_rate])          # update the model to server
                    uploaded_model += 1
                else:
                    # start train
                    print("\n \n Client ", client_seq_number, "start train!")
                    # load newest global model and dataloader
                    usual_loader = worker_conf[client_seq_number][1]
                    privacy_loader = worker_conf[client_seq_number][2]
                    train_loader = worker_conf[client_seq_number][3]
                    validation_loader_p = worker_conf[client_seq_number][4]
                    validation_loader_t = worker_conf[client_seq_number][5]
                    using_train_model =  worker_conf[client_seq_number][-1]

                    if conf["model_name"] == "resnet18":
                        local_model = resnet_model.ResNet18(num=conf["CLASS_NUM"]).to(device)
                    elif conf["model_name"] == "vgg16":
                        local_model = resnet_model.Vgg16(num=conf["CLASS_NUM"]).to(device)
                    elif conf["model_name"] == "CNN":
                        local_model = resnet_model.CNN(num=conf["CLASS_NUM"]).to(device)
                    elif conf["model_name"] == "LSTM":
                        local_model = resnet_model.LSTM(num=conf["CLASS_NUM"]).to(device)
                    else:
                        pass
                    local_model.load_state_dict(torch.load(using_train_model))
                    optimizer = torch.optim.SGD(local_model.parameters(), lr=conf['local_lr'], momentum=conf['local_momentum'])
                    local_model = train_model(local_model, optimizer, usual_loader, conf, client_seq_number)
                    u2p_acc, u2p_nun = valid_model(local_model, validation_loader_p)
                    u2u_acc, u2u_nun = valid_model(local_model, validation_loader_t)
                    print("\n Usual:", u2p_acc, u2u_acc, u2p_nun, u2u_nun)
                    local_model.load_state_dict(torch.load(using_train_model))
                    optimizer = torch.optim.SGD(local_model.parameters(), lr=conf['local_lr'], momentum=conf['local_momentum'])
                    local_model = train_model(local_model, optimizer, privacy_loader, conf, client_seq_number)
                    p2p_acc, p2p_nun = valid_model(local_model, validation_loader_p)
                    p2u_acc, p2u_nun = valid_model(local_model, validation_loader_t)
                    print("\n Privacy:", p2p_acc, p2u_acc, p2p_nun, p2u_nun)
                    local_model.load_state_dict(torch.load(using_train_model))
                    optimizer = torch.optim.SGD(local_model.parameters(), lr=conf['local_lr'], momentum=conf['local_momentum'])
                    local_model = train_model(local_model, optimizer, train_loader, conf, client_seq_number)
                    f2p_acc, f2p_nun = valid_model(local_model, validation_loader_p)
                    f2u_acc, f2u_nun = valid_model(local_model, validation_loader_t)
                    print("\n Full:", f2p_acc, f2u_acc, f2p_nun, f2u_nun)


                    full_acc = f2p_nun+f2u_nun
                    privacy_acc = p2p_nun+p2u_nun
                    usual_acc = u2p_nun+u2u_nun

                    if u2u_nun < 10:
                        privacy_acc = int(privacy_acc / 9)
                    usual_sharkley = full_acc - privacy_acc + usual_acc
                    privacy_sharkley = full_acc + privacy_acc - usual_acc

                    if privacy_sharkley + usual_sharkley == 0:
                        privacy_rate = 0
                    elif privacy_acc == 0 and usual_acc == 0:
                        privacy_rate = -10
                    else:
                        privacy_rate = privacy_sharkley / (privacy_sharkley + usual_sharkley)
                    print(usual_sharkley, privacy_sharkley, privacy_rate, ((numpy.e)**(privacy_rate/3 - 0.3)))

                    print("Client ", client_seq_number, "finish train and upload gradient!")
                    gra =  "./sfl/gradient_" + str(client_seq_number) + ".pt"
                    have_recieved_model.append([client_seq_number, gra, privacy_rate])          # update the model to server
                    uploaded_model += 1
                             
            else:
                print("Client ", client_seq_number, "keep training!") # keep training(idling)
        
        recieved_amount = len(have_recieved_model)
        print("\nUsing ", time_clock, " time clocks and recieve ", recieved_amount, " models! \n")

        time.sleep(0.5)

        if recieved_amount < conf["k"]:
            print("Waiting for enough models! Need ", conf["k"], ", but recieved ", recieved_amount)  # have not recieved enough models, keep waiting
        else:
            print("Having recieved enough models. Need ", conf["k"], ", and recieved ", recieved_amount)
            # aggregrate
            privacy_r =[]
            for update in have_recieved_model:
                privacy_r.append(update[2])

            global_model = aggregrate_model(global_model, have_recieved_model, conf, global_epoch, privacy_r)

            # evaluation
            total_acc, total_loss = eval_model(global_model, eval_loader) 
            this_time = time.time()
            print("Global Epoch ", global_epoch, "\t total loss: ", total_loss, " \t total acc: ", total_acc)

            with open('sflshapdp_'+conf["model_name"]+'_'+conf["type"]+'_acc_with'+'_alpha_'+str(conf["alpha"])+'.csv', mode='a+', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([global_epoch, total_acc, total_loss, this_time - start_time, privacy_r])
            have_recieved_model = have_recieved_model[conf["k"]:]
            global_epoch += 1
            torch.save(global_model.state_dict(), "./sfl/global_model_"+str(global_epoch)+".pt")
            for client_seq_number in range(workers):
                worker_conf[client_seq_number][-1] = './sfl/global_model_'+str(global_epoch)+'.pt'

            print("Finish aggregrate and leave ", len(have_recieved_model), " models!")

        time.sleep(0.5)

if __name__ == "__main__":
    main()
