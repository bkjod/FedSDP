# -*- coding: utf-8 -*-

from torchvision import datasets, transforms
import numpy as np
import torch
import random
import json

def get_dataset(dir, name, roll, conf, user_id):

    if torch.cuda.is_available():
        pin = True
    else:
        pin = False

    if name == 'mnist':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        transform_test = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
 
        train_dataset = datasets.EMNIST(dir, train=True, download=True, transform=transform_train,split = 'byclass' )
        eval_dataset = datasets.EMNIST(dir, train=False, transform=transform_test,split = 'byclass' )
    elif name == 'cifar10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
 
        train_dataset = datasets.CIFAR10(dir, train=True, download=True, transform=transform_train)
        eval_dataset = datasets.CIFAR10(dir, train=False, transform=transform_test)
    elif name == 'cifar100':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
 
        train_dataset = datasets.CIFAR100(dir, train=True, download=True, transform=transform_train)
        eval_dataset = datasets.CIFAR100(dir, train=False, transform=transform_test)
    
    
    if roll == "s":
        label_to_indices = {}
        eval_loader_list = {}

        for i in range(len(eval_dataset)):
            label = eval_dataset.targets[i]
            if label not in label_to_indices:
                label_to_indices[label] = []
            if len(label_to_indices[label]) < conf["server_eval_size"]:
                label_to_indices[label].append(i)

        
        for label, indices in label_to_indices.items():
            eval_loader_label = torch.utils.data.DataLoader(eval_dataset,batch_size=conf["batch_size"],pin_memory = pin,
                                sampler=torch.utils.data.sampler.SubsetRandomSampler(indices), num_workers=4)
            eval_loader_list[label] = eval_loader_label
        eval_loader = torch.utils.data.DataLoader(eval_dataset,batch_size=conf["batch_size"],shuffle=True, drop_last=True,pin_memory = pin, num_workers=4)
        return eval_loader, eval_loader_list
    else:
        if user_id == 0:
            if conf["redistribution"] == "y":
                label_to_indices = {}
                eval_loader_list = {}

                for i in range(len(train_dataset)):
                    label = int(train_dataset.targets[i])
                    if label not in label_to_indices:
                        label_to_indices[label] = []
                    label_to_indices[label].append(i)
                if conf["non_iid"] == "shap":
                    Data_partition = []
                    Privacy_partition = []
                    labels = list(label_to_indices.keys())
                    labels.remove(conf["target_label"])
                    for i in range(conf["no_models"]):
                        data_indices = []
                        privacy_indices = []
                        if i == 1:
                            for k,v in label_to_indices.items():
                                if k in labels:
                                    data_indices += random.sample(v, int(conf["alpha"]*len(v)))
                        else:
                            for k,v in label_to_indices.items():
                                if k in labels:
                                    data_indices += random.sample(v, int(conf["alpha"]*len(v)))
                                else:
                                    privacy_indices += random.sample(v, int(conf["alpha"]*len(v)))
                                
                        Data_partition.append(data_indices)
                        Privacy_partition.append(privacy_indices)
                elif conf["non_iid"] == "iid":
                    Data_partition = []
                    Privacy_partition = []
                    for i in range(conf["no_models"]):
                        data_indices = random.sample(list(range(0, len(train_dataset))), int(conf["alpha"]*len(train_dataset)))
                        print(len(data_indices))
                        Data_partition.append(data_indices)
                    else:
                        pass

                torch.save(Data_partition, "./data_partition_with_"+conf["non_iid"]+"_dataset_"+str(conf["type"])+"_"+str(conf["alpha"])+"_and_"+str(conf["no_models"])+"_models.pt")
                torch.save(Privacy_partition, "./privacy_partition_with_"+conf["non_iid"]+"_dataset_"+str(conf["type"])+"_"+str(conf["alpha"])+"_and_"+str(conf["no_models"])+"_models.pt")
            else:
                Data_partition = torch.load("./data_partition_with_"+conf["non_iid"]+"_dataset_"+str(conf["type"])+"_"+str(conf["alpha"])+"_and_"+str(conf["no_models"])+"_models.pt")
                Privacy_partition = torch.load("./privacy_partition_with_"+conf["non_iid"]+"_dataset_"+str(conf["type"])+"_"+str(conf["alpha"])+"_and_"+str(conf["no_models"])+"_models.pt")
        else:
            Data_partition = torch.load("./data_partition_with_"+conf["non_iid"]+"_dataset_"+str(conf["type"])+"_"+str(conf["alpha"])+"_and_"+str(conf["no_models"])+"_models.pt")
            Privacy_partition = torch.load("./privacy_partition_with_"+conf["non_iid"]+"_dataset_"+str(conf["type"])+"_"+str(conf["alpha"])+"_and_"+str(conf["no_models"])+"_models.pt")

        train_indices = []
        if conf["non_iid"] == "shap":
            train_indices = Data_partition[user_id]
            privcay_indices = Privacy_partition[user_id]
            train = int(len(train_indices)*0.8)
            privacy = int(len(privcay_indices)*0.8)
            # validation_indices = privcay_indices[privacy:] + train_indices[train:]
            validation_indices_p = privcay_indices[privacy:]
            validation_indices_t = train_indices[train:]
            train_indices = train_indices[:train]
            privcay_indices = privcay_indices[:privacy]
            # print(len(train_indices), len(privcay_indices), len(validation_indices_p))
        else:
            pass

        usual_loader = torch.utils.data.DataLoader(train_dataset, batch_size=conf["batch_size"], pin_memory = pin,drop_last=True,
                                    sampler=torch.utils.data.sampler.SubsetRandomSampler(train_indices), num_workers=4)
        privacy_loader = torch.utils.data.DataLoader(train_dataset, batch_size=conf["batch_size"], pin_memory = pin,drop_last=True,
                                    sampler=torch.utils.data.sampler.SubsetRandomSampler(privcay_indices), num_workers=4)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=conf["batch_size"], pin_memory = pin,drop_last=True,
                                    sampler=torch.utils.data.sampler.SubsetRandomSampler(train_indices+privcay_indices), num_workers=4)
        validation_loader_p = torch.utils.data.DataLoader(train_dataset, batch_size=conf["batch_size"], pin_memory = pin,drop_last=True,
                                    sampler=torch.utils.data.sampler.SubsetRandomSampler(validation_indices_p), num_workers=4)
        validation_loader_t = torch.utils.data.DataLoader(train_dataset, batch_size=conf["batch_size"], pin_memory = pin,drop_last=True,
                                    sampler=torch.utils.data.sampler.SubsetRandomSampler(validation_indices_t), num_workers=4)
        return usual_loader,privacy_loader,train_loader,validation_loader_p,validation_loader_t

if __name__ == "__main__":
    import json
    import time
    with open("tconf.json", 'r', encoding='utf-8') as f:
        conf = json.load(f)

    train_loader = get_dataset("../data/", conf["type"], "c", conf, 0)
