#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 12:48:54 2020

@author: alex
"""
import matplotlib.pyplot as plt




def main():
    train_acc = []
    val_acc = []
    train_acc_list = []
    val_acc_list = []
    
    Acc_file = open("log-train1.txt", "r")#get train&val acc from txt file
    for line in Acc_file:
        if "[300/315]" in line:
            train_acc.append(line)
        elif "Epoch(train)" in line:
            val_acc.append(line)
         
    i = 0
    while i < len(train_acc):#get train acc to an array from the list of the log
        result = train_acc[i].find('acc')
        temp = train_acc[i][result+5:result+12]
        temp=float(temp)        
        train_acc_list.append(temp)
        i += 1             

    print(train_acc_list)
    
    i = 0
    while i < len(val_acc):#get val acc to an array from the list of the log
        result = val_acc[i].find('acc')
        temp2 = val_acc[i][result+5:result+12]
        temp2=float(temp2)        
        val_acc_list.append(temp2)
        i += 1             

    print(val_acc_list)  
    
    
    plt.plot(train_acc_list, label="Trainnig set")
    plt.plot(val_acc_list, label="Validation set")
    plt.ylim(94, 100)
    #plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.legend( loc='lower right')
    plt.xlabel('Number of epochs')
    plt.ylabel('Accuracy (%)')
    plt.axes
    plt.savefig('foo.png')
    plt.show()


if __name__ == '__main__':
    main()