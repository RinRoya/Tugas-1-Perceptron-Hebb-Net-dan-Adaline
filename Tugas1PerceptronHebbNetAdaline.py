#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Mengimport library yang diperlukan
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


# Inisialisasi input bipolar
xxb_input = np.array([[1,1,1],
                      [1,-1,1],
                      [-1,1,1],
                      [-1,-1,1]])


# Inisialisasi learning rate
learning_rate = np.random.uniform(0.1,0.2)

# Inisialisasi threshold
threshold = np.random.uniform(0,0.25)

# Inisialisasi toleransi
tol = 0.05


# In[3]:


# Inisialisasi logika NAND dengan target bipolar
target_nand = np.array([-1,1,1,1])

# Inisialisasi logika AND NOT dengan target bipolar
target_andnot = np.array([-1,1,-1,-1])

# Inisialisasi logika OR dengan target bipolar
target_or = np.array([1,1,1,-1])

# Inisialisasi logika NOR dengan target bipolar
target_nor = np.array([-1,-1,-1,1])

# Inisialisasi logika XOR dengan target bipolar
target_xor = np.array([-1,1,1,-1])


# In[4]:


# Function untuk metode Perceptron
def PerceptronMethod(xxb,t,alpha,threshold):
#     Inisialisasi weights dan bias serta wadah weight changes
    wwb = np.zeros(xxb.shape[1])
    weight_changes = np.ones(xxb.shape)
    max_iteration = 1
    
    while not np.array_equal(weight_changes,np.zeros(xxb.shape)) and max_iteration<100:
        for i in range(len(t)):
            y_in = sum(xxb[i,:]*wwb)
            theta = threshold
            if y_in > theta:
                y = 1
            elif y_in < -theta:
                y = -1
            else:
                y = 0
            temp = wwb
            if y != t[i]:
                wwb = wwb + alpha*t[i]*xxb[i,:]
            weight_changes[i,:] = wwb-temp
        max_iteration += 1
    return wwb


# In[5]:


# Function untuk metode Hebb Net
def HebbNetMethod(xxb,y):
#     Inisialisasi weights dan bias
    wwb = np.zeros(xxb.shape[1])
    
    for i in range(len(y)):
        wwb = wwb + xxb[i,:]*y[i]
    return wwb


# In[6]:


# Function untuk metode Adaline
def AdalineMethod(xxb,t,alpha,tol):
#     Inisialisasi weight dan bias
    wwb = np.zeros(xxb.shape[1])
    delta = np.full(xxb.shape[1],tol+1)
    max_iteration = 1
    
    while max(delta[:-1])>tol and max_iteration<=100:
        for i in range(len(t)):
            y_in = sum(xxb[i,:]*wwb)
            delta = alpha*(t[i]-y_in)*xxb[i,:]
            wwb = wwb + delta
        max_iteration += 1
    if not (np.array_equal(t,TestingAdaline(xxb,wwb))):
        wwb = np.zeros(xxb.shape[1])
    return wwb

# Function untuk testing Adaline
def TestingAdaline(xxb,wwb):
    y = np.zeros(xxb.shape[0])
    
    for i in range(xxb.shape[0]):
        y_in = sum(xxb[i,:]*wwb)
        if y_in >=0:
            y[i] = 1
        else:
            y[i] = -1
    return y


# In[7]:


# Hasil keluaran dengan metode Perceptron
perceptron_result = {
    0 : PerceptronMethod(xxb_input,target_nand,learning_rate,threshold), # Perceptron dengan logika AND
    1 : PerceptronMethod(xxb_input,target_andnot,learning_rate,threshold), # Perceptron dengan logika AND NOT
    2 : PerceptronMethod(xxb_input,target_or,learning_rate,threshold), # Perceptron dengan logika OR
    3 : PerceptronMethod(xxb_input,target_nor,learning_rate,threshold), # Perceptron dengan logika NOR
    4 : PerceptronMethod(xxb_input,target_xor,learning_rate,threshold) # Perceptron dengan logika XOR
}

# Hasil keluaran dengan metode Hebb Nett
hebbnet_result = {
    0 : HebbNetMethod(xxb_input,target_nand), # Hebb Net dengan logika AND
    1 : HebbNetMethod(xxb_input,target_andnot), # Hebb Net dengan logika AND NOT
    2 : HebbNetMethod(xxb_input,target_or), # Hebb Net dengan logika OR
    3 : HebbNetMethod(xxb_input,target_nor), # Hebb Net dengan logika NOR
    4 : HebbNetMethod(xxb_input,target_xor) # Hebb Net dengan logika XOR
}

# Hasil keluaran dengan metode Adaline
adaline_result = {
    0 : AdalineMethod(xxb_input,target_nand,learning_rate,tol), # Adaline dengan logika AND
    1 : AdalineMethod(xxb_input,target_andnot,learning_rate,tol), # Adaline dengan logika AND NOT
    2 : AdalineMethod(xxb_input,target_or,learning_rate,tol), # Adaline dengan logika OR
    3 : AdalineMethod(xxb_input,target_nor,learning_rate,tol), # Adaline dengan logika NOR
    4 : AdalineMethod(xxb_input,target_xor,learning_rate,tol), # Adaline dengan logika XOR
}


# In[8]:


# Komparasi hasil keluaran dengan metode Perceptron, Hebb Net, dan Adaline

# Membuat header tabel
print("Metode Perceptron".center(40),end="")
print("Metode Hebb Net".center(40),end="")
print("Metode Adaline".center(40))

print("".center(120,"-"))
for k in range(3):
    print("Logika".center(10) + "w1".center(10) + "w2".center(10)+ "b".center(10),end="")
print("".center(120,"-"))

# Membuat nama logika yang digunakan
logic_name = {
    0 : "NAND",
    1 : "AND NOT",
    2 : "OR",
    3 : "NOR",
    4 : "XOR"
}

for i in range(5):
    logic = logic_name.get(i)
    result = {
        0 : perceptron_result[i],
        1 : hebbnet_result[i],
        2 : adaline_result[i]
    }
    
#     Menampilkan hasil training
    for j in range(3):
        print(logic.center(10) + 
              "{:.2f}".format(result[j][0]).center(10) +
              "{:.2f}".format(result[j][1]).center(10) + 
              "{:.2f}".format(result[j][2]).center(10), end="")
    print()
    
print("".center(120,"-"))


# In[19]:


# Visualisasi hasil

target = {
    0 : target_nand,
    1 : target_andnot,
    2 : target_or,
    3 : target_nor,
    4 : target_xor,
}
title = {
    0 : ["Perceptron NAND","Perceptron AND NOT","Perceptron OR","Perceptron NOR","Perceptron XOR"],
    1 : ["Hebb Net NAND","Hebb Net AND NOT","Hebb Net OR","Hebb Net NOR","Hebb Net XOR"],
    2 : ["Adaline NAND","Adaline AND NOT","Adaline OR","Adaline NOR","Adaline XOR"]
}
result = {
    0 : perceptron_result,
    1 : hebbnet_result,
    2 : adaline_result
}

# Function untuk memvisualisasikan hasil
def plot_result(k):
    plt.figure(figsize=(30,30))
    for i in range(5):
#         Warna marker
        color = ["r" if c==1 else "b" for c in target[i]]
        
#         Transformasikan hasil ke dalam garis linier
        x = np.linspace(-2,2,2)
        if result.get(k)[i][1]==0:
            m=0
            n=3
        else:
            m = -result.get(k)[i][0]/result.get(k)[i][1]
            n = -result.get(k)[i][2]/result.get(k)[i][1]
        y = m*x+n
        
#         Visualisasikan hasil
        plt.subplot(1,5,i+1)
        plt.plot(x,y,"g-")
#         Visualisasikan marker penanda target
        plt.scatter(xxb_input[:,0],xxb_input[:,1],c=color,linewidths=10)
        plt.scatter(3,3,c="r",linewidths=5,label="+1")
        plt.scatter(3,3,c="b",linewidths=5,label="-1")

#         Buat garis sumbu-x dan sumbu-y
        plt.plot([-2, 2],[0, 0],"k-")
        plt.plot([0, 0],[-2, 2],"k-")
        
#         Buat keterangan grafik
        plt.title(title.get(k)[i],fontsize = 20)
        plt.legend(fontsize = "x-large",loc = "best")
        plt.axis("square")
        
#         Batasi sumbu-x dan sumbu-y
        plt.xlim(-2,2)
        plt.ylim(-2,2)
        plt.xticks([])
        plt.yticks([])
    plt.show()


# In[20]:


# Komparasikan visualisasi hasil
for k in range (3):
    plot_result(k)





