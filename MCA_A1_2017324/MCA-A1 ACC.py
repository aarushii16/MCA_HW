#!/usr/bin/env python
# coding: utf-8

# In[33]:


from PIL import Image
import numpy as np
import random


# In[30]:


path2="HW-1/images/"
import os


# In[37]:


image_path = path2
image_names = [f for f in os.listdir(image_path) if os.path.isfile(os.path.join(image_path,f)) and random.uniform(0,1)<0.02]


# In[39]:


len(image_names)


# In[27]:


image_set=["all_souls_000091","all_souls_000026","oxford_003410","oxford_002985","all_souls_000015","oxford_000926",
     "all_souls_000055","oxford_001955","cornmarket_000107","hertford_000035","jesus_000320","hertford_000034",
     "all_souls_000076","magdalen_000053","magdalen_000449","pitt_rivers_000115","pitt_rivers_000058",
     "radcliffe_camera_000440","radcliffe_camera_000360","oxford_000026","oxford_000246","oxford_000311",
     "oxford_000380","oxford_000383","oxford_000384","oxford_000385","oxford_000403","oxford_000402",
     "oxford_000401","oxford_000400","oxford_000410","oxford_000409","oxford_000440","oxford_000453"]


# In[40]:


image_names.extend(image_set)


# In[2]:


img1 = Image.open('HW-1/images/all_souls_000013.jpg') 
img2 = Image.open('HW-1/images/all_souls_000026.jpg')


# In[21]:


# ground_truth = open("HW-1/train/ground_truth/all_souls_1_good.txt","r")


# In[22]:


# query = open("HW-1/train/query/all_souls_1_query.txt","r")


# In[5]:


img_arr1 = np.asarray(img_arr1)
img_arr2 = np.asarray(img_arr2)


# In[4]:


img_arr1 = img1.resize((128,96),Image.ANTIALIAS)
img_arr2 = img2.resize((128,96),Image.ANTIALIAS)


# In[6]:


img_arr1 = img_arr1/255
img_arr2 = img_arr2/255


# In[7]:


arr_gs1 = np.asarray(img1.convert('LA'))
arr_gs2 = np.asarray(img2.convert('LA'))


# In[8]:


import cv2


# In[10]:


from sklearn.cluster import MiniBatchKMeans
import matplotlib.pyplot as plt


# In[11]:


def color_quantization(img_arr,k):
    img_arr2 = img_arr
    img_arr2 = img_arr2.reshape(img_arr.shape[0]*img_arr.shape[1],3)
    kmeans = MiniBatchKMeans(k)
    kmeans.fit(img_arr2)
    new_colors = kmeans.cluster_centers_[kmeans.predict(img_arr2)]
    centroids = kmeans.cluster_centers_
    new_colors2 = new_colors.reshape(img_arr.shape)
    fig, ax = plt.subplots(1, 2, figsize=(16, 6),subplot_kw=dict(xticks=[], yticks=[]))
    fig.subplots_adjust(wspace=0.05)
    ax[0].imshow(img_arr)
    ax[0].set_title('Original Image', size=16)
    ax[1].imshow(new_colors2)
    st = str(k)+"-color Image"
    ax[1].set_title(st, size=16)
    return new_colors2,centroids


# In[12]:


nc1,centroids1 = color_quantization(img_arr1,64)
nc2,centroids2 = color_quantization(img_arr2,64)


# In[13]:


def corr(nc,centroids):
    colors=[]
    
    for i in range(nc.shape[0]):
        for j in range(nc.shape[1]):
            if list(nc[i][j]) not in colors:
                colors.append(list(nc[i][j]))
                colors.append([0,0,0,0])
                colors.append([0,0,0,0])
            num = colors[colors.index(list(nc[i][j]))+1]
            den = colors[colors.index(list(nc[i][j]))+2]
            
            #sub-matrix of distace=7 around it
            for ii in range(i-7,i+8):
                for jj in range(j-7,j+8):
                    dist = max(abs(i-ii),abs(j-jj))
                    if dist==1 or dist==3 or dist==5 or dist==7:
                        den[((dist+1)//2)-1]+=1
                        if ii>=0 and jj>=0 and ii<nc.shape[0] and jj<nc.shape[1] and list(nc[i][j])==list(nc[ii][jj]):
                            num[((dist+1)//2)-1]+=1
            colors[colors.index(list(nc[i][j]))+1] = num
            colors[colors.index(list(nc[i][j]))+2] = den
                    
    
    acc=np.zeros((len(colors)//3,4))
    for k in range(len(colors)//3):
        temp=[]
        for p in range(4):
            a=colors[(k*3)+1]
            b=colors[(k*3)+2]
            temp.append(a[p]/b[p])
        acc[k] = temp
    return colors,acc


# In[14]:


# test = np.array([[0,0,0,0,0,0,0,0],
#                  [0,0,0,0,0,0,0,0],
#                   [0,0,0,0,0,0,0,0],
#                  [0,1,1,0,0,1,1,0],
#                  [0,1,1,0,0,1,1,0],
#                  [0,0,0,0,0,0,0,0],
#                  [0,0,0,0,0,0,0,0],
#                  [0,0,0,0,0,0,0,0]])


# In[15]:


# xy = np.zeros((8,8,3))


# In[16]:


# for i in range(test.shape[0]):
#     for j in range(test.shape[1]):
#         if test[i][j]==0:
#             xy[i][j]=[0,0,0]
#         elif test[i][j]==1:
#             xy[i][j]=[1,1,1]


# In[17]:


# cent = [[0,0,0],[1,1,1]]


# In[18]:


colors_test1,acc_test1 = corr(nc1,centroids1)
colors_test2,acc_test2 = corr(nc2,centroids2)


# In[19]:


def difference(acc1,acc2):
    return (np.abs(acc1-acc2)/(1+acc1+acc2)).sum()/64


# In[20]:


print(difference(acc_test1,acc_test2))


# In[29]:


img1 = Image.open('HW-1/images/all_souls_000013.jpg') 
img_arr1 = img1.resize((128,96),Image.ANTIALIAS)
img_arr1 = np.asarray(img_arr1)
img_arr1 = img_arr1/255
nc1,centroids1 = color_quantization(img_arr1,64)
colors_test1,acc_test1 = corr(nc1,centroids1)
differences=[]

for i in (image_set):
    img2 = Image.open("HW-1/images/"+i+".jpg")
    img_arr2 = img2.resize((128,96),Image.ANTIALIAS)
    img_arr2 = np.asarray(img_arr2)
    img_arr2 = img_arr2/255
    nc2,centroids2 = color_quantization(img_arr2,64)
    colors_test2,acc_test2 = corr(nc2,centroids2)

    print(difference(acc_test1,acc_test2))
    differences.append(difference(acc_test1,acc_test2))


# In[46]:


top=np.asarray(differences).argmin()


# In[47]:


s=image_set[top]
im="HW-1/images/"+s+".jpg"


# In[48]:


Image.open(im)


# In[45]:


s


# In[49]:


z=dict(zip(image_set,differences))


# In[50]:


zz=sorted(z.items(), key = 
             lambda kv:(kv[1], kv[0]))


# In[51]:


zz[:5]


# In[60]:


for i in zz[:10]:
    s=i[0]
    im="HW-1/images/"+s+".jpg"
    Image.open(im)
#     print(s)


# In[62]:


s=zz[0][0]
im="HW-1/images/"+s+".jpg"
Image.open(im)


# In[63]:


s=zz[1][0]
im="HW-1/images/"+s+".jpg"
Image.open(im)


# In[64]:


s=zz[2][0]
im="HW-1/images/"+s+".jpg"
Image.open(im)


# In[65]:


s=zz[4][0]
im="HW-1/images/"+s+".jpg"
Image.open(im)


# In[66]:


s=zz[5][0]
im="HW-1/images/"+s+".jpg"
Image.open(im)


# In[ ]:




