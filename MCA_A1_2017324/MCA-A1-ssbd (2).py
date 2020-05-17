#!/usr/bin/env python
# coding: utf-8

# In[44]:


import numpy as np
import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy
from math import sqrt


# In[45]:


# img = cv2.imread("HW-1/images/oxford_001216.jpg",0)
img = cv2.imread('sunflower.jpg',0)  


# In[46]:


plt.imshow(img)
plt.show()


# In[47]:


img = cv2.imread('sunflower.jpg',0)  


# In[48]:


# k = 1.414
# sigma = 1.0
img = img/255.0


# In[49]:


def mesh_grid(l,h,rows,columns):
    n=int(h-l+1)
    a=np.zeros((int(rows)+1,1))
    b=np.zeros((1,int(columns)+1))
    for i in range(n):
        a[i][0]=l+i
        b[0][i]=l+i
    return a,b


# In[50]:


print(mesh_grid(-6//2,6/2,(6/2)-(-6/2),(6/2)-(-6/2)))


# In[51]:


def lap_of_gau(sigma):
    var=pow(sigma,2)
    n = np.ceil(6*sigma)
    y,x = mesh_grid(-n/2,n/2,n,n)
    x_filter = np.asarray([np.exp(-(xx*xx/(2*var))) for xx in x])
    y_filter = np.asarray([np.exp(-(yy*yy/(2*var))) for yy in y])
    x_s=x*x
    y_s=y*y
    fin=y_filter*x_filter
    den=scipy.pi*var**2*2
    a = (x_s+y_s+(-2)*var)
    b = 1/den
    filter_f = a*b*fin
    return filter_f


# In[52]:


def none_array(h):
    return [None for i in range(h)]


# In[53]:


def convolution(img,k,sigma):
    log_images = [None for i in range(9)]
    ind=0
    for i in range(9):
        y = pow(k,i)
        filter_log = lap_of_gau(y*sigma)
        image = cv2.filter2D(img,-2,filter_log) 
        image = np.square(image)
        log_images[i]=image
    log_image_np=none_array(len(log_images))
    for i in range(len(log_images)):
        log_image_np[i]=log_images[i].reshape(img.shape[0], img.shape[1])
    return np.asarray(log_image_np)


log_image_np = convolution(img,sqrt(2),1)
#print(log_image_np.shape)


# In[54]:


def unflatten(e,shape0,shape1,shape2):
    a=e%shape2
    e=e//shape2
    b=e%shape1
    c=e//shape1
    return [c,b,a]  


# In[56]:


def detection(log_image_np,k,sigma):
    co_ordinates = [] 
#     (h,w) = img.shape
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            slice_img = log_image_np[:,i:i+3,j:j+3] 
            result = np.max(slice_img) 
            if result >= 0.03: 
                coords=unflatten(slice_img.argmax(),slice_img.shape[0],slice_img.shape[1],slice_img.shape[2])
                z=coords[0]
                x=coords[1]
                y=coords[2]
                co_ordinates.append((i+x,j+y,k**z*sigma)) 
    return co_ordinates

co_ordinates = detection(log_image_np,sqrt(2),1)


# In[57]:


plt.imshow(img,cmap="gray") 
for blob in co_ordinates:
    y,x,r = blob
    c = plt.Circle((x, y), r*sqrt(2), color='yellow', linewidth=0.5, fill=False)
    plt.gcf().gca().add_artist(c)
plt.plot()  
plt.show()


# In[ ]:




