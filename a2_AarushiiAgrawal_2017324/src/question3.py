import pickle
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

pickle_out = open("Dataset/spectograms/spec1.pickle","wb")
pickle.dump(spec_X, pickle_out)
pickle_out.close()

pickle_out = open("mfcc.pickle","wb")
pickle.dump(mfcc_X, pickle_out)
pickle_out.close()

pickle_out = open("Dataset/spectograms/vspec1.pickle","wb")
pickle.dump(vspec_X, pickle_out)
pickle_out.close()

pickle_out = open("vmfcc.pickle","wb")
pickle.dump(vmfcc_X, pickle_out)
pickle_out.close()

#Spectogram

to_remove=[]
vto_remove=[]

for i in range(len(vmfcc_X)):
    if len(vmfcc_X[i])!=67:
        print(len(vmfcc_X[i]))
        vmfcc_X.remove(vmfcc_X[i])
        vto_remove.append(i)

for i in range(len(mfcc_X)):
    if len(mfcc_X[i])!=67:
        print(len(mfcc_X[i]))
        mfcc_X.remove(mfcc_X[i])
        to_remove.append(i)

for i in range(len(mfcc_Y)):
    if i in to_remove:
        mfcc_Y.remove(mfcc_Y[i])

for i in range(len(vmfcc_Y)):
    if i in vto_remove:
        vmfcc_Y.remove(vmfcc_Y[i])

vmfcc_Y.remove(1)

mfcc_Y.remove(1)

mfcc_Y.remove(2)

zeros = [0 for i in range(128)]
a=0
while a==0:
    a=1
    for i in range(len(spec_X)):
        if len(spec_X[i][0])!=123:
            a=0
            print(len(spec_X[i][0]))
            spec_X[i]=np.insert(spec_X[i],0,zeros,axis=1)
            print("final ",len(spec_X[i][0]))

zeros = [0 for i in range(128)]
a=0
while a==0:
    a=1
    for i in range(len(vspec_X)):
        if len(vspec_X[i][0])!=123:
            a=0
            print(len(vspec_X[i][0]))
            vspec_X[i]=np.insert(vspec_X[i],0,zeros,axis=1)
            print("final ",len(vspec_X[i][0]))

X = [[] for i in range(len(spec_X))]

for i in range(len(spec_X)):
    X[i].extend(np.asarray(spec_X[i]).flatten())

vX = [[] for i in range(len(vspec_X))]

for i in range(len(vspec_X)):
    vX[i].extend(np.asarray(vspec_X[i]).flatten())

clf1 = SVC(gamma='auto')
clf1.fit(X,spec_Y)

clf1.predict(vX)

print(accuracy_score(vspec_Y,clf1.predict(vX)))

#MFCC    

Xx = [[] for i in range(len(mfcc_X))]

for i in range(len(mfcc_X)):
    Xx[i].extend(np.asarray(mfcc_X[i]).flatten())

vXx = [[] for i in range(len(vmfcc_X))]

for i in range(len(vmfcc_X)):
    vXx[i].extend(np.asarray(vmfcc_X[i]).flatten())

clf2 = SVC(gamma='auto',kernel='poly')
clf2.fit(Xx,mfcc_Y)

clf2.predict(vXx)


vmy = []
for i in range(50):
    vmy.append(i//5)

accuracy_score(vmy,clf2.predict(vXx))

from sklearn.metrics import recall_score
recall_score(vmy,clf2.predict(vXx),average=None)

from sklearn.metrics import precision_score
precision_score(vmy,clf2.predict(vXx),average=None)


#combining all spetrograms for different classes

pickle_in = open("Dataset/spectograms/spec1.pickle","rb")
spec1 = pickle.load(pickle_in)
pickle_in = open("Dataset/spectograms/spec2.pickle","rb")
spec2 = pickle.load(pickle_in)
pickle_in = open("Dataset/spectograms/spec3.pickle","rb")
spec3 = pickle.load(pickle_in)
pickle_in = open("Dataset/spectograms/spec4.pickle","rb")
spec4 = pickle.load(pickle_in)
pickle_in = open("Dataset/spectograms/spec5.pickle","rb")
spec5 = pickle.load(pickle_in)

pickle_in = open("Dataset/spectograms/vspec1.pickle","rb")
vspec1 = pickle.load(pickle_in)
pickle_in = open("Dataset/spectograms/vspec2.pickle","rb")
vspec2 = pickle.load(pickle_in)
pickle_in = open("Dataset/spectograms/vspec3.pickle","rb")
vspec3 = pickle.load(pickle_in)
pickle_in = open("Dataset/spectograms/vspec4.pickle","rb")
vspec4 = pickle.load(pickle_in)
pickle_in = open("Dataset/spectograms/vspec5.pickle","rb")
vspec5 = pickle.load(pickle_in)

spec = spec1
spec.extend(spec2)
spec.extend(spec3)
spec.extend(spec4)
spec.extend(spec5)

vspec = vspec1
vspec.extend(vspec2)
vspec.extend(vspec3)
vspec.extend(vspec4)
vspec.extend(vspec5)

zeros = [0 for i in range(128)]
a=0
while a==0:
    a=1
    for i in range(len(spec)):
        if len(spec[i][0])!=123:
            a=0
            print(len(spec[i][0]))
            spec[i]=np.insert(spec[i],0,zeros,axis=1)
            print("final ",len(spec[i][0]))

zeros = [0 for i in range(128)]
a=0
while a==0:
    a=1
    for i in range(len(vspec)):
        if len(vspec[i][0])!=123:
            a=0
            print(len(vspec[i][0]))
            vspec[i]=np.insert(vspec[i],0,zeros,axis=1)
            print("final ",len(vspec[i][0]))

X = [[] for i in range(len(spec))]

for i in range(len(spec)):
    X[i].extend(np.asarray(spec[i]).flatten())

vX = [[] for i in range(len(vspec))]

for i in range(len(vspec)):
    vX[i].extend(np.asarray(vspec[i]).flatten())

Y = [0,0]
for i in range(1000):
    Y.append(i//100)
    print(i//100)

vY = []
for i in range(100):
    vY.append(i//10)
    print(i//10)

clf1 = SVC(gamma='auto',kernel='poly')
clf1.fit(X,Y)

pred = clf1.predict(vX)
# len(pred)

print(accuracy_score(vY,pred))

from sklearn.metrics import recall_score
recall_score(vY,pred,average=None)

from sklearn.metrics import precision_score
precision_score(vY,pred,average=None)
