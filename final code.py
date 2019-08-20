
#preparing model for predicting dissolved oxygen Q value


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


dataset_do = pd.read_csv('do.csv')
X_do = dataset_do.iloc[:, 0:1].values
y_do = dataset_do.iloc[:, 1:].values


from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
poly_reg_do = PolynomialFeatures(degree = 4)
X_poly_do = poly_reg_do.fit_transform(X_do)
lin_reg_do = LinearRegression()
lin_reg_do.fit(X_poly_do, y_do)




X_grid_do = np.arange(min(X_do), max(X_do), 0.2)
X_grid_do = X_grid_do.reshape((len(X_grid_do), 1))
plt.scatter(X_do, y_do, color = 'red')
plt.plot(X_grid_do, lin_reg_do.predict(poly_reg_do.fit_transform(X_grid_do)), color = 'blue')

#preparing model for predicting change in temperature Q value

dataset_temp=pd.read_csv('temp.csv')
X_temp=dataset_temp.iloc[:,0:1].values
y_temp=dataset_temp.iloc[:,1:].values
 

poly_reg_temp=PolynomialFeatures(degree=6)
X_poly_temp=poly_reg_temp.fit_transform(X_temp)
lin_reg_temp=LinearRegression()
lin_reg_temp.fit(X_poly_temp,y_temp)



X_grid_temp=np.arange(min(X_temp),max(X_temp),0.1)
X_grid_temp=X_grid_temp.reshape((len(X_grid_temp),1))
plt.scatter(X_temp,y_temp,color='red')
plt.plot(X_grid_temp,lin_reg_temp.predict(poly_reg_temp.fit_transform(X_grid_temp)),color='blue')

#preparing model for predicting ph Q value
 
dataset_ph=pd.read_csv('ph.csv')
X_ph=dataset_ph.iloc[:,0:1].values
y_ph=dataset_ph.iloc[:,1:].values


poly_reg_ph=PolynomialFeatures(degree=12)
X_poly_ph=poly_reg_ph.fit_transform(X_ph)
lin_reg_ph=LinearRegression()
lin_reg_ph.fit(X_poly_ph,y_ph)

X_grid_ph=np.arange(min(X_ph),max(X_ph),0.1)
X_grid_ph=X_grid_ph.reshape((len(X_grid_ph),1))
plt.scatter(X_ph,y_ph,color='red')
plt.plot(X_grid_ph,lin_reg_ph.predict(poly_reg_ph.fit_transform(X_grid_ph)),color='blue')



#preparing model for predicting phosphate Q value
 

dataset_phos=pd.read_csv('phosphate.csv')
X_phos=dataset_phos.iloc[:,0:1].values
y_phos=dataset_phos.iloc[:,1:].values


poly_reg_phos=PolynomialFeatures(degree=6)
X_poly_phos=poly_reg_phos.fit_transform(X_phos)
lin_reg_phos=LinearRegression()
lin_reg_phos.fit(X_poly_phos,y_phos)

X_grid_phos=np.arange(min(X_phos),max(X_phos),0.1)
X_grid_phos=X_grid_phos.reshape((len(X_grid_phos),1))
plt.scatter(X_phos,y_phos,color='red')
plt.plot(X_grid_phos,lin_reg_phos.predict(poly_reg_phos.fit_transform(X_grid_phos)),color='blue')


#preparing model for predicting nitrate Q value



dataset_nit=pd.read_csv('nitrate.csv')
X_nit=dataset_nit.iloc[:,0:1].values
y_nit=dataset_nit.iloc[:,1:].values


poly_reg_nit=PolynomialFeatures(degree=6)
X_poly_nit=poly_reg_nit.fit_transform(X_nit)
lin_reg_nit=LinearRegression()
lin_reg_nit.fit(X_poly_nit,y_nit)

X_grid_nit=np.arange(min(X_nit),max(X_nit),0.1)
X_grid_nit=X_grid_nit.reshape((len(X_grid_nit),1))
plt.scatter(X_nit,y_nit,color='red')
plt.plot(X_grid_nit,lin_reg_nit.predict(poly_reg_nit.fit_transform(X_grid_nit)),color='blue')


#preparing model for predicting turbidity Q value



dataset_tur=pd.read_csv('turbidity file.csv')
X_tur=dataset_tur.iloc[:,0:1].values
y_tur=dataset_tur.iloc[:,1:].values


poly_reg_tur=PolynomialFeatures(degree=4)
X_poly_tur=poly_reg_tur.fit_transform(X_tur)
lin_reg_tur=LinearRegression()
lin_reg_tur.fit(X_poly_tur,y_tur)

X_grid_tur=np.arange(min(X_tur),max(X_tur),0.1)
X_grid_tur=X_grid_tur.reshape((len(X_grid_tur),1))
plt.scatter(X_tur,y_tur,color='red')
plt.plot(X_grid_tur,lin_reg_tur.predict(poly_reg_tur.fit_transform(X_grid_tur)),color='blue')

#calculating the main dataset and predicting the optimal location of water

dataset=pd.read_csv('datset112.csv')


# extracting Q values


tur2=poly_reg_tur.fit_transform(dataset.iloc[:,0:1].values)
dataset.iloc[:,0:1]=lin_reg_tur.predict(tur2)

ph2=poly_reg_ph.fit_transform(dataset.iloc[:,1:2].values)
dataset.iloc[:,1:2]=lin_reg_ph.predict(ph2)


temp2=poly_reg_temp.fit_transform(dataset.iloc[:,2:3].values)
dataset.iloc[:,2:3]=lin_reg_temp.predict(temp2)

phos2=poly_reg_phos.fit_transform(dataset.iloc[:,3:4].values)
dataset.iloc[:,3:4]=lin_reg_phos.predict(phos2)

nit2=poly_reg_nit.fit_transform(dataset.iloc[:,4:5].values)
dataset.iloc[:,4:5]=lin_reg_nit.predict(nit2)

do2=poly_reg_do.fit_transform(dataset.iloc[:,5:6].values)
dataset.iloc[:,5:6]=lin_reg_do.predict(do2)





X_tur1=(dataset.iloc[:,0:1].values)*0.08
X_ph1=(dataset.iloc[:,1:2].values)*0.11
X_temp1=(dataset.iloc[:,2:3].values)*0.1
X_phos1=(dataset.iloc[:,3:4].values)*0.1
X_nit1=(dataset.iloc[:,4:5].values)*0.1
X_do1=(dataset.iloc[:,5:6].values)*0.17
pollution=(X_tur1+X_ph1+X_temp1+X_phos1+X_nit1+X_do1)/0.66
pollution.shape






from math import sin, cos, sqrt, atan2, radians

# approximate radius of earth in km
def vamshi(lat1,lon1,lat2,lon2):
    R = 6373.0

    lat1 = radians(lat1)
    lon1 = radians(lon1)
    lat2 = radians(lat2)
    lon2 = radians(lon2)

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2 
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    return R * c

print("enter the village location")
lat1=float(input())    
lon1=float(input())
totalcost=200000000
pollution_cost=10
travel_cost=5
lat2=dataset.iloc[:,7:8].values
lon2=dataset.iloc[:,8:9].values
cd=pollution.shape



for i in range (0,pollution.size):
    k=vamshi(lat1,lon1,lat2[i],lon2[i])
    if(pollution[i]>65 and pollution[i]<82 and k<200):
        pollution[i]=(82-pollution[i])*pollution_cost
    k=k*travel_cost+pollution[i]
    print("cur cost=",k)
    if(totalcost>k and k<200000):
            totalcost=k
            popy=i

print("total cost=",totalcost)
print("Latitude= ",lat2[popy])
print("Longitude",lon2[popy])


