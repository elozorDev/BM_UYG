#! /usr/bin/env python
# -*- coding: UTF-8 -*-
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures

# CSV dosyasını oku
veri = pd.read_csv("2016dolaralis.csv")

print(veri.head())  # Verinin doğru okunup okunmadığını kontrol et

x = veri["Gun"].values.reshape(-1, 1)  # NumPy array'e çevir ve reshape yap
y = veri["Fiyat"].values.reshape(-1, 1)

# Scatter plot (Veri noktalarını göster)
plt.scatter(x, y, label="Gerçek Veriler")
plt.xlabel("Gün")
plt.ylabel("Fiyat")
plt.legend()
plt.show()

# Lineer Regresyon Modeli
tahminlineer = LinearRegression()
tahminlineer.fit(x, y)

plt.plot(x, tahminlineer.predict(x), color="red", label="Lineer Regresyon")
plt.scatter(x, y, label="Gerçek Veriler")
plt.legend()
plt.show()

# Polinom Regresyon (3. derece)
tahminpolinom = PolynomialFeatures(degree=3)
Xyeni = tahminpolinom.fit_transform(x)

polinommodel = LinearRegression()
polinommodel.fit(Xyeni, y)

plt.plot(x, polinommodel.predict(Xyeni), color="blue", label="Polinom Regresyon (3. Derece)")
plt.scatter(x, y, label="Gerçek Veriler")
plt.legend()
plt.show()

# Hata Hesaplama
hatakaresilineer = np.sum((y - tahminlineer.predict(x)) ** 2)
hatakaresipolinom = np.sum((y - polinommodel.predict(Xyeni)) ** 2)

print("Lineer Regresyon Hata Karesi:", hatakaresilineer)
print("Polinom Regresyon (3. Derece) Hata Karesi:", hatakaresipolinom)

# 8. Dereceden Polinom Regresyon
tahminpolinom8 = PolynomialFeatures(degree=8)
Xyeni8 = tahminpolinom8.fit_transform(x)

polinommodel8 = LinearRegression()
polinommodel8.fit(Xyeni8, y)

plt.plot(x, polinommodel8.predict(Xyeni8), color="green", label="Polinom Regresyon (8. Derece)")
plt.scatter(x, y, label="Gerçek Veriler")
plt.legend()
plt.show()

# 201. günü kontrol et
if 201 < len(y):
    print((float(y[201]) - float(polinommodel8.predict(Xyeni8)[201])))
else:
    print("Dizi boyutu 201'den küçük!")
