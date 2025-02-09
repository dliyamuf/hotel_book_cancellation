# -*- coding: utf-8 -*-
"""hotel_booking_cancellation.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1YPRWiQW92x6RJP262RQxRmaPcNwwJxwd

**PROBLEM SCOPE OF DOMAIN KNOWLEDGE**

Nowadays, travelers most significantly use internet to book hotel and make an reservation [1](https://doi.org/10.1177/1938965511418779). Hotel bookings represent a contract between customer and the hotel. The demand of hotel bookings can influence hotel's revenue, allocation, and budgets. Hence, search for factor of bookings cancellation can prevent revenue loss and get better marketing strategy [2](http://dx.doi.org/10.18089/tms.2017.13203).

***

**BUSINESS UNDERSTANDING**

**Problem Statement**

1. When is the most bookings cancellation date?
2. What is the major factor influence bookings cancellation?
3. How accurate is machine learning model can predict bookings cancellation? What is the best model?

**Goals**
1. Identify most bookings cancellation date.
2. Analysis of bookings cancellation's influence factor.
3. Investigate accuration of machine learning models prediction of booking cancellation and choose the best model.

**Solution statement**

1. Use correlation of arrival date and book cancellation data.
2. Explore features based on EDA, mutual information, and Pearson correlation?
3. Use 5 difference classification machine learning algorithm (Logistic Regresion, KNN, Random Forest, Gradient Boosting and XGBoost) with accuracy >95%. Algorithm selection is based on higher on accuracy and recall score.

***

# **DATA UNDERSTANDING**
Dataset from [Hotel Booking Demand](https://www.sciencedirect.com/science/article/pii/S2352340918315191#s0005) by Nuno Antonio, Ana de Almeida, and Luis Nunesthat collected from comprehend booking from July 1, 2015 until August 31, 2017. This dataset contains two hotel H1 and H2. H1 is an resort hotel and the other one (H2) is an city hotel.

**Variables**

- **IsCanceled**: Value indicating if the booking was
canceled (1) or not (0).
- **LeadTime**: Number of days that elapsed between
the entering date of the booking into the
PMS (Property Management System) and the arrival date
- **ArrivalDateYear**: Year of arrival date.
- **ArrivalDateMonth**: Month of arrival date with 12 categories:
“January” to “December”.
- **ArrivalDateWeekNumber**: Week number of the arrival date.
- **ArivalDateDayOfMonth**: Day of the month of the arrival date.
- **StaysInWeekendNights**: Number of weekend nights (Saturday or
Sunday) the guest stayed or booked to
stay at the hotel.
- **StaysInWeekNights**: Number of week nights (Monday to Friday) the guest stayed or booked to stay
at the hotel.
- **Adults**: Number of adults
- **Children**: Number of children.
- **Babies**: Number of babies.
- **Meal**: Type of meal booked. Categories are
presented in standard hospitality meal
packages:  Market segment designation. In
categories, the term “TA” means “Travel
Agents” and “TO” means “Tour
Operators”.
Undefined/SC – no meal package;
BB – Bed & Breakfast;
HB – Half board (breakfast and one
other meal – usually dinner);
FB – Full board (breakfast, lunch and
dinner)
- **Country**: Country of origin. Categories are represented in the ISO 3155–3:2013 forma.
- **MarketSegment**: Market segment designation. In
categories, the term “TA” means “Travel
Agents” and “TO” means “Tour
Operators”.
- **DistributionChannel**: Booking distribution channel. The term
“TA” means “Travel Agents” and “TO”
means “Tour Operators”.
- **IsRepeatedGuest**: Value indicating if the booking name
was from a repeated guest (1) or not (0).
- **PreviousCancellations**: Number of previous bookings that were
cancelled by the customer prior to the
current booking.
- **PreviousBookingsNotCanceled**:  Number of previous bookings not
cancelled by the customer prior to the
current booking.
- **ReservedRoomType**: Code of room type reserved. Code is
presented instead of designation for
anonymity reasons.
- **AssignedRoomType**: Code for the type of room assigned to the
booking. Sometimes the assigned room
type differs from the reserved room type
due to hotel operation reasons (e.g.
overbooking) or by customer request.
Code is presented instead of designation
for anonymity reasons.
- **BookingChanges**: Number of changes/amendments made
to the booking from the moment the
booking was entered on the PMS until
the moment of check-in or cancellation.
- **DepositType**: Indication on if the customer made a
deposit to guarantee the booking. This
variable can assume three categories: No Deposit – no deposit was made; Non Refund – a deposit was made in
the value of the total stay cost; Refundable – a deposit was made with
a value under the total cost of stay.
- **Agent**:  ID of the travel agency that made the
booking.
- **Company**: ID of the company/entity that made the
booking or responsible for paying the
booking. ID is presented instead of designation for anonymity reasons.
- **DaysInWaitingList**:  Number of days the booking was in the
waiting list before it was confirmed to
the customer.
- **CustomerType**: Type of booking, assuming one of four
categories:
BO and BL
Contract - when the booking has an
allotment or other type of contract
associated to it;
Group – when the booking is associated to a group;
Transient – when the booking is not
part of a group or contract, and is not
associated to other transient booking;
Transient-party – when the booking is
transient, but is associated to at least
other transient booking.
- **ADR**: Average Daily Rate as defined by American Hotel and Lodging Association.
- **RequiredCarParkingSpaces**: Number of car parking spaces required
by the customer.
- **TotalOfSpecialRequests**:  Number of special requests made by the
customer (e.g. twin bed or high floor).
- **ReservationStatus**: l Reservation last status, assuming one of
three categories:
BO
Canceled – booking was canceled by
the customer;
Check-Out – customer has checked in
but already departed;
No-Show – customer did not check-in
and did inform the hotel of the reason
why.
- **ReservationStatusDate**: Date at which the last status was set.
This variable can be used in conjunction
with the ReservationStatus to understand
when was the booking canceled or when
did the customer checked-out of the
hotel.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

from google.colab import drive
drive.mount('/content/drive')

df1 = pd.read_csv('/content/drive/MyDrive/BOOTCAMP/advanced machine learning dicoding/H1.csv')
df2 = pd.read_csv('/content/drive/MyDrive/BOOTCAMP/advanced machine learning dicoding/H2.csv')

"""We add column **IsResort** in both dataset. For dataset 1, the value is 1 and the other is 0. Then we combine those dataset."""

#add IsResort Hotel column
df1['IsResort'] = 1
df2['IsResort'] = 0
#combine df1 drom hotel 1 and df2 from hotel 2
df = pd.concat([df1, df2])

df

df.info()

"""## **Check Missing Values**"""

df.isnull().sum()

"""There are some missing values in **Country** and **Children** features, so we will drop it."""

df.dropna(inplace=True)

"""## Feature Engineering

As we can saw, data type of **Children** should be int64 and data type of **ReservationStatusDate** should be datetime.
"""

df['Children'] = df['Children'].astype('int')
df['ReservationStatusDate'] = pd.to_datetime(df['ReservationStatusDate']).astype('datetime64[ns]')

"""We can make another column:
- **Arrival**: datetime of customer's arrival (YYYY-MM-DD)
"""

df["Arrival"] = df["ArrivalDateYear"].astype(str) + "-" + df["ArrivalDateMonth"].astype(str) + "-" + df["ArrivalDateDayOfMonth"].astype(str)
df["Arrival"] = pd.to_datetime(df["Arrival"])

df.drop(columns=["ArrivalDateYear", "ArrivalDateMonth", "ArrivalDateDayOfMonth"], inplace=True)

"""## **Exploratory Data Analysis**

### **Univariate Analysis**
"""

fig, ax = plt.subplots(1,2,figsize=(15,5))
palette = sns.color_palette("crest", n_colors=2)
sns.countplot(ax=ax[0], data=df, x="IsCanceled", hue="IsCanceled", palette=palette)
ax[1].pie(df["IsCanceled"].value_counts(), labels=["Not Canceled", "Canceled"], autopct="%.02f%%", colors=palette)
plt.show()

"""From pie chart above, we saw >60% of dataset is not canceled by customer."""

fig, ax = plt.subplots(3,3,figsize=(15,12))
sns.countplot(ax=ax[0][0], data=df, x="Adults", color=palette[0], stat="percent")
sns.countplot(ax=ax[0][1], data=df, x="Children", color=palette[0], stat="percent", width=.3)
sns.countplot(ax=ax[0][2], data=df, x="Babies", color=palette[0], stat="percent", width=.3)

sns.histplot(ax=ax[2][0], data=df, x="CustomerType", color=palette[0], stat="percent")
ax[2][0].set_xticklabels(labels=df["CustomerType"].unique(), rotation=45)
sns.histplot(ax=ax[2][1], data=df, x="MarketSegment", color=palette[0], stat="percent")
ax[2][1].set_xticklabels(labels=df["MarketSegment"].unique(), rotation=45)
sns.histplot(ax=ax[2][2], data=df, x="DistributionChannel", color=palette[0], stat="percent")
ax[2][2].set_xticklabels(labels=df["DistributionChannel"].unique(), rotation=45)

sns.histplot(ax=ax[1][0], data=df, x="DepositType", color=palette[0], stat="percent")
sns.histplot(ax=ax[1][1], data=df, x="Meal", color=palette[0], stat="percent")
sns.countplot(ax=ax[1][2], data=df, x="IsResort", color=palette[0], stat="percent")
plt.show()

"""Observations:
- The majority of customers is two adult and percentage of underage person is minimum, with more than 75% is transient customer (not a part of group, contract or other transient party).
- More than 80% Customer's deposit type is no deposit.
- More than 60% of dataset is an customer of city hotel.
- The majority of booking distribution is delivered by travel agent/tour operators and market segmentation is also delivered by online travel agent.

### **Multivariate Analysis**
"""

fig, ax = plt.subplots(2,2,figsize=(15,12))
palette = sns.color_palette("crest", n_colors=2)
sns.histplot(ax=ax[0][0], data=df, x="CustomerType", palette={0:palette[0], 1:palette[1]},
             stat="percent", hue='IsCanceled', hue_order=[1,0], multiple="stack")
ax[1][0].set_xticklabels(labels=df["DistributionChannel"].unique(), rotation=45)
sns.histplot(ax=ax[0][1], data=df, x="DepositType", palette={0:palette[0], 1:palette[1]},
             stat="percent", hue='IsCanceled', hue_order=[1,0], multiple="stack")

sns.histplot(ax=ax[1][0], data=df, x="DistributionChannel", palette={0:palette[0], 1:palette[1]},
             stat="percent", hue='IsCanceled', hue_order=[1,0], multiple="stack")
sns.histplot(ax=ax[1][1], data=df, x="MarketSegment", palette={0:palette[0], 1:palette[1]},
             stat="percent", hue='IsCanceled', hue_order=[1,0], multiple="stack")
ax[1][1].set_xticklabels(labels=df["MarketSegment"].unique(), rotation=45)

plt.show()

"""Observations:
- The most canceled booking is by transient customer which is has the most number of booked.
- The most booking channel and market segment of customer is travel agent with the most canceled booking.
- Groups market segment has most canceled booking based on proportion.
- All of Non-refund deposit type is canceled booking.

"""

#create dataframe of canceled booking
book_canceled = df[df['IsCanceled']==1][['Arrival']]
book_canceled['Number'] = 1
book_canceled = book_canceled.set_index('Arrival').resample('M').sum()
#create dataframe of not canceled booking
book_not_canceled = df[df['IsCanceled']==0][['Arrival']]
book_not_canceled['Number'] = 1
book_not_canceled = book_not_canceled.set_index('Arrival').resample('M').sum()

plt.figure(figsize=(15,5))
sns.lineplot(data=book_canceled,x="Arrival",  y="Number", color=palette[1])
sns.lineplot(data=book_not_canceled, x="Arrival", y="Number", color=palette[0])
plt.legend(["Canceled", "Not Canceled"])
plt.title("Number of Bookings Based on Arrival Date")
plt.show()

"""Observations:
- Both canceled and non-canceled bookings increase during certain periods (e.g., mid-2016, late 2016, and mid-2017).These peaks likely correspond to high-travel seasons (e.g., holidays or vacation periods).
- The number of cancellations generally follows the pattern of total bookings.
Higher booking periods also have higher cancellations, which suggests that during peak seasons, people make more reservations but also cancel more frequently.
- Some months show spikes in cancellations (e.g., early 2016, late 2016, and mid-2017), which may indicate factors such as policy changes, pricing adjustments, or market disruptions.
"""

fig, ax = plt.subplots(1,2,figsize=(10,5))
sns.histplot(ax=ax[0],data=df, x="PreviousCancellations", hue="IsCanceled", palette=palette, stat="percent", multiple="stack")
sns.histplot(ax=ax[1],data=df, x="PreviousBookingsNotCanceled", hue="IsCanceled", palette=palette, stat="percent", multiple="stack")
plt.show()

"""Observation:
- The current status is not influenced by status of previous booking (canceled or not canceled)
"""

fig, ax = plt.subplots(1,2,figsize=(10,5), sharey=True)
sns.countplot(ax=ax[0], data=df, x="AssignedRoomType", hue="IsCanceled", palette=palette)
sns.countplot(ax=ax[1], data=df, x="ReservedRoomType", hue="IsCanceled", palette=palette)

"""Observations:
- The majority of room type booked by customer is D. It also has a high number of cancellations, indicating that this room type might be more prone to cancellations.
- Some room types appear more in the assigned than the reserved category (e.g., 'E' and 'G'). A potential cause for cancellations could be room assignment mismatches—customers may cancel if they do not get the room they originally reserved.
"""

fig, ax = plt.subplots(1,2,figsize=(10,5), sharey=True)
palette = sns.color_palette("crest", n_colors=2)
sns.countplot(ax=ax[0], data=df, x="RequiredCarParkingSpaces", hue="IsCanceled", palette=palette)
sns.countplot(ax=ax[1], data=df, x="TotalOfSpecialRequests", hue="IsCanceled", palette=palette)

"""Observations:
- Hotel with 0 car parking space most likely canceled by customers.
- Hotel with 0 special requests (e.g. twin bed or high floor) most likely canceled by customers.

# **DATA PREPARATION**

## **Feature Encoding**
"""

#encoding and one-hot encoding
df.info()

df2 = df.copy()
df2["Country"] = LabelEncoder().fit_transform(df2["Country"])
df2["MarketSegment"] = LabelEncoder().fit_transform(df2["MarketSegment"])
df2["DistributionChannel"] = LabelEncoder().fit_transform(df2["DistributionChannel"])
df2["ReservedRoomType"] = LabelEncoder().fit_transform(df2["ReservedRoomType"])
df2["CustomerType"] = LabelEncoder().fit_transform(df2["CustomerType"])
df2["DepositType"] = LabelEncoder().fit_transform(df2["DepositType"])
df2["Agent"] = LabelEncoder().fit_transform(df2["Agent"])
df2["Company"] =LabelEncoder().fit_transform(df2["Company"])
df2["ReservationStatus"] = LabelEncoder().fit_transform(df2["ReservationStatus"])

"""[Antonio et al, (2019)](http://dx.doi.org/10.18089/tms.2017.13203) shows distribution of **Meal, AssignedRoomType**, and **Country** is different between canceled and not-canceled bookings. We make one-hot encoding based on **Meal** and **AssignedRoomType** categorical features because the large amount of variables of **Country**."""

df1 = df2.copy()  # Create a copy outside the loop

for col in ["Meal", "AssignedRoomType"]:
    one_hot_encoded = pd.get_dummies(df2[[col]], prefix=col, drop_first=False)

    # Convert boolean columns in one_hot_encoded to int64
    for column in one_hot_encoded.select_dtypes(include=['bool']).columns:
        one_hot_encoded[column] = one_hot_encoded[column].astype(np.int64)

    df1 = pd.merge(df1, one_hot_encoded, left_index=True, right_index=True, how='left')
    df1 = df1.drop(columns=[col])

df1.info()

"""## **Feature Selection**

We use top 20 based on Pearson correlation.
"""

corr_matrix = df1.corr(numeric_only=True)
top_features = corr_matrix['IsCanceled'].abs().sort_values(ascending=False)[:20]
data_top = df1[top_features.index.to_list()]
corr_top_20 = corr_matrix.loc[top_features.index, top_features.index] #for heatmap display

palette = sns.color_palette("crest", n_colors=5)

top_features

plt.figure(figsize=(25,15))
sns.heatmap(corr_top_20, cmap=palette, annot=True)
plt.show()

"""Drop features with correlation >0.7. Between **MarketSegment** and **DistributionChannel**, we choose **DistributionChannel** it has more correlation with **IsCanceled** feature."""

final = data_top.drop(columns=["MarketSegment"])

final.info()

"""## **Data Splitting**

We choose 70:30 for training and testing set. Train set is used for train the model and test set is used for model evaluation.
"""

X = final.drop(columns=['IsCanceled'])
y = final['IsCanceled']

#split data 70:30
X_train, X_test, y_train, y_tst = train_test_split(X,y, test_size=0.3, random_state=42)

"""## **Standarization**"""

X_train.describe()

"""We use Standard Scaler for feature Standarization because it's"""

scaler = StandardScaler()
num_col = X_train.select_dtypes(include=['float64', 'int64']).columns.tolist()
X_train[num_col] = scaler.fit_transform(X_train[num_col])
X_test[num_col] = scaler.fit_transform(X_test[num_col])

"""# **DATA MODELING**

We choose K-nearest neighbor and logistic regression algorithm ro predict hotel booking cancellation.

## **K-Nearest Neighbor**

Pros:
- Non-parametric & Flexible
- No Assumption on Data Distribution
- Can detect non-linear patterns in cancellations.

Cons:
- Slower for large datasets (distance calculations increase with more data).
- Sensitive to Feature Scaling
- Performance drops as feature count increases (Curse of Dimensionality).
"""

from sklearn.neighbors import KNeighborsClassifier
knn_start_time = time.time()
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
knn_end_time = time.time()
knn_execution_time = knn_end_time - knn_start_time
print('Logistic regression has trained')
print(f"Execution time: {knn_execution_time} seconds")

"""## **Logistic Regression**

Pros:
- Works well on large datasets with many features.
- If the data has a clear linear relationship, LR performs effectively.
- Doesn't require much feature selection.

Cons:
- Assumes Linear Relationship
- Not Effective for Complex Patterns
- Sensitive to Outliers
"""

lr_start_time = time.time()
lr = LogisticRegression()
lr.fit(X_train, y_train)
lr_end_time = time.time()
lr_execution_time = lr_end_time - lr_start_time
print('Logistic regression has trained')
print(f"Execution time: {lr_execution_time} seconds")

"""# **EVALUATION**

## **Train Set**
"""

#KNN
y_pred_knn_train = knn.predict(X_test)
acc_knn_train = round(accuracy_score(y_tst, y_pred_knn_train), 4)
pre_knn_train = round(precision_score(y_tst, y_pred_knn_train),4)
rec_knn_train = round(recall_score(y_tst, y_pred_knn_train),4)
f1_knn_train = round(f1_score(y_tst, y_pred_knn_train),4)
#Logistic Regression
y_pred_lr_train = lr.predict(X_test)
acc_lr_train = round(accuracy_score(y_tst, y_pred_knn_train), 10)
pre_lr_train = round(precision_score(y_tst, y_pred_knn_train),10)
rec_lr_train = round(recall_score(y_tst, y_pred_knn_train),10)
f1_lr_train = round(f1_score(y_tst, y_pred_knn_train),10)

"""## **Test Set**

"""

#KNN
y_pred_knn_test = knn.predict(X_test)
acc_knn_test = round(accuracy_score(y_tst, y_pred_knn_test),4)
pre_knn_test = round(precision_score(y_tst, y_pred_knn_test),4)
rec_knn_test = round(recall_score(y_tst, y_pred_knn_test),4)
f1_knn_test = round(f1_score(y_tst, y_pred_knn_test),4)
#Logistic regression
y_pred_lr_test = lr.predict(X_test)
acc_lr_test = round(accuracy_score(y_tst, y_pred_lr_test),4)
pre_lr_test = round(precision_score(y_tst, y_pred_lr_test),4)
rec_lr_test = round(recall_score(y_tst, y_pred_lr_test),4)
f1_lr_test = round(f1_score(y_tst, y_pred_lr_test),4)

"""## **Confusion Matrix**

### **KNN**
"""

cm = confusion_matrix(y_tst, y_pred_knn_test)
sns.heatmap(cm/np.sum(cm), annot=True, fmt=".2%", cmap=palette, xticklabels=["Positive", "Negative"],
            yticklabels=["Positive", "Negative"])
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.title("Confusion Matrix (K-Nearest Neighbor)")
plt.show()

"""### **Logistic Regression**"""

cm = confusion_matrix(y_tst, y_pred_lr_test)
sns.heatmap(cm/np.sum(cm), annot=True, fmt=".2%", cmap=palette, xticklabels=["Positive", "Negative"],
            yticklabels=["Positive", "Negative"])
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.title("Confusion Matrix (Logistic regression)")
plt.show()

"""Observation:
- TP for both algorithms is identical (51.90%) and TN for KNN algorithm is slightly better (48.04%) than Logistic Regression (46.75%).
- FP for Logistic Regression algorithm (0.00%) is smaller than KNN (0.001%), but the FN is bigger (1.34% versus 0.06%). It is means that Logistic Regression algorithm tend to misclassified canceled booking as not canceled booking which can be dangerous.

## **Comparison between algorithms**
"""

diff = pd.DataFrame({'Name': ['K-Nearest Neighbor','Logistic regression'],
                     'accuracy (train)': [acc_knn_train, acc_lr_train],
                     'accuracy (test)': [acc_knn_test, acc_lr_test],
                     'precision (train)': [pre_knn_train, pre_lr_train],
                     'precision (test)': [pre_knn_test, pre_lr_test],
                     'recall (train)': [rec_knn_train, rec_lr_train],
                     'recall (test)': [rec_knn_test, rec_lr_test],
                     'f1 (train)': [f1_knn_train, f1_lr_train],
                     'f1 (test)': [f1_knn_test, f1_lr_test],
                    'time (s)': [knn_execution_time, lr_execution_time]})
diff

"""Observation:
- KNN has better recall (99.88%) than Logistic Regression (97.19%), meaning it correctly identifies more high-risk cases.
- Logistic Regression has perfect precision (100%), meaning when it predicts high risk, it's always correct, but it misclassifies more actual high-risk cases as low risk.
- F1-score is slightly higher for KNN (99.93%) compared to Logistic Regression (98.57%), indicating a better balance between precision and recall.

# **CONCLUSION**

- The number of cancellations generally follows the pattern of total bookings. Some month has spike cancellation e.g. early 2016, late 2016, and mid 2017.
- The major factor of boking cancelation based on Pearson correlation is **ReservationStatus, IsResort, Deposit Type,** and **Lead Time**.
- Based on evaluation, the best model for hotel booking cancellation prediction is KNN which has better performance to prevent misclassified status of booking (canceled or not).
"""