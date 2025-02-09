# **Machine Learning Project Report - Dliya Awliya Mufidah**
## **DOMAIN KNOWLEDGE OF KNOWLEDGE**
Nowadays, travelers most significantly use internet to book hotel and make an reservation ([Toh et al. 2011](https://doi.org/10.1177/1938965511418779)). Hotel bookings represent a contract between customer and the hotel. The demand of hotel bookings can influence hotel's revenue, allocation, and budgets. Hence, search for factor of bookings cancellation can prevent revenue loss and get better marketing strategy ([Antonio et al, 2019](http://dx.doi.org/10.18089/tms.2017.13203)).
## **BUSINESS UNDERSTANDING**
### **Problem Statement**
1. When is the most bookings cancellation date?
2. What is the major factor influence bookings cancellation?
3. How accurate is machine learning model can predict bookings cancellation? What is the best model?
### **Goals**
1. Identify most bookings cancellation date.
2. Analysis of bookings cancellation's influence factor.
3. Investigate accuration of machine learning models prediction of booking cancellation and choose the best model.
### **Solution statement**
1. Use correlation of arrival date and book cancellation data.
2. Explore features based on EDA, mutual information, and Pearson correlation?
3. Use 5 difference classification machine learning algorithm (Logistic Regresion, KNN, Random Forest, Gradient Boosting and XGBoost) with accuracy >95%. Algorithm selection is based on higher on accuracy and recall score.

## **DATA UNDERSTANDING**
Dataset from [Hotel Booking Demand](https://www.sciencedirect.com/science/article/pii/S2352340918315191#s0005) by Nuno Antonio, Ana de Almeida, and Luis Nunesthat collected from comprehend booking from July 1, 2015 until August 31, 2017. This dataset contains two hotel H1 and H2. H1 is an resort hotel and the other one (H2) is an city hotel.
### **Variables**
- **IsCanceled**: Value indicating if the booking was canceled (1) or not (0).
- **LeadTime**: Number of days that elapsed between the entering date of the booking into the
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

We add column **IsResort** in both dataset. For dataset 1, the value is 1 and the other is 0. Then we combine those dataset.
(gambar info)
There are 119390 rows and 32 columns contains: float64(2), int64(17), and object(13) data types.
### **Feature Engineering**
We change datetype of **Children** and make another column name **Arrival**, datetime of customer's arrival (YYYY-MM-DD) based on columns **ArrivalDateYear, ArrivalDateMonth**, and **ArrivalDateDayOfMonth**.
### **Exploratory Data Analysis**
**Univariate Analysis**
<p align="center">
  <img width="460" height="300" src="https://picsum.photos/460/300">
</p>

## **DATA PREPARATION**
### **Feature Encoding and One-hot Encoding**
Feature Encoding is an process to convert categorical variables into numerical variables for machine learning modeling. Therefore, one-hot encoding is an process to ...
[Antonio et al, (2019)](http://dx.doi.org/10.18089/tms.2017.13203) shows distribution of **Meal, AssignedRoomType**, and **Country** is different between canceled and not-canceled bookings. We make one-hot encoding based on **Meal** and **AssignedRoomType** categorical features because the large amount of variables of **Country**. Also, we make **MarketSegment, DistributionChannel, ReservedRoomType, CustomerType, DepositType, Agent, Company, Country, and ReservationStatus** feature encoded
### **Feature Selection**
We select top 20 features based on Pearson correlation value.
(gambar pearson corr)
We also visualize heatmap based on Pearson correlation.
(gambar heatmap)
Drop features with correlation >0.7. Between **MarketSegment** and **DistributionChannel**, we choose **DistributionChannel** it has more correlation with **IsCanceled** feature.
### **Data Splitting**
We choose 70:30 for training and testing set. Train set is used for train the model and test set is used for model evaluation.
### **Standarization**
We use Standard Scaler because models that rely on distance calculations (KNN, K-Means, SVM, PCA) can be biased if features have different magnitudes. StandardScaler subtracts the mean and scales to unit variance. This helps gradient-based models (Logistic Regression, Neural Networks) converge faster.

## **MODELING**
We choose K-nearest neighbor and logistic regression algorithm ro predict hotel booking cancellation.

**K-Nearest Neighbor**

Pros:
- Non-parametric & Flexible
- No Assumption on Data Distribution
- Can detect non-linear patterns in cancellations.

Cons:
- Slower for large datasets (distance calculations increase with more data).
- Sensitive to Feature Scaling
- Performance drops as feature count increases (Curse of Dimensionality).

**Logistic Regression**

Pros:
- Works well on large datasets with many features.
- If the data has a clear linear relationship, LR performs effectively.
- Doesn't require much feature selection.

Cons:
- Assumes Linear Relationship
- Not Effective for Complex Patterns
- Sensitive to Outliers

## **EVALUATION**
We use confusion matrix to evaluate performance of classification model. Confusion matrix compares predicted value and actual values. The components of confusion matrix is: 
- True Positive (TP): when model correctly predicted positive class.
- True Negative (TN): when model correctly predicted  negative class.
- False Positive (FP): when model misclassify positive class as negative class.
- False Negative (FN): when model misclassify negative class as positive class.

Based on those components of confusion matrix, we can calculate performance score of models:
- Accuracy: measures the overall proportion of correct values predicted by model.
- Recall: measures the ability of model to correctly predicted positive class.
- Precision: measures proportion of positive prediction that are actually correct.
- F1-score: metric to balance both recall and precision score.

(cm of knn)

(cm of lr)

- TP for both algorithms is identical (51.90%) and TN for KNN algorithm is slightly better (48.04%) than Logistic Regression (46.75%).
- FP for Logistic Regression algorithm (0.00%) is smaller than KNN (0.001%), but the FN is bigger (1.34% versus 0.06%). It is means that Logistic Regression algorithm tend to misclassified canceled booking as not canceled booking which can be dangerous.

Comparison between KNN and Logistic Regression performance.

(table)

- KNN has better recall (99.88%) than Logistic Regression (97.19%), meaning it correctly identifies more high-risk cases.
- Logistic Regression has perfect precision (100%), meaning when it predicts high risk, it's always correct, but it misclassifies more actual high-risk cases as low risk.
- F1-score is slightly higher for KNN (99.93%) compared to Logistic Regression (98.57%), indicating a better balance between precision and recall.

## **CONCLUSION**
- The number of cancellations generally follows the pattern of total bookings. Some month has spike cancellation e.g. early 2016, late 2016, and mid 2017.
- The major factor of boking cancelation based on Pearson correlation is Reservation Status, IsResort, Deposit Type, and Lead Time.
- Based on evaluation, the best model for hotel booking cancellation prediction is KNN which has better performance to prevent misclassified status of booking (canceled or not).

