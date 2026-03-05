# DEMO PRESENTATION SCRIPT
**Project:** Automobile Insurance Fraud Detection System
**Estimated Time:** 6 Minutes

---

## 1. Introduction (Viraj - 1.5 mins)
**Viraj:** "Good morning everyone. We are excited to present our Machine Learning project: 'Automobile Insurance Fraud Detection System.' Our team consists of myself (Viraj), Samruddhi, Snehal, and Tanuja. 

Let's start with the problem. Insurance fraud costs companies and honest customers billions of dollars every year. A fake claim might look perfectly normal to a human reviewer. Our objective was to build a system that acts as an intelligent assistant for insurance agents. Instead of manually reviewing hundreds of columns of data, our system predicts instantly if an automobile claim is 'Fraud' or 'Genuine' using historical data patterns. 

For the data, we used a dataset with standard insurance features like the customer's age, how many months they've been with the company, their policy premium, and details about the incident itself, such as the total claim amount and whether there were any witnesses."

---

## 2. Data Preprocessing & EDA (Samruddhi - 1.5 mins)
**Samruddhi:** "Thank you, Viraj. I'll cover how we prepared the data. Raw data is rarely ready for Machine Learning, so we started with Data Preprocessing. We handled any missing values and scaled down monetary features—like 'total claim amount'—so that our algorithms wouldn't be biased towards larger numbers. 

We also performed Exploratory Data Analysis, or EDA, to understand the data before feeding it to our models. A key finding from our visualization was that major damages with zero witnesses and very high claim ratios are heavily correlated with fraudulent activity. By visualizing these patterns using Seaborn and Matplotlib heatmaps and plot boxes, we could select the top 10 most influential features to simplify our model and prevent overfitting."

---

## 3. Model Training & Comparison (Snehal - 1.5 mins)
**Snehal:** "Thanks, Samruddhi. Next was the core of our project—Model Building. We wanted to see which traditional Machine Learning algorithm performed the best on our structured tabular data. We trained six different estimators: Logistic Regression, Decision Tree, Random Forest, K-Nearest Neighbors, Naïve Bayes, and Support Vector Machines.

We evaluated them not just on Accuracy, but more importantly on 'Recall.' In fraud detection, a False Negative—which means predicting a fraud claim as genuine—is extremely costly for the company. Random Forest emerged as our champion model, giving us the highest F1 Score and Recall at over 90%. We then exported this optimized model as a pickle file to be used in our web application."

---

## 4. Web Deployment & Conclusion (Tanuja - 1.5 mins)
**Tanuja:** "Thank you, Snehal. To make our model usable in the real world, we deployed it using the Flask web framework. 

(Demonstrate the live app running at localhost:5000 if showing the screen)
As you can see, we built a clean and responsive HTML front-end where an agent can input the details of a new claim—such as age, annual premium, and claim amounts. Once submitted, the Flask backend processes the inputs, applies the same scaling as our training data, and queries our Random Forest model. It instantly replies with either a 'FRAUDULENT' alert in red or a 'GENUINE' confirmation in green.

In conclusion, we successfully built an end-to-end Machine Learning pipeline that automates a complex financial task, providing a practical, fast, and medium-level AI solution to insurance fraud. 

Thank you for listening. We are now open to any questions!"
