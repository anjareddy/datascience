# ##############################################
# Code 3-1
# Perceptron w/ Simple Data
# Machine Learning and Deep Learning Course
# Kent State University
# Jungyoon Kim, Ph.D.
# ##############################################
from sklearn.linear_model import Perceptron

# Trainning Set (OR)
X=[[0,0],[0,1],[1,0],[1,1]]
y=[-1,1,1,1]

# Perceptron Trainning w/ fit method
p=Perceptron()
p.fit(X,y)

print("Parameter of Trained Perceptron: ",p.coef_,p.intercept_)
print("Prediction of Trainning Set: ",p.predict(X))
print("Accuracy: ",p.score(X,y)*100,"%")

# Testing Set (XOR)
X=[[0,0],[0,1],[1,0],[1,1]]
y=[-1,1,1,-1]

print("Parameter of Trained Perceptron: ",p.coef_,p.intercept_)
print("Prediction of Trainning Set: ",p.predict(X))
print("Accuracy: ",p.score(X,y)*100,"%")