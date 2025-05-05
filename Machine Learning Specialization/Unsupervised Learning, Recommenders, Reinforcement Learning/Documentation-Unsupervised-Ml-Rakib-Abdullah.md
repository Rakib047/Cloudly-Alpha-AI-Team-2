# Unsupervised Learning

# Week 1

## Clustering

- No output label
- group data into clusters

## K means Clustering Algorithm

**Step 1** : Select k centers randomly
**Step 2** : Iterate through each point and assign them to the closest center
**Step 3** : Reassign center by calculating the average of the points
**Step 4** : Repeat from step 2 until no changes

<img width="574" alt="image" src="https://github.com/user-attachments/assets/8bbbe1e2-f86e-4417-a1e3-334cdc9516cc" />


## K means optimization objective

### Cost Funtion
- remember here,`c(i)` is cluster # and `u(i)` is the centroid's certesian point

<img width="566" alt="image" src="https://github.com/user-attachments/assets/7a4b831d-d423-430b-ac6d-68f51a342b5a" />


- the algorithm is designed in way that the cost function will reduce at each step

## How to Guess Initial Centers

- Randomly choose k centers
- Try different combination
- calculate the cost function for each

<img width="566" alt="image" src="https://github.com/user-attachments/assets/6338197e-1e83-455b-8c98-2d15b191f3dc" />


## How to Choose K

### Elbow Method 

- Doesn't always work

<img width="566" alt="image" src="https://github.com/user-attachments/assets/b435eeef-1e21-401d-8d39-048748785c1b" />


### Choose initial centers based on for what purpose we are doing the clustering

- Like here,k center should be initialized based on how many sizes of tshirt we want -> 3 or 5?It's a manual process

<img width="566" alt="image" src="https://github.com/user-attachments/assets/a3433d92-513c-4d0b-a853-e327775fcf96" />


## Anomaly Detection
  
- Main idea is to investigate how good examples actually look like,then anything that deviates from good examples are considered as anomaly
- Basic Idea

<img width="566" alt="image" src="https://github.com/user-attachments/assets/26d18c3e-65ca-41c1-bdb2-0948242a1742" />


## Anomaly Detection using Density Estimation

<img width="568" alt="image" src="https://github.com/user-attachments/assets/f379a4e8-9b68-44fa-8acd-6bc8e6055440" />


## Some applications

- Manufacturing
- Fake account Detection
- Financial Fraud Detection

## Parameter Estimation in gaussian formula

- meu ->mean
- SD ->Standard Deviation

<img width="566" alt="image" src="https://github.com/user-attachments/assets/e744a64c-713f-4216-b9bc-5a96fd794fe1" />


for the above figure
- Purple example (anomalous)
- green example (ok)

<img width="566" alt="image" src="https://github.com/user-attachments/assets/42487bb2-67e8-4c63-8740-bf32ca8416cd" />


## Anomaly Detection Algorithm

<img width="566" alt="image" src="https://github.com/user-attachments/assets/8575f5a1-540f-40e1-b627-194631bcaf62" />


### Algorithm Visualization

- Anomaly if not in the peaks or the hilly surface

<img width="566" alt="image" src="https://github.com/user-attachments/assets/4d830ac5-3427-426f-86e9-8f3fcfdf655e" />


## Developing and evaluating an anomaly detection

<img width="566" alt="image" src="https://github.com/user-attachments/assets/71264d3c-7349-4bc1-abe6-202fd2c21d92" />


### How to evaluate

- the main intution here is that :
- Use cross validation set for anomaly detection + choose parameter epsilon

<img width="566" alt="image" src="https://github.com/user-attachments/assets/732fdfd0-b3bd-42e7-865a-fc2dc40deff1" />


## Anomaly Detection vs Supervised Algorithm (When to use what)

- key idea is that if there is chance of coming new type of problems

<img width="566" alt="image" src="https://github.com/user-attachments/assets/3a20646c-40b4-408c-901f-1ee3da6b05e5" />



`Examples`

<img width="566" alt="image" src="https://github.com/user-attachments/assets/87133020-aa7d-42df-9d46-9b78a4bb8c8d" />


## Choosing features for anomaly detection

- Try to convert non-gaussian features to gaussian features

<img width="573" alt="image" src="https://github.com/user-attachments/assets/16ea996b-11b1-406f-a091-c2c86cca856d" />


## Error analysis for anomaly detection

<img width="566" alt="image" src="https://github.com/user-attachments/assets/fdfe2fbd-ee7a-4503-9b9a-b5ae97d98693" />


- Sometimes anomaly can't be detected due to large value of p(x),in this case we add another feature and check x1 with x2(the another feature)

***An example***

- creating new features like x5 or x6,for example when unusually huge cpu load but low network traffic

<img width="566" alt="image" src="https://github.com/user-attachments/assets/859bb3ed-8f36-4369-adbd-7838c0df62b1" />

---

# Week 2
## Collaborative Filtering Algorithm

- mutiple users have given rating to movies,from that we can derive what features those movies have

### An example Recommender System (User rating prediction for movies)
<img width="645" alt="image" src="https://github.com/user-attachments/assets/1e3b8c40-2d4e-4224-bd0e-b75a4cd4bb2a" />


- Adapting the model with linear regression model for each of the user.Basically here for each user there is a linear regression model we are using

- fetures her are romance and action,using these features we are predicting movie rating

<img width="645" alt="image" src="https://github.com/user-attachments/assets/3b5f6f8d-a517-402f-b2ee-303449132dcf" />


***notations***

<img width="645" alt="image" src="https://github.com/user-attachments/assets/eec6d2e8-9f48-4713-9610-4d71a1e1f125" />


- our goal and cost function,the below cost funtion is for one user

<img width="645" alt="image" src="https://github.com/user-attachments/assets/1f4db3e6-0e17-45be-8971-36837d50464c" />


- after little modification and also for all user,the cost function of individual and for all users : 

<img width="645" alt="image" src="https://github.com/user-attachments/assets/ca5598bc-b3a3-46e7-9bc5-e64229458d1b" />



## Guessing the features value if parameters are given(we assume)

<img width="645" alt="image" src="https://github.com/user-attachments/assets/5dacd2a0-7225-4130-ac50-3e893dad6376" />


- Now if we combine them:

<img width="649" alt="image" src="https://github.com/user-attachments/assets/c2b2b4a2-7298-4bd3-bc13-05759f7125ec" />


### Gradient descent for this cost function

<img width="651" alt="image" src="https://github.com/user-attachments/assets/d8fc9775-8100-4aab-a889-3155186287f2" />


## Binary Classification using collaborative filtering

### Example

<img width="645" alt="image" src="https://github.com/user-attachments/assets/a4acf1e1-ae50-4f80-bcb8-2b4485e2fd2e" />


<img width="645" alt="image" src="https://github.com/user-attachments/assets/b941d77c-2e1b-4aae-9980-d381f9cf3696" />


## from regression to binary classification

<img width="646" alt="image" src="https://github.com/user-attachments/assets/e9889265-145b-440d-8a57-bf11117c7e18" />


### Cost funtion for binary classification using collaborative filtering

<img width="653" alt="image" src="https://github.com/user-attachments/assets/b14a5aa5-51bf-4250-a89d-e950e47fc5e3" />


## Recommender Systems Implementation

### Mean Normalization

<img width="645" alt="image" src="https://github.com/user-attachments/assets/4a278889-070f-43f5-84b8-b3fb0dc2958d" />


<img width="645" alt="image" src="https://github.com/user-attachments/assets/86f94cd4-387e-4210-bbff-2e08548e2fb0" />


## How we want related items in websites

<img width="645" alt="image" src="https://github.com/user-attachments/assets/ec4c5c7c-cccc-497a-8621-5348da8302fe" />


## Limitations of Collaborative filtering

<img width="645" alt="image" src="https://github.com/user-attachments/assets/47d2db3b-8d03-4e29-95bc-a5d2acd8065f" />

## Content based filtering

- main idea is to match user features and content features

### Content based vs collaborative filtering

![image](https://github.com/user-attachments/assets/93dfbe3d-b5f9-4329-961d-9188bd1cfc22)


## Example of features in content based filtering

![image](https://github.com/user-attachments/assets/c258ffb0-e3ba-42cb-a056-69483d5a20c9)


### How content based filtering works

![image](https://github.com/user-attachments/assets/97ff6c9b-d0f8-4bae-9fd5-dd605e6010e2)


## Deep learning for content based filtering

- we train the network in such a way that we get v_u and v_m values that will reduce the cost function value

![image](https://github.com/user-attachments/assets/12286c7a-7b1e-4a19-92d9-47d1989e4019)


**Try to minimize || V_m^k - V_m^(i) || as much as possible**

![image](https://github.com/user-attachments/assets/ff275d28-ce6f-474e-b58c-34cf7225f26b)


## Recommending from large catalogue

- It is done in two steps -> `Retrieval` and `Ranking`

![image](https://github.com/user-attachments/assets/eb0bdf8d-f42e-4ed8-bb6e-005ad4254434)


![image](https://github.com/user-attachments/assets/e446bdcb-0745-416c-b064-fcec952b13e2)

## TensorFlow implementation of content-based filtering

![image](https://github.com/user-attachments/assets/71cb6231-5f86-4556-a837-65a0d59d83dc)

## Some other Unsupervised Learning Algorithm 

## PCA (Principal Component Analysis)

### Why to use PCA

- Some features are actually not needed

<img width="706" alt="image" src="https://github.com/user-attachments/assets/592f5ad4-2e27-472a-92b4-20e0304c8468" />


- The goal of PCA is to find one or more new axis

<img width="706" alt="image" src="https://github.com/user-attachments/assets/6437983f-deed-49bb-bbb1-d88eabdfbc12" />


<img width="706" alt="image" src="https://github.com/user-attachments/assets/b718c955-23cf-439a-86bb-0e8ce17a30ce" />


### An Example

<img width="706" alt="image" src="https://github.com/user-attachments/assets/f5201bd1-953b-4c7c-a868-f71468d0ceed" />


## The PCA algorithm

- Normalized the features to zero mean
- Scale the features values so that some are not too big or too small

- Choose an axis,axis choice is important because it defines how we are going to capture the nature of the data

<img width="709" alt="image" src="https://github.com/user-attachments/assets/a2b3a469-b383-4aed-84eb-fca33d7d650d" />


<img width="707" alt="image" src="https://github.com/user-attachments/assets/b930a23a-6561-446e-aa0a-7bac5f7b6a9b" />


**The axis with highest variance is the principal axis**

<img width="707" alt="image" src="https://github.com/user-attachments/assets/74bcd52e-1ef4-4299-9802-7e923beacbd5" />


- More than one principal component

<img width="707" alt="image" src="https://github.com/user-attachments/assets/0655610e-1b04-4ba1-9c84-7862766f67f9" />


### We can also reconstruct the data from PCA

<img width="707" alt="image" src="https://github.com/user-attachments/assets/0bd5a6dc-d7fd-40de-b828-607e91b41019" />


## PCA in code

### Steps

<img width="707" alt="image" src="https://github.com/user-attachments/assets/1d313f00-1937-4bc7-8bbf-4b79058c129b" />


- The code

<img width="732" alt="image" src="https://github.com/user-attachments/assets/3f6fbb10-6cf5-47f0-9bb2-eb73cd2cd5a9" />


## Application of PCA

<img width="698" alt="image" src="https://github.com/user-attachments/assets/6a668bce-90d4-4269-979b-ee7d37f03744" />

# Week 3

## Reinforcement Learning

- State,Reward function,action (S,a,R(s),S')
- When it is hard to give x and y mapping in supervised learning,we use reinforcement learning(just give a reward function)
- how much reward we get in future is decided by discount factor

- policy pi function -> take a state and output an action

![alt text](image-99.png)

## Summary

![alt text](image-100.png)

**The reinforcement learning process is known as Markov Decison Process**

![alt text](image-101.png)

## How to pick actions

![alt text](image-103.png)

**Discount factor affects the action of agent at each state**   

### Bellman Equation

![alt text](image-104.png)

- An example
- remember for terminal State Q(S,a)=R(s)

![alt text](image-105.png)

### The core meaning of bellman equation

![alt text](image-106.png)

**Example of this**

![alt text](image-107.png)

## Stochastic Reinforcement Learning

- next state can be random

- Instead of maximizing the return,rather we focus on maximizing the expected(average) return -> choose a policy according to this

![alt text](image-108.png)

**So modified bellman equation for stochastic environment**

![alt text](image-109.png)

## Continuos State

- state value can take continuos values,not just fixed certain numbers

- here in the below example represented as a vector ->the value of the state

![alt text](image-110.png)

## Learning the state-value function(Q(s,a)) using deep neural network

![alt text](image-111.png)

### But we need training example,how do we make it?

![alt text](image-113.png)

**Now the main algorithm**

![alt text](image-114.png)

## Improved Architecture

- Previously we have to compute for all four of the Q(s,nothing),Q(s,left),Q(s,main),Q(s,right)

![alt text](image-115.png)

- improved network is below 

![alt text](image-116.png)

## How to take action while we still learning

![alt text](image-117.png)

![alt text](image-118.png)
