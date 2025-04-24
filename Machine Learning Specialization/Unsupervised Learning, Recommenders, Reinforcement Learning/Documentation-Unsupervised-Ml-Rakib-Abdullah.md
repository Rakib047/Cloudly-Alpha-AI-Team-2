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








