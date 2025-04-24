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

![alt text](image-42.png)

## K means optimization objective

### Cost Funtion
- remember here,`c(i)` is cluster # and `u(i)` is the centroid's certesian point

![alt text](image-43.png)

- the algorithm is designed in way that the cost function will reduce at each step

## How to Guess Initial Centers

- Randomly choose k centers
- Try different combination
- calculate the cost function for each

![alt text](image-44.png)

## How to Choose K

### Elbow Method 

- Doesn't always work

![alt text](image-45.png)

### Choose initial centers based on for what purpose we are doing the clustering

- Like here,k center should be initialized based on how many sizes of tshirt we want -> 3 or 5?It's a manual process

![alt text](image-46.png)

## Anomaly Detection
  
- Main idea is to investigate how good examples actually look like,then anything that deviates from good examples are considered as anomaly
- Basic Idea

![alt text](image-47.png)

## Anomaly Detection using Density Estimation

![alt text](image-48.png)

## Some applications

- Manufacturing
- Fake account Detection
- Financial Fraud Detection

## Parameter Estimation in gaussian formula

- meu ->mean
- SD ->Standard Deviation

![alt text](image-50.png)

for the above figure
- Purple example (anomalous)
- green example (ok)

![alt text](image-51.png)

## Anomaly Detection Algorithm

![alt text](image-52.png)

### Algorithm Visualization

- Anomaly if not in the peaks or the hilly surface

![alt text](image-53.png)

## Developing and evaluating an anomaly detection

![alt text](image-54.png)

### How to evaluate

- the main intution here is that :
- Use cross validation set for anomaly detection + choose parameter epsilon

![alt text](image-55.png)

## Anomaly Detection vs Supervised Algorithm (When to use what)

- key idea is that if there is chance of coming new type of problems

![alt text](image-56.png)

`Examples`

![alt text](image-57.png)

## Choosing features for anomaly detection

- Try to convert non-gaussian features to gaussian features

![alt text](image-58.png)

## Error analysis for anomaly detection

![alt text](image-59.png)

- Sometimes anomaly can't be detected due to large value of p(x),in this case we add another feature and check x1 with x2(the another feature)

***An example***

- creating new features like x5 or x6,for example when unusually huge cpu load but low network traffic

![alt text](image-60.png)







