---
title: Clustering, K-Means, Mixture Models, Expectation Maximization
date: 2017-03-05
math: true
draft: true
---
Clustering is an unsupervised form of learning. We have data but no
labels and we want to infer some sort of labels from the data alone. 
So what does clustering do? Say we have a bunch of data points $ x $
that look like below.

{{< image src="/img/traditional/clustering/clusterable.png" class="center-image" width="400px" >}}

Clearly, the green, red, and blue data points represent three different
clusters. The goal of clustering is to be able to determine these clusters
meaning figure out which data point belongs to which cluster. 

Clustering is an important concept because it is an unsupervised method that
can provide labels to an otherwise label-less dataset. Group similar data points
using clustering. Say your $ x $ values are customer data
points you can then group customers into similar groups using clustering. 

Intuitively we want to group data points that are close together in a single
cluster. The first and simplest clustering algorithm that leverages this
property is the k-means algorithm. 

## K-Means

Say that the center of cluster $ k $ is defined as $ \mu_k $. Sometimes $ \mu_k
$ is called the prototype vector of cluster $ k $. Intuitively, data points
that belong to cluster $ k $ should be closest to the center of cluster $ k $, $ \mu_k $ relative to all the other cluster centers.

Let's start with $ r_{nk} $ being 1 if point $x_n$ belongs to cluster 
$ k $ or 0 otherwise. Like all other things in machine learning we need a cost function that we can
minimize. Define the cost as the distance between each point and it's
assigned cluster. We can use $ r_{nk} $ to write this cleanly.

$$ J = \sum_n \sum_{k} r_{nk} \lVert x_n - \mu_k \rVert_2^2 $$

First, let's minimize this cost function with respect to $ r_{nk} $ to find
the optimal cluster assignments. Fix $ \mu_k $ and take the partial
derivative of $ J $ with respect to $ r_{nk} $ and set it equal to zero
to find the optimal cluster assignments $ r_{nk}^{*} $. We find that:

$$ 
r_{nk}^{*} = \begin{cases} 1 & \text{if $k = \text{argmin}_j \lVert x_n - \mu_j \rVert_2^2$} \\\\ 0 & \text{otherwise} \end{cases} $$

$\text{argmin}_x f(x)$ returns the value of $ x $ that minimizes $ f(x)
$. The optimal cluster assignment for $ x_n $ is simply the closest cluster prototype vector to
point $ x_n $.

Next let's do that same process of taking the derivative of the cost function with
respect to the cluster centers $ \mu_k $ to find the optimal cluster centers
$ \mu_k^{*} $. Doing so gives:

$$
\mu_k = \frac{\sum_n r_{nk} x_n}{\sum_n r_{nk}}
$$

This should also be pretty intuitive. The best center of the cluster is just
the average of the points in that cluster.

The k-means process is listed below:

- Initialize $ \mu_k $ to a random points in the dataset. 
- Update the cluster assignments through the equation: 
$$ 
r_{nk}^{*} = \begin{cases} 1 & \text{if $k = \text{argmin}_j \lVert x_n -
\mu_j \rVert_2^2$}  \\\\ 0 & \text{otherwise} \end{cases} 
$$
- Update the cluster means through the equation:
$$
\mu_k = \frac{\sum_n r_{nk} x_n}{\sum_n r_{nk}}
$$
- Compute the cost $ J $. If the cost has not changed much since last iteration stop. 
- Repeat

Turns out it can be difficult to reach a global maximum of the cost function
$ J $ through this process. Therefore, the common practice is to use multiple
random initializations and see which gives the best lowest $ J $ in the end
and use that clustering.

Overall, this is a pretty straight forward algorithm. However, things get more
complicated if you look at the example of clusters below.

{{< image src="/img/traditional/clustering/gmm_example.jpg" class="center-image" width="400px" >}}

In the above picture the data points do not necessarily stay close to their
cluster centers but it is easy to see there are three distinct clusters. Making
the statement about data points should stay close to center clusters was just a
stipulation on how we think the data should be distributed about a cluster. It is possible
generated data does not obey this rule. Like in the
above case the data appears normally distributed around the cluster center. We
can then model each cluster not with the points closest to it but with a
generalized probability distribution. 

## Gaussian Mixture Models

Think of a Gaussian mixture model (GMM) as a generalized form of k-means (we
will prove that it is later). GMM assumes that the data is generated from $ k
$ distinct Gaussian distributions. 

{{< image src="/img/traditional/clustering/2dmixture.png" class="center-image" width="400px" >}}

The above image shows data that has been generated from a combination of two
normal distributions. We can write the final distribution for a generalized GMM
as the following.

$$
p(x) = \sum_k \omega_k N(x \lvert \mu_k, \Sigma_k)
$$

$$
\sum_k \omega_k = 1, \omega_k > 0 
$$

Note that this is for the multinomial distribution, a generalized version of
the normal distribution for multiple dimensions. We therefore, write the
covariance matrix $ \Sigma_k $ and the mean vector $ \mu_k $ as parameters
of the model. Don't let this confuse you, think of $ \Sigma_k $ as the
variance and $ \mu_k $ as the mean. The stipulation on $ \omega_k $ is just
to ensure the combined probability distribution is a valid distribution. 

With k-means we leveraged knowledge about how data in a cluster should
be distributed. We were utilizing data about not just $ p(x) $ but 
also the additional random variable of $ z $, the cluster index that we were
modeling. So we were actually using marginal distribution of the joint distribution $ p(x, z) $
to model $ p(x) $ in k-means. We can rewrite the joint distribution using Bayes rule.

$$
p(x,z) = p(z)p(x \lvert z)
$$

In our case of GMM we have. 

$$
p(z=k) = \omega_k
$$

$$
p(x \lvert z=k) = N(x \lvert \mu_k, \Sigma_k)
$$

Therefore, we have:

$$
p(x,z=k) = p(z=k)p(x \lvert z=k) = \omega_k N(x \lvert \mu_k, \Sigma_k)
$$

Then finding the marginal distribution of the data

$$
p(x) = \sum_k p(x,z=k) = \sum_k \omega_k N(x \lvert \mu_k, \Sigma_k)
$$

Which is what we had before.

Now how we do we go about learning the parameters of each Gaussian
distribution $ \Sigma_k, \mu_k $ We want to find $ \theta $ (where $ \theta = (\Sigma_k, \mu_k, \omega_k )$) that maximizes the conditional probability $ p(x \lvert \theta) $. Starting with:

$$
p(x \lvert \theta) = \sum_z p(x,z \lvert \theta) 
$$

Then express the joint distribution of the entire dataset and take the log of
it to find the maximum likelihood estimator of $ \theta $ (If you don't know
what maximum likelihood estimation (MLE) is check out [this
article from towards data science](https://towardsdatascience.com/probability-concepts-explained-maximum-likelihood-estimation-c7b4342fdbb1))

$$ 
\theta = \text{argmax}_{\theta} \log \prod_n p(x_n, \lvert \theta) = \sum_n \log \sum_z p(x_n, z_n \lvert \theta) 
$$

Turns out taking the derivative of this expression to analytically solve for
the argmax of $ \theta $ is intractable because we are
taking the logarithm of a sum. This problem has no analytical solution.
Instead, we will use an iterative method to solve this. We call this
method **Expectation Maximization** (EM).


## Expectation Maximization (EM)

Essentially, the trick of EM is to take the expectation of the variable $ z_n $ instead of summing over all possible values. More generally, this variable
$ z_n $ is called a latent variable. In the case of clustering it is the
cluster index. The variable is not directly observed which is
why we call it latent. 

Let's define a new expression the *expected* log likelihood with respect
to $ q(z_n) $. In this case $ q(z_n) $ is just some distribution of $ z_n $ don't worry too much about it for now we will see what $ q(z_n) $ should
be soon.

$$
Q(\theta) = \sum_n \mathbb{E}_{z_n ~ q(z_n)} \log p(x_n, z_n \lvert \theta)
$$

$$
Q(\theta) = \sum_n \sum_{z_n} q(z_n) \log p(x_n, z_n \lvert \theta)
$$

Now the log is outside of the sum and we can solve this problem. What is $
q(z_n) $ all about though? Turns out if we want to bound the loss of the Q
function to the original MLE problem we should use the following choice:

$$
q(z) = p(z \lvert x; \theta)
$$

This should seem pretty intuitive as well (this is about the most basic choice
of a distribution over the latent variable). This is the distribution of the
latent variable given our data. So now we have 

$$
Q(\theta) = \sum_n \sum_{z_n} p(z \lvert x; \theta) \log p(x_n, z_n \lvert \theta)
$$

Now let's think about how we could actually evaluate this expression. How are
supposed to know the value of $ \theta $? This expression depends only on
information that we do not know. Let's alter this expression a little bit to
include stuff we do know. 

$$
Q(\theta, \theta^{t-1}) = \sum_n \sum_{z_n} p(z_n \lvert x; \theta^{t-1}) \log p(x_n, z_n \lvert \theta)
$$

Remember the EM is an iterative algorithm. We will iteratively be updating
our guesses for $ \theta $ For our next time step prediction of $ \theta^{t} $
we select $ \theta^{t} $ such that:

$$
\theta^t = \text{argmax}_{\theta} Q(\theta, \theta^{t-1})
$$

We know our previous value of $ \theta $ which is $ \theta^{t-1} $.
However, we don't know $ \theta $ in the Q function. We can then maximize that
function by choosing a value of $ \theta $.

Now let's lay out the complete steps for the EM algorithm.

- Randomly initialize the initial values of $ \theta $.
- Remember our complete Q function is given by 
  $$
  Q(\theta, \theta^{t-1}) = \sum_n \sum_{z_n} p(z_n \lvert x; \theta^{t-1}) \log p(x_n, z_n \lvert \theta)
  $$
- The first step is *expectation*. Evaluate the expression $ p(z \lvert x;\theta^{t-1} )$ using the already known value of $ \theta^{t-1} $.
  When actually evaluating this probability we can use Bayes rule to make it
  a little easier to compute. 
  $$
  p(z_n=k \lvert x; \theta^{t-1}) = \frac{p(x_n \lvert z_n = k) p(z_n = k)}{\sum_{k'}
  p(x_n \lvert z_n = k') p(z_n = k')}
  $$
- Next step is <b>maximization</b>. 
  $$
  \theta^t = \text{argmax}_{\theta} Q(\theta, \theta^{t-1})
  $$
  Then set $ \theta^{t-1} = \theta^t $.

This algorithm has theoretical guarantees that the loss of the loss will be
minimized each step. However, like k-means this algorithm can often reach local
minima and it is best to randomly start the algorithm multiple times to get the
best loss. 

Let's see what EM looks like for our GMM problem. 

- Expectation step:
$$
p(z_n=k \lvert x; \theta^{t-1}) = \frac{p(x_n \lvert z_n = k) p(z_n = k)}{\sum_{k'}
  p(x_n \lvert z_n = k') p(z_n = k')}
$$

$$
p(z_n=k \lvert x; \theta^{t-1}) = \frac{\omega_k^{t-1} N(\mu_k^{t-1},
\Sigma_k^{t-1})}{\sum_{k'}
\omega_{k'}^{t-1} N(\mu_{k'}^{t-1}, \Sigma_{k'}^{t-1})}
$$
- Maximization step:
$$
Q(\theta, \theta^{t-1}) = \sum_n \sum_{z_n} p(z_n \lvert x; \theta^{t-1}) \log p(x_n, z_n \lvert \theta)
$$

$$
Q(\theta, \theta^{t-1}) = \sum_n \sum_{k} p(z_n=k \lvert x; \theta^{t-1}) \log p(x_n, z_n \lvert \theta)
$$

$$
Q(\theta, \theta^{t-1}) = \sum_n \sum_{k} p(z_n=k \lvert x; \theta^{t-1}) \log
p(z_n = k) p(x_n \lvert z_n=k ; \theta)
$$

$$
Q(\theta, \theta^{t-1}) = \sum_n \sum_{k} p(z_n=k \lvert x; \theta^{t-1})
\left( \log(w_k) + \log(N(x_n \lvert \mu_k, \Sigma_k)) \right)
$$

And then of course plugging in the value for 
$ p(z_n=k \lvert x; \theta^{t-1}) $ found from the expectation step.

- We then take the derivative of this expression with respect to our parameters
and set it equal to zero to solve for the optimal value of the parameter to
minimize $ Q(\theta, \theta^{t-1}) $. 
$$
\frac{\partial Q}{\partial \mu_k}=0, \frac{\partial Q}{\partial \Sigma_k}=0,
\frac{\partial Q}{\partial \omega_k} = 0
$$
Solving for the respective parameters in each of the cases gives the following.

$$
\omega_k^{*} = \frac{\sum_{n} \gamma_{nk}}{\sum_{k} \sum_{n} \gamma_{nk}}
$$

$$
\Sigma_k^{*} = \frac{1}{\sum_{n} \gamma_{nk}} \sum_{n} \gamma_{nk} (x_n -
\mu_k)(x_n-\mu_k)^T
$$

$$
\mu_k^{*} = \frac{1}{\sum_{n} \gamma_{nk}} \sum_{n} \gamma_{nk} x_n
$$

Where $ \gamma_{nk} = p(z_n=k \lvert x_n; \theta^{t-1}) $

GMM is a more general form of k-means if we set $ \Sigma_k = \sigma^2 I $
then as $ \sigma^2 \rightarrow 0 $ the Q function becomes the same cost
function used in k-means. 

We can use EM to solve the GMM problem and to find complex clusters in our
data. This can be very useful for finding how your data is structured or
finding unsupervised labels.


