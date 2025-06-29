# Naive Bayes & Generative vs Discriminative Models

## What are Models?

**DEFINITION:** Models are representations of the real world using math and statistics, trained to find patterns or make predictions without explicit programming.

Example: Laws of Physics, GenAI, Linear Regression, Classification or Decision Trees

Models have some characteristic properties:

1. Abstraction of Reality

    - Simplify complex systems into rules or equations

2. Learns from Data

    - Uses algorithms to identify relationships between *inputs* and *outputs*

3. Generalization

    - A good model performs well on data that it has never seen before

    Ex. We can study many apples falling from a tree and deduce that F = ma for variables F, m, and a that we defined. However, this model becomes useful or good only when the same formula applies well to other apples we did not study. It gets more generalizable when we find that it applies to basketballs, or flowing water, or the motion of planets. 

## Generative vs Discriminative Models

![generative vs discriminative models](https://substackcdn.com/image/fetch/$s_!1x4N!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F1b23e2b9-5705-4fe5-9c9e-61d9abcd4b98_3078x3882.jpeg)

### What Are These Models?

Machine learning models can be broadly categorized into two types based on how they approach learning and making predictions: **generative** and **discriminative** models.

### Discriminative Models

Discriminative models focus on learning the **boundary** between different classes or categories. They ask: "Given some input data, what's the most likely category it belongs to?"

#### How They Work
- They learn to distinguish between different classes directly
- They model the probability of a class given the input data: P(class|data)
- They draw decision boundaries to separate different categories

#### Examples
- **Logistic Regression**: Learns to classify emails as spam or not spam
- **Support Vector Machines (SVM)**: Creates boundaries to separate different groups
- **Neural Networks**: Can classify images as cats, dogs, or birds

#### Think of it like...
A discriminative model is like a security guard at a club who learns to identify whether someone is over 21 or under 21. The guard doesn't need to know everything about what makes a person that age - just enough features to make the distinction.

### Generative Models

Generative models learn to understand the **underlying patterns** of the data itself. They ask: "What does typical data in each category look like, and can I create new examples?"

#### How They Work
- They learn the probability distribution of the data itself: P(data|class)
- They can generate new, similar data points
- They understand what makes each class "tick"

#### Examples
- **Gaussian Naive Bayes**: Models what typical spam and non-spam emails look like
- **GANs (Generative Adversarial Networks)**: Can create realistic fake images
- **Large Language Models**: Generate human-like text based on patterns they've learned

#### Think of it like...
A generative model is like an artist who studies thousands of paintings from different art movements. Not only can they identify whether a painting is Impressionist or Cubist, but they can also paint new works in either style.

### Key Differences

**Purpose:**
- Discriminative: "Which category does this belong to?"
- Generative: "What does data from this category look like?"

**Learning Focus:**
- Discriminative: Learns decision boundaries
- Generative: Learns data distributions

**Capabilities:**
- Discriminative: Classification and prediction
- Generative: Classification, prediction, AND data generation

**Data Efficiency:**
- Discriminative: Often needs less data for classification tasks
- Generative: Usually requires more data but provides richer understanding

### When to Use Which?

**Choose Discriminative Models when:**
- You only need to classify or predict
- You have limited training data
- You want simpler, faster models
- Accuracy on classification is your main goal

**Choose Generative Models when:**
- You want to understand the data structure
- You need to generate new data
- You're working with missing data
- You want to detect outliers or anomalies

### Real-World Example

Imagine you're building a system to identify handwritten digits (0-9):

**Discriminative approach:** Train a model to look at pixel patterns and directly classify "this looks like a 7" or "this looks like a 3." It learns the minimal features needed to distinguish between digits.

**Generative approach:** Learn what each digit typically looks like - the curves of a 2, the angles of a 7, etc. This model could then generate new handwritten digits that look realistic, and it can also classify existing ones.

Both approaches can solve the classification problem, but the generative model gives you the bonus ability to create new handwritten digits that look authentic.

## Math Review: Bayes's Theorem
On Board - Refer to Wikipedia

## The 'Naive' Assumption

Sometimes, the context around a given target event that you wish to model is complicated and has many interconnected parts.

For example, when using the text of an email to decide whether an email is spam or not spam (called 'ham'), the words used, the word order, the images, the capitalization, the urgency of the message, and more all contribute to the outcome and are related to one another in deep ways.

However, we often don't know these inter-relationships or modeling them is difficult. So, in practice, we might just want to say that they are all independent and unrelated. Then, we can use the rule for multiplying probabilities to turn a collection of complicated conditions into an easy product of simpler conditions!

This is usually very incorrect and in theory, shouldn't work. In practice though, it actually works pretty well for many scenarios.