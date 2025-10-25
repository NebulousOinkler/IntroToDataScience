# Multilayer Perceptrons and Backpropagation: A Practical Introduction

**Duration:** 3 hours total (2 hours lecture + 1 hour hands-on notebook exercises)  
**Prerequisites:** Basic Python, basic linear algebra (matrices, vectors)  
**Materials:** MLP_Class.ipynb notebook

---

## Learning Objectives

By the end of this lesson, students will be able to:
1. Explain the architecture and components of a multilayer perceptron (MLP)
2. Understand how neural networks make predictions through forward propagation
3. Grasp the intuition behind backpropagation for training neural networks
4. Implement key components of an MLP from scratch in Python
5. Train a simple neural network to solve a regression problem

---

## Part 1: Introduction to Neural Networks (25 minutes)

### What is a Neural Network?

A neural network is a computational model inspired by how biological neurons work. Think of it as a sophisticated function approximator that learns patterns from data.

**Key Components:**

- **Input Layer:** Where data enters the network
- **Hidden Layer(s):** Where the network learns complex patterns
- **Output Layer:** Where predictions are made
- **Weights:** The parameters the network learns
- **Activation Functions:** Add non-linearity (help network learn complex patterns)

### Real-World Analogy

Imagine teaching a child to recognize animals:

1. You show them examples (forward pass)
2. They make a guess ("Is this a cat?")
3. You tell them if they're right or wrong (loss)
4. They adjust their understanding (backpropagation)
5. Over time, they get better at recognizing animals

This is exactly how neural networks learn!

### Why "Multilayer"?

- **Single layer:** Can only learn linear patterns (like a straight line)
- **Multiple layers:** Can learn complex, non-linear patterns (curves, shapes, etc.)

### Historical Context (5 minutes)

**The Perceptron (1950s-60s):**
- Single layer neural network
- Could only solve linearly separable problems
- Led to "AI Winter" when limitations were discovered

**The Breakthrough (1980s):**
- Backpropagation algorithm formalized
- Allowed training of deeper networks
- Opened path to modern deep learning

**Today:**
- Neural networks power: image recognition, language models (ChatGPT), recommendation systems, autonomous vehicles, and more!

---

## Part 2: Network Architecture - Building Blocks (30 minutes)

### 2.1 Weights and Biases

**Weights (W):** Think of these as the "strength" of connections between neurons
- Positive weights amplify signals
- Negative weights suppress signals
- Zero weights mean no connection

**Bias:** A constant we add to allow flexibility (like the y-intercept in y = mx + b)
- In our notebook, we've cleverly hidden bias by appending 1 to our input features

### 2.2 Activation Functions: The Key to Non-linearity

**Without activation functions:** Even with many layers, the network can only learn linear patterns

**With activation functions:** The network can learn complex curves and patterns

**ReLU (Rectified Linear Unit):** Our choice for this lesson
```
ReLU(z) = max(z, 0)
```
- If input is positive → pass it through unchanged
- If input is negative → output zero
- Simple, fast, and works well in practice!

**Why is this non-linear?**
The "max" operation creates a bend in the function. Combining many of these bends allows the network to approximate any curve!

### 2.3 Matrix Representation

Neural networks use matrices for efficiency. Instead of processing one example at a time, we process many simultaneously!

**Example:**
- Input: 100 data points, each with 2 features → Matrix of size (100, 2)
- First layer: 2 input nodes, 3 hidden nodes → Weight matrix (2, 3)
- Multiplication: (100, 2) @ (2, 3) = (100, 3) ← All 100 examples processed at once!

---

## Part 3: Forward Propagation - Making Predictions (20 minutes)

### The Forward Pass Process

Think of forward propagation as data flowing through the network, being transformed at each layer.

**Step-by-Step:**

1. **Start with input data (X)**
   - Example: [temperature, humidity] for weather prediction

2. **For each layer:**
   - **Multiply by weights:** `A = Z × W`
     - This combines features in different ways
   
   - **Apply activation function:** `Z = ReLU(A)`
     - This adds non-linearity (except for the last layer)

3. **Final layer produces prediction**
   - No activation on final layer for regression (we want any real number)

### Visualization

```
Input Layer     Hidden Layer      Output Layer
   (X)              (Z₁)              (Z₂)
    
    x₁ ───┐
          ├──→ [W₁] ──→ h₁ ───┐
    x₂ ───┘              h₂ ───┼──→ [W₂] ──→ ŷ
                         h₃ ───┘
```

### Important Lists: A and Z

To make backpropagation efficient, we save intermediate results:
- **A (activations before ReLU):** Needed to compute gradients
- **Z (activations after ReLU):** Become inputs to next layer

---

## Part 4: Loss Functions - Measuring Error (15 minutes)

### What is Loss?

Loss measures how wrong our predictions are. Lower loss = better predictions!

**For Regression: Mean Squared Error (MSE)**

```
MSE = (1/n) × Σ(prediction - actual)²
```

**Why square the errors?**
1. Makes all errors positive (we care about magnitude, not direction)
2. Penalizes large errors more than small ones
3. Mathematically convenient for calculus

**Intuition:**
- If we predict 5 but actual is 3 → error = 2 → squared error = 4
- If we predict 10 but actual is 3 → error = 7 → squared error = 49 (ouch!)

### The Gradient of Loss

The gradient tells us **which direction to adjust our weights** to reduce loss.

For MSE, the gradient with respect to output is:
```
∇MSE = (2/n) × (prediction - actual)
```

**No calculus required to understand:** 
- If prediction > actual → gradient is positive → we need to decrease our prediction
- If prediction < actual → gradient is negative → we need to increase our prediction
- The factor of 2 comes from the derivative of x² (just a scaling constant)

---

## Part 5: Backpropagation - The Learning Algorithm (30 minutes)

### The Big Picture

Backpropagation answers the question: **"How should we adjust each weight to reduce our loss?"**

It works backward through the network (hence "back-propagation"), computing how much each weight contributed to the error.

### The Chain Rule (No Calculus Required!)

Imagine you're in a relay race:
- Runner 4 (output) crosses the finish line too late
- How much is each runner responsible?
- We work backward: check runner 3, then 2, then 1

Similarly, backpropagation traces error backward through the network.

### The Algorithm (Intuitive Version)

1. **Start with the error at the output**
   - "How wrong was our final prediction?"

2. **For each layer (going backward):**
   - Compute gradient for that layer's weights
   - Pass the "blame" to the previous layer (scaled appropriately)

3. **Use ReLU gradient to know what gets through**
   - Remember: ReLU zeros out negative values
   - So gradients don't flow back through "dead" neurons (where input was negative)

### Key Equation: Computing Weight Gradients

For each layer, the gradient for weights W is:
```
∇W = Z_previous.T @ delta
```

Where:
- `Z_previous` = outputs from the previous layer (what came in)
- `delta` = the error signal flowing backward
- `.T` means transpose (flip rows and columns)
- `@` means matrix multiplication

**Intuition:** This tells us how much each input (Z_previous) contributed to the error (delta) at this layer.

### Updating Delta for Next Layer

After computing the gradient for weights, we update delta to pass backward:
```
delta_new = ReLU_grad(A) × (delta @ W.T)
```

**Breaking it down:**
- `delta @ W.T` → How much error flows back through the weights
- `ReLU_grad(A)` → Only let error flow back through neurons that were "active" (positive)
- `×` is element-wise multiplication (both must match this shape)

---

## Part 6: Training with Gradient Descent (15 minutes)

### The Optimization Loop

Now we have all the pieces! Training is just a loop:

```python
for each epoch:
    1. Forward pass → make predictions
    2. Compute loss → how wrong are we?
    3. Backward pass → compute gradients
    4. Update weights → W = W - learning_rate × gradient
```

### Learning Rate: The Step Size

The learning rate controls how much we adjust weights each iteration.

**Too small:** Learning is very slow (might take forever!)
**Too large:** We overshoot and never converge (bouncing around wildly)
**Just right:** Steady progress toward better predictions

**Typical values:** 0.001 to 0.1 (often requires experimentation)

### Watching Training Progress

Good signs:
- Loss decreases over time
- Predictions get closer to actual values
- Final plot shows network fitting the data well

Bad signs:
- Loss increases or fluctuates wildly → learning rate too high
- Loss barely changes → learning rate too low OR network stuck
- Perfect training fit but poor on new data → overfitting

---

## Part 7: Hands-On Implementation (60 minutes - separate session)

### Overview

This session is dedicated to implementing the concepts in the MLP_Class.ipynb notebook. Students will work through exercises with instructor support.

**Structure:**
- 5 minutes: Review notebook structure and expectations
- 50 minutes: Guided implementation with checkpoints
- 5 minutes: Run full training and discuss results

### Implementation Tasks

**Task 1: Initialize Weights (10 min)**
- Create random weight matrices for each layer
- Remember: shape must be (input_size, output_size)
- **Checkpoint:** Run initweights tests

**Task 2: Forward Pass (15 min)**
- Loop through weight matrices
- Apply matrix multiplication and ReLU
- Remember: No ReLU on the final layer!
- Store both A and Z for backprop
- **Checkpoint:** Run forward_pass tests

**Task 3: MSE Loss and Gradient (10 min)**
- Compute mean squared error
- Compute gradient: 2/n × (prediction - actual)
- **Checkpoint:** Run MSE tests

**Task 4: Backpropagation (15 min)**
- Start with delta from MSE gradient
- Loop backward through layers
- Compute weight gradients
- Update delta for previous layer
- **Checkpoint:** Run backprop tests

**Tips for Success:**
- Pay attention to matrix shapes (use .shape to check!)
- Remember: Gradients must match weight shapes
- The zip() with [::-1] reverses the lists for backward pass
- Don't forget to reverse gradients list at the end!

### Training and Experimentation (10 minutes)

**Run the Training Loop**

Execute the final cell to see your network learn!

**What to observe:**

1. **Loss curve:** Should decrease over time
2. **Final fit:** Red line should match the blue dots reasonably well
3. **Training time:** 100,000 epochs might take 30-60 seconds

**Experiment!**

Try adjusting:

- **Learning rate:** What happens with 0.0001 vs 0.01?
- **Network architecture:** Change [2, 20, 1] to [2, 10, 10, 1] (two hidden layers!)
- **Hidden layer size:** More neurons = more capacity (but slower)

---

## Part 8: Common Pitfalls and Debugging (Integrated into hands-on)

### Shape Mismatches
**Error:** "operands could not be broadcast together"
**Fix:** Check that matrix dimensions align for multiplication

### Gradient Direction Error
**Symptom:** Loss increases instead of decreases
**Fix:** Make sure you're subtracting gradients, not adding: `W = W - lr × grad`

### Reversed Gradients
**Symptom:** Tests pass except for order-checking test
**Fix:** Don't forget `gradients.reverse()` at the end of backprop!

### Zero Gradients
**Symptom:** Loss doesn't change at all
**Fix:** Make sure you're not setting gradients to zero accidentally

---

## Part 10: Key Takeaways and Wrap-Up (10 minutes)

### What We Learned

1. **Neural networks are universal function approximators**
   - With enough layers and neurons, they can learn almost any pattern

2. **Forward propagation is just matrix multiplication + activation functions**
   - Data flows forward, gets transformed at each layer

3. **Backpropagation uses the chain rule to compute gradients**
   - Error flows backward, tells each weight how to improve

4. **Training is an iterative optimization process**
   - Adjust weights little by little using gradients

5. **Everything is just linear algebra and basic calculus**
   - No magic! Just matrix operations repeated many times

### The Power and Limitations

**Strengths:**
- Learn complex patterns automatically
- Work across many domains (vision, language, etc.)
- Can improve with more data

**Limitations:**
- Need lots of data to train well
- Can be computationally expensive
- "Black box" - hard to interpret decisions
- Can overfit if not careful

---

## Additional Resources

### For Students Who Want More Math
- 3Blue1Brown YouTube series on Neural Networks (excellent visualizations!)
- "Neural Networks and Deep Learning" by Michael Nielsen (free online book)
- Andrew Ng's Machine Learning course on Coursera

### For Hands-On Practice
- Kaggle competitions and datasets
- Fast.ai course (practical deep learning)
- TensorFlow/PyTorch tutorials

### For Intuitive Understanding
- "Deep Learning" by Ian Goodfellow (comprehensive textbook)
- Distill.pub (interactive articles on ML)
- Papers with Code (see state-of-the-art implementations)

---

## Assessment Questions

### Conceptual Understanding
1. Why do we need activation functions in neural networks?
2. What does the gradient tell us during backpropagation?
3. Why do we save the outputs (A and Z) during forward propagation?

### Applied Understanding
1. If your loss is increasing during training, what might be wrong?
2. How would you modify the network to learn more complex patterns?
3. Why might a network with many layers train more slowly?

---

## Instructor Notes

### Time Management (3-hour class)

**Hour 1: Foundations (60 minutes)**
- Part 1: Introduction (25 min)
- Part 2: Architecture & Building Blocks (30 min)
- Buffer/Questions (5 min)

**Hour 2: Theory & Algorithms (60 minutes)**
- Part 3: Forward Propagation (20 min)
- Part 4: Loss Functions (15 min)
- Part 5: Backpropagation (20 min)
- Buffer/Questions (5 min)

**Hour 3: Implementation (60 minutes)**
- Part 6: Training with Gradient Descent (15 min)
- Part 7: Hands-on Notebook Exercises (50 min)
  - Students work through MLP_Class.ipynb
  - Instructor circulates and helps
  - Checkpoints after each function
- Part 8: Debugging tips (integrated into hands-on)
- Part 9: Wrap-up (5 min)

### Suggested Break Schedule

- **After Hour 1** (10-minute break)
- **After Hour 2** (5-minute break before switching to hands-on)

### Pacing Tips

- **If running ahead:** Add more examples, deeper dives into activation functions, or discuss network architecture design principles
- **If running behind:** Reduce historical context, skip some activation function alternatives, focus on ReLU only

### Common Student Questions

**Q: "Why do we square the error instead of using absolute value?"**
A: Squaring is mathematically smoother (has a derivative everywhere) and penalizes large errors more heavily. In practice, both work!

**Q: "How do we know what architecture to use?"**
A: Often through experimentation! Start simple, add complexity if needed. Rules of thumb exist but aren't perfect.

**Q: "Is backpropagation the only way to train neural networks?"**
A: No! But it's by far the most common and efficient for most problems. Alternatives include genetic algorithms, reinforcement learning, etc.

**Q: "What if my calculus is rusty?"**
A: Focus on the intuition! Gradients point in the direction of steepest increase. We subtract them to decrease loss. The math details are less important than the concept.

### Extension Activities

For faster students:
1. Implement different activation functions (sigmoid, tanh)
2. Add L2 regularization to prevent overfitting
3. Implement mini-batch gradient descent
4. Try a classification problem instead of regression

---

## Conclusion

Congratulations! You've now implemented a neural network from scratch and understand how it learns. This foundation will serve you well as you explore more advanced architectures like CNNs, RNNs, and Transformers - they all use these same basic building blocks!

**Remember:** Deep learning is powerful, but it's not magic. It's just matrix multiplication, activation functions, and gradient descent - concepts you now understand!

---

*This lesson plan is designed for Cornell's Machine Learning curriculum and pairs with the MLP_Class.ipynb notebook.*
