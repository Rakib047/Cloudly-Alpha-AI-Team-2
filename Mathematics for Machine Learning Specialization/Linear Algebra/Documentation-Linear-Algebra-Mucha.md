
# Week 1
## Operations with vectors

Vectors are objects that can represent movement in space or describe an object's attributes. In physics, vectors are often thought of as entities that move us through physical space, while in data science, they are used to represent attributes of objects, like a house's characteristics (e.g., square footage, number of bedrooms, price).

The key operations on vectors include:

1. **Addition**: Vectors can be added together. The order of addition doesn’t affect the result (commutative property), and the addition of three vectors can be grouped in any way (associative property).
2. **Scalar Multiplication**: A vector can be scaled by multiplying it by a number (scalar), which changes its magnitude.
3. **Vector Subtraction**: Subtracting a vector is equivalent to adding the negative of that vector.
4. **Coordinate Systems**: Vectors are often represented in a coordinate system using basis vectors (e.g., i and j) to describe directions.

These operations allow vectors to be used not only in geometric spaces but also in non-spatial contexts, like data science, where vectors can represent complex attributes of objects.

### Mathematical Equation:

For **vector addition** and **scalar multiplication**:

Let:

* $\mathbf{r} = \langle 3, 2 \rangle$,
* $\mathbf{s} = \langle -1, 2 \rangle$.

The **addition** of $\mathbf{r}$ and $\mathbf{s}$ is:

$$
\mathbf{r} + \mathbf{s} = \langle 3 + (-1), 2 + 2 \rangle = \langle 2, 4 \rangle
$$

Scalar multiplication of $\mathbf{r}$ by 2:

$$
2\mathbf{r} = 2 \times \langle 3, 2 \rangle = \langle 6, 4 \rangle
$$

These equations demonstrate how vector addition and scalar multiplication work in a 2D space.
# Week 2

## Modulus & inner product

#### Length (or Magnitude) of a Vector:

The length or magnitude of a vector is calculated using the Pythagorean theorem. For a 2D vector 
$\mathbf{r} = a\mathbf{i} + b\mathbf{j}$, the length of the vector is the square root of the sum of the squares of its components:

$$
\text{Length of } \mathbf{r} = \sqrt{a^2 + b^2}
$$

This idea can be generalized to vectors with more dimensions or even those with different units (like length, time, price, etc.).

#### Dot Product:

The dot product (or inner product) of two vectors $\mathbf{r}$ and $\mathbf{s}$ is calculated by multiplying their corresponding components and summing the results:

$$
\mathbf{r} \cdot \mathbf{s} = r_i \times s_i + r_j \times s_j
$$

This results in a scalar value. The dot product has several important properties:

- **Commutativity**: The dot product is commutative, meaning 
  $$
  \mathbf{r} \cdot \mathbf{s} = \mathbf{s} \cdot \mathbf{r}
  $$

- **Distributivity**: The dot product is distributive over vector addition:
  $$
  \mathbf{r} \cdot (\mathbf{s} + \mathbf{t}) = \mathbf{r} \cdot \mathbf{s} + \mathbf{r} \cdot \mathbf{t}
  $$

- **Associativity with scalar multiplication**:
  $$
  a (\mathbf{r} \cdot \mathbf{s}) = \mathbf{r} \cdot (a \mathbf{s})
  $$

#### Connection Between Dot Product and Length:

The dot product of a vector with itself gives the square of its length:

$$
\mathbf{r} \cdot \mathbf{r} = r_1^2 + r_2^2 + \dots + r_n^2
$$

This shows that the length of a vector can be found by taking the square root of the dot product of the vector with itself.

#### Mathematical Equation:

The key equations are:

- **Length of a vector** $\mathbf{r}$:
  $$
  \text{Length of } \mathbf{r} = \sqrt{r_1^2 + r_2^2 + \dots + r_n^2}
  $$

- **Dot product of vectors** $\mathbf{r}$ and $\mathbf{s}$:
  $$
  \mathbf{r} \cdot \mathbf{s} = r_1 s_1 + r_2 s_2 + \dots + r_n s_n
  $$

- **Length from dot product**:

  $$
  \text{Length of } \mathbf{r} = \sqrt{\mathbf{r} \cdot \mathbf{r}}
  $$

These equations define how to compute both the magnitude (length) and the dot product of vectors, essential tools in vector analysis.

## Cosine & dot product

#### Cosine Rule:

The cosine rule in algebra states that in a triangle with sides \(a\), \(b\), and \(c\), and the angle between \(a\) and \(b\) being \(\theta\):

$$
c^2 = a^2 + b^2 - 2ab \cdot \cos(\theta)
$$

This is the basis for deriving the relationship between the dot product of two vectors.

#### Translation into Vector Notation:

We define two vectors, \(\mathbf{r}\) and \(\mathbf{s}\), and the difference between them as \(\mathbf{r} - \mathbf{s}\). The length of this difference vector is given by:

$$
\|\mathbf{r} - \mathbf{s}\|^2 = \|\mathbf{r}\|^2 + \|\mathbf{s}\|^2 - 2 \|\mathbf{r}\| \|\mathbf{s}\| \cos(\theta)
$$

This equation mirrors the cosine rule but in the context of vectors.

#### Expanding the Dot Product:

We expand the dot product of \(\mathbf{r} - \mathbf{s}\) with itself:

$$
(\mathbf{r} - \mathbf{s}) \cdot (\mathbf{r} - \mathbf{s}) = \mathbf{r} \cdot \mathbf{r} - 2 \mathbf{r} \cdot \mathbf{s} + \mathbf{s} \cdot \mathbf{s}
$$

This simplifies to:

$$
\|\mathbf{r}\|^2 - 2 \mathbf{r} \cdot \mathbf{s} + \|\mathbf{s}\|^2
$$

By comparing this with the right-hand side of the original cosine rule equation, we derive the formula for the dot product:

$$
\mathbf{r} \cdot \mathbf{s} = \|\mathbf{r}\| \|\mathbf{s}\| \cos(\theta)
$$

This shows that the dot product gives the product of the magnitudes of the two vectors, scaled by the cosine of the angle between them.

#### Interpretation of the Dot Product:

- If the vectors are in the same direction (\(\theta = 0^\circ\)), \(\cos(0^\circ) = 1\), so the dot product equals the product of their magnitudes.
- If the vectors are orthogonal (at 90 degrees), \(\cos(90^\circ) = 0\), and the dot product is 0.
- If the vectors are in opposite directions (\(\theta = 180^\circ\)), \(\cos(180^\circ) = -1\), so the dot product equals the negative product of their magnitudes.

#### Mathematical Equation:

The important equation derived from the cosine rule and dot product properties is:

$$
\mathbf{r} \cdot \mathbf{s} = \|\mathbf{r}\| \|\mathbf{s}\| \cos(\theta)
$$

This equation shows that the dot product of two vectors is the product of their magnitudes and the cosine of the angle between them.

#### Key Scenarios for the Dot Product:

- **Same Direction (\(\theta = 0^\circ\))**:
  $$
  \mathbf{r} \cdot \mathbf{s} = \|\mathbf{r}\| \|\mathbf{s}\|
  $$

- **Orthogonal Vectors (\(\theta = 90^\circ\))**:
  $$
  \mathbf{r} \cdot \mathbf{s} = 0
  $$

- **Opposite Directions (\(\theta = 180^\circ\))**:
  $$
  \mathbf{r} \cdot \mathbf{s} = -\|\mathbf{r}\| \|\mathbf{s}\|
  $$

This highlights how the dot product encodes information about the directionality and magnitude relationship between vectors.

## Projection:

The projection of one vector \(\mathbf{s}\) onto another vector \(\mathbf{r}\) represents how much of \(\mathbf{s}\) "shadows" or aligns with \(\mathbf{r}\). This projection is related to the dot product and involves the cosine of the angle \(\theta\) between the two vectors:

$$
\mathbf{r} \cdot \mathbf{s} = \|\mathbf{r}\| \|\mathbf{s}\| \cos(\theta)
$$

The projection of \(\mathbf{s}\) onto \(\mathbf{r}\) is essentially the adjacent side in the triangle formed by the two vectors, and it represents the magnitude of \(\mathbf{s}\) in the direction of \(\mathbf{r}\).

#### Scalar Projection:

The scalar projection is the length of the projection, and it is given by:

$$
\frac{\mathbf{r} \cdot \mathbf{s}}{\|\mathbf{r}\|}
$$

This gives us the length of the shadow (projection) of \(\mathbf{s}\) onto \(\mathbf{r}\).

#### Vector Projection:

The vector projection is a vector in the same direction as \(\mathbf{r}\), with its magnitude equal to the scalar projection. This is defined as:

$$
\frac{\mathbf{r} \cdot \mathbf{s}}{\mathbf{r} \cdot \mathbf{r}} \mathbf{r}
$$

This result provides the projection of \(\mathbf{s}\) onto \(\mathbf{r}\) as a vector, which points in the same direction as \(\mathbf{r}\).

#### Interpretation of Projection:

- When the vectors are perpendicular (\(\theta = 90^\circ\)), the dot product is zero, meaning there is no projection.
- The dot product also reveals how much one vector goes in the same direction as another. If the vectors are pointing in the same direction, the projection is maximized; if they are in opposite directions, the projection is negative.

#### Mathematical Equations:

- **Scalar Projection**:
  $$
  \text{Scalar Projection of } \mathbf{s} \text{ onto } \mathbf{r} = \frac{\mathbf{r} \cdot \mathbf{s}}{\|\mathbf{r}\|}
  $$

- **Vector Projection**:
  $$
  \text{Vector Projection of } \mathbf{s} \text{ onto } \mathbf{r} = \frac{\mathbf{r} \cdot \mathbf{s}}{\mathbf{r} \cdot \mathbf{r}} \mathbf{r}
  $$

- **General Projection Formula**:
  $$
  \mathbf{r} \cdot \mathbf{s} = \|\mathbf{r}\| \|\mathbf{s}\| \cos(\theta)
  $$

This establishes how the dot product not only calculates the angle between vectors but also how much one vector projects onto another, giving us both scalar and vector projections that reflect the alignment and magnitude in that direction.

## Coordinate System:

A vector, such as \(\mathbf{r}\), is an object that represents movement from the origin to some point in space.

The space can be defined using a coordinate system, which is often based on basis vectors. These basis vectors provide the framework to describe any vector in that space.

#### Basis Vectors:

A basis vector is a vector that defines the coordinate system. In a 2D space, basis vectors \(\mathbf{e}_1\) and \(\mathbf{e}_2\) could be unit vectors along the x and y axes.

A vector can be expressed as a linear combination of the basis vectors. For example, in the coordinate system defined by \(\mathbf{e}_1\) and \(\mathbf{e}_2\), a vector \(\mathbf{r} = 3\mathbf{e}_1 + 4\mathbf{e}_2\) might be written as \(\mathbf{r} = \langle 3, 4 \rangle\).

#### Changing Coordinate Systems:

You can change from one coordinate system (with one set of basis vectors) to another. The new coordinate system can be defined by a new set of basis vectors, like \(\mathbf{b}_1\) and \(\mathbf{b}_2\).

The key to switching between coordinate systems is projection. You can find the coordinates of \(\mathbf{r}\) in the new basis by projecting \(\mathbf{r}\) onto the new basis vectors.

This is done by computing the dot product of the vector with the basis vectors in the new coordinate system.

#### Projection and Dot Product:

The projection of \(\mathbf{r}\) onto a new basis vector \(\mathbf{b}_1\) is given by:

$$
\text{Projection of } \mathbf{r} \text{ onto } \mathbf{b}_1 = \frac{\mathbf{r} \cdot \mathbf{b}_1}{\|\mathbf{b}_1\|^2} \mathbf{b}_1
$$

Similarly, you can compute the projection of \(\mathbf{r}\) onto \(\mathbf{b}_2\).

Once you have these projections, you can express \(\mathbf{r}\) in terms of the new basis vectors.

#### Checking for Orthogonality:

To safely use projections in a new coordinate system, the new basis vectors need to be orthogonal (at right angles to each other).

You can check for orthogonality by calculating the dot product of the basis vectors. If the dot product is zero, the vectors are orthogonal.

#### Example of Changing Coordinates:

The example works through finding the coordinates of \(\mathbf{r} = (3, 4)\) in the new basis \(\mathbf{b}_1 = (2, 1)\) and \(\mathbf{b}_2 = (-2, 4)\).

By projecting \(\mathbf{r}\) onto these basis vectors and using the dot product, you can find that \(\mathbf{r}\) in the new basis is:

$$
\mathbf{r} = 2\mathbf{b}_1 + \frac{1}{2} \mathbf{b}_2
$$

#### Mathematical Equations:

##### Projection of \(\mathbf{r}\) onto a Basis Vector:

$$
\text{Projection of } \mathbf{r} \text{ onto } \mathbf{b}_1 = \frac{\mathbf{r} \cdot \mathbf{b}_1}{\|\mathbf{b}_1\|^2} \mathbf{b}_1
$$

##### Dot Product for Orthogonality Check:

$$
\mathbf{b}_1 \cdot \mathbf{b}_2 = 0
$$

If the dot product is zero, the vectors are orthogonal.

##### Vector Representation in the New Basis:

After finding the projections, the vector \(\mathbf{r}\) in the new basis is expressed as:

$$
\mathbf{r} = \text{Projection of } \mathbf{r} \text{ onto } \mathbf{b}_1 + \text{Projection of } \mathbf{r} \text{ onto } \mathbf{b}_2
$$

This shows how the vector \(\mathbf{r}\) can be described in a new coordinate system using a new set of orthogonal basis vectors. The process involves using projections and dot products to switch from one set of basis vectors to another, making it an essential technique in vector algebra and data science.


## Basis:

A basis is a set of vectors that defines a vector space. For a set of \(n\) vectors to be a basis, they must be linearly independent, meaning no vector in the set can be written as a linear combination of the others.

### Linear Independence:

If you cannot express a vector as a combination of others in the set, the vectors are linearly independent. For example, in 2D, two vectors \(\mathbf{b}_1\) and \(\mathbf{b}_2\) can span the entire plane, meaning any point in the plane can be reached by scaling and adding these two vectors. Adding a third vector \(\mathbf{b}_3\) that is not a linear combination of \(\mathbf{b}_1\) and \(\mathbf{b}_2\) allows you to span a 3D space.

If a vector can be written as a combination of the others, it is linearly dependent and doesn't contribute to increasing the dimensionality of the space.

### Dimensionality:

The number of linearly independent basis vectors in a space determines its dimension. For example:
- Two linearly independent vectors span a 2D space.
- Three linearly independent vectors span a 3D space, and so on.

### Basis Vectors:

Basis vectors do not have to be unit vectors (vectors of length 1), and they do not have to be orthogonal (at 90 degrees to each other). However, orthonormal basis vectors (both unit length and orthogonal) make computations much easier.

### Orthonormal Basis:

If the basis vectors are orthogonal and have unit length, the math involved in vector operations becomes simpler, as operations like dot products become straightforward.

### Mapping Between Basis Vectors:

When transitioning from one coordinate system (defined by one set of basis vectors) to another, the mapping preserves the linear structure of the space. The grid defined by the original basis vectors is projected onto the new basis, maintaining evenly spaced relationships between points, though values may differ.

Even if the new basis vectors are not orthogonal, the operations (vector addition and scalar multiplication) still hold, but the transformation may involve more complex operations (such as using matrices) instead of simple dot products.

## Key Concepts and Definitions:

- **Basis**: A set of linearly independent vectors that spans a vector space.
- **Linear Independence**: A set of vectors is linearly independent if no vector can be written as a linear combination of the others.
- **Dimension**: The number of linearly independent vectors in a basis, defining the dimensionality of the space (e.g., 2D, 3D).
- **Orthonormal Basis**: A set of vectors that are both orthogonal (at right angles to each other) and of unit length. While not required, these make computations simpler.
- **Mapping Between Basis Vectors**: The transformation from one basis to another maintains the structure of the vector space, but may require matrix operations if the basis vectors are not orthogonal.

## Mathematical Concept:

### Linear Independence Test:

If you have a set of vectors \(\{\mathbf{b}_1, \mathbf{b}_2, \mathbf{b}_3\}\), they are linearly independent if no scalar coefficients \(a_1, a_2, a_3\) exist such that:

$$
a_1 \mathbf{b}_1 + a_2 \mathbf{b}_2 + a_3 \mathbf{b}_3 = 0
$$

where \(a_1 = a_2 = a_3 = 0\) is the only solution.

### Dimension of the Space:

- A 2D space has two linearly independent vectors.
- A 3D space has three linearly independent vectors.

By understanding these concepts, we can describe a vector space in terms of its basis and identify how transformations between different coordinate systems can be handled using projections or matrix operations, depending on whether the basis vectors are orthogonal.

## Applications of changing basis

### Data Points and Linear Relationships:

Imagine having a set of 2D data points that nearly lie on a straight line. You can transform this data by projecting the points onto the line, measuring how far each point is along the line and how far it is from the line.

This transformation reduces the 2D space into two components: how far along the line (which is useful) and how far off the line (which represents the "noise").

### Noise and Fit Quality:

The distance from the line represents the noise in the data. Smaller distances suggest the data fits the line well, while larger distances indicate more variation from the line.

This noise can be useful in understanding the quality of the fit. A good fit minimizes this noise, while a poor fit maximizes it. The goal is to collapse the noise dimension in data science to focus on the important features.

### Orthogonality and Projections:

The directions "along the line" and "away from the line" are orthogonal to each other. This orthogonality makes it possible to use the dot product to project the data from the 2D space onto the 1D space along the line and perpendicular to it.

By projecting the data along these orthogonal directions, we can focus on the most important feature of the data (how far along the line) while minimizing the noise (how far off the line).

### Application in Machine Learning:

In machine learning, for example, when dealing with image recognition (like facial recognition), the goal is to transform the raw pixel data into a new basis that captures meaningful features (e.g., nose shape, skin tone, eye distance).

The learning process of a neural network is to identify the basis vectors that represent the most significant features of the data and discard less useful information (like raw pixel data).

### Dimensionality and Linear Independence:

The dimensionality of a vector space depends on the number of linearly independent basis vectors. The basis vectors define how many independent directions exist in the space.

A set of vectors is linearly independent if no vector in the set can be written as a linear combination of the others. This is crucial for understanding the number of dimensions and how transformations can occur in a space.

### Key Concepts and Definitions:

- **Basis Vectors**: A set of vectors that define a coordinate system. They are linearly independent, meaning no vector in the set can be written as a combination of others.
- **Linear Independence**: A set of vectors is linearly independent if none of them can be expressed as a linear combination of the others. This determines the dimensionality of the space.
- **Projections**: The process of mapping a vector onto another vector or subspace. In the case of data points on a line, we project the data onto the line and measure how far off the line each point is (representing noise).
- **Dimensionality**: The number of linearly independent vectors in a basis defines the dimension of a space. In machine learning, this dimensionality helps in reducing complex data into more manageable, informative features.

### Machine Learning Application:

In a neural network, instead of using raw pixel data, we want to transform the data into a new basis (e.g., capturing features like the nose shape or the distance between eyes) that provides the most meaningful information for the task at hand (such as facial recognition).

### Mathematical Concept:

#### Projection Formula:

If you have a vector \(\mathbf{r}\) and a line defined by a basis vector \(\mathbf{b}_1\), the projection of \(\mathbf{r}\) onto \(\mathbf{b}_1\) is given by:

$$
\text{Projection of } \mathbf{r} \text{ onto } \mathbf{b}_1 = \frac{\mathbf{r} \cdot \mathbf{b}_1}{\|\mathbf{b}_1\|^2} \mathbf{b}_1
$$

This allows you to measure how far along the line \(\mathbf{r}\) lies.

#### Noise and Fit Quality:

The "noise" in data, or how far points are from the line, is represented by the projection onto the orthogonal direction. Minimizing this noise improves the quality of your data fit.

# Week 3

## Types of matrix transformation

### Identity Matrix:

The identity matrix is a matrix that leaves any vector unchanged when multiplied by it. It is composed of basis vectors:

$$
\mathbf{I} = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}
$$

Multiplying it by a vector \( \begin{pmatrix} x \\ y \end{pmatrix} \) results in the same vector \( \begin{pmatrix} x \\ y \end{pmatrix} \). This matrix does nothing to the vector and is represented by \(\mathbf{I}\).

### Scaling Transformation:

If you have a matrix like:

$$
\begin{pmatrix} 3 & 0 \\ 0 & 2 \end{pmatrix}
$$

It scales the \(x\)-axis by a factor of 3 and the \(y\)-axis by a factor of 2. The result is a stretch of space, turning a unit square into a rectangle.

### Reflection/Flipping:

A matrix like:

$$
\begin{pmatrix} -1 & 0 \\ 0 & 1 \end{pmatrix}
$$

Flips the \(x\)-axis, changing the direction of the \(x\)-axis while leaving the \(y\)-axis unchanged.

A matrix like:

$$
\begin{pmatrix} -1 & 0 \\ 0 & -1 \end{pmatrix}
$$

Inverts both the \(x\)- and \(y\)-axes, flipping the entire space and changing the orientation (like a 180-degree rotation).

### Mirror Reflections:

A matrix like:

$$
\begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}
$$

Swaps the axes, akin to reflecting space over a diagonal mirror.

Other mirrors include:

$$
\begin{pmatrix} 0 & -1 \\ -1 & 0 \end{pmatrix}
$$

This reflects across another line, flipping the space similarly.

### Shearing Transformation:

A shear matrix like:

$$
\begin{pmatrix} 1 & 1 \\ 0 & 1 \end{pmatrix}
$$

Shifts one axis, causing space to "slant." This changes squares into parallelograms. The shear affects one axis but keeps the other unchanged.

### Rotation Transformation:

A rotation matrix, such as:

$$
\begin{pmatrix} 0 & 1 \\ -1 & 0 \end{pmatrix}
$$

Rotates the space by 90 degrees. The general formula for rotating by an angle \(\theta\) in 2D is:

$$
R_{\theta} = \begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix}
$$

This rotates the space counterclockwise by \(\theta\).

### Applications in Data Science:

These transformations are important in data science, especially when dealing with tasks like facial recognition or when images need to be rotated, stretched, or adjusted to align correctly. For example, rotating or scaling images to fit a standard orientation is a common task in preprocessing data for machine learning.

### Mathematical Concepts and Equations:

#### Identity Matrix:

$$
\mathbf{I} = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}
$$

This matrix does nothing to the vector it multiplies.

#### Scaling Matrix:

$$
\mathbf{S} = \begin{pmatrix} 3 & 0 \\ 0 & 2 \end{pmatrix}
$$

This scales the \(x\)-axis by 3 and the \(y\)-axis by 2.

#### Reflection/Flipping Matrix:

##### Flip \(x\)-axis:

$$
\mathbf{R} = \begin{pmatrix} -1 & 0 \\ 0 & 1 \end{pmatrix}
$$

##### Flip both axes:

$$
\mathbf{R} = \begin{pmatrix} -1 & 0 \\ 0 & -1 \end{pmatrix}
$$

#### Mirror Reflection:

##### Swap axes (mirror at 45 degrees):

$$
\mathbf{M} = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}
$$

##### Reflect across another axis:

$$
\mathbf{M} = \begin{pmatrix} 0 & -1 \\ -1 & 0 \end{pmatrix}
$$

#### Shear Transformation:

$$
\mathbf{Sh} = \begin{pmatrix} 1 & 1 \\ 0 & 1 \end{pmatrix}
$$

This shears space by moving the second axis.

#### Rotation Matrix:

##### General 2D rotation by an angle \(\theta\):

$$
R_{\theta} = \begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix}
$$

##### 90-degree rotation matrix:

$$
R_{90} = \begin{pmatrix} 0 & 1 \\ -1 & 0 \end{pmatrix}
$$

## Composition or combination of matrix transformations


### Matrix Composition:

When performing transformations, you can combine multiple transformations by applying them one after the other. This is known as composition of transformations.

For example, if you have two transformations \(A_1\) and \(A_2\), applying them sequentially (first \(A_1\), then \(A_2\)) to a vector \(\mathbf{r}\) gives the combined effect \(A_2 A_1\) on \(\mathbf{r}\).

#### Example with Transformations:

##### First transformation (A1):
A 90-degree anticlockwise rotation of the basis vectors:

$$
\mathbf{e}_1' = \begin{pmatrix} 0 \\ -1 \end{pmatrix}, \quad \mathbf{e}_2' = \begin{pmatrix} 1 \\ 0 \end{pmatrix}
$$

##### Second transformation (A2):
A vertical reflection, which flips the \(x\)-axis and leaves the \(y\)-axis unchanged:

$$
\mathbf{e}_1'' = \begin{pmatrix} -1 \\ 0 \end{pmatrix}, \quad \mathbf{e}_2'' = \begin

## Matrix Inverses

###  Gaussian elimination

#### Apples and Bananas Problem:

The problem involves two equations related to buying apples and bananas:

$$
2a + 3b = 8 \quad (\text{cost of 2 apples and 3 bananas is 8 euros})
$$

$$
10a + b = 13 \quad (\text{cost of 10 apples and 1 banana is 13 euros})
$$

This system of equations can be represented as:

$$
A \times r = s
$$

Where \( A \) is the matrix of coefficients, \( r \) is the vector of unknowns (cost of apples and bananas), and \( s \) is the result (total cost).

#### Matrix Inverse:

The goal is to find the inverse of matrix \( A \), denoted as \( A^{-1} \), which when multiplied by \( A \) gives the identity matrix:

$$
A^{-1} \times A = I
$$

By multiplying both sides of the equation \( A \times r = s \) by \( A^{-1} \), you can solve for \( r \):

$$
r = A^{-1} \times s
$$

Finding the inverse of \( A \) allows you to solve for \( r \) (the costs of apples and bananas).

#### Elimination Method (Gaussian Elimination):

Instead of finding the inverse directly, you can solve the system of equations using Gaussian elimination.

The goal of Gaussian elimination is to transform the coefficient matrix into a triangular matrix (Echelon form) and then use back substitution to solve for the variables.

Through row operations, you eliminate variables from the lower rows and eventually solve for each unknown in the system.


#### Efficiency of Gaussian Elimination:

This method is computationally efficient and works for any system of linear equations, even for large systems.

Transforming the Matrix:

Through Gaussian elimination, the matrix is transformed into the identity matrix, which is the goal of finding the inverse. This process is closely related to the method for finding the inverse of a matrix.

#### Key Concepts:

##### Matrix Inverse:

The matrix \( A^{-1} \) is the inverse of matrix \( A \), and multiplying a matrix by its inverse gives the identity matrix:

$$
A^{-1} \times A = I
$$

You can solve \( A \times r = s \) by multiplying both sides by \( A^{-1} \):

$$
r = A^{-1} \times s
$$

##### Gaussian Elimination:

Row operations are used to simplify the system of equations into a triangular matrix (Echelon form).

Back substitution is then used to find the solutions for the variables.

##### Echelon Form:

A matrix is in Echelon form when all entries below the leading diagonal are zero.

This form makes it easy to solve the system using back substitution.

### Computational Efficiency:

Gaussian elimination is a computationally efficient method for solving systems of linear equations and is often used in practice, especially when dealing with large systems.

#### Mathematical Equations:

##### System of Equations:

$$
A \times r = s
$$

Where \( A \) is the matrix of coefficients, \( r \) is the vector of unknowns, and \( s \) is the result vector.

##### Inverse of a Matrix:

$$
A^{-1} \times A = I
$$

Where \( I \) is the identity matrix, and multiplying by \( A^{-1} \) allows you to solve for \( r \):

$$
r = A^{-1} \times s
$$

##### Gaussian Elimination:

Row operations are used to simplify the system to Echelon form:

$$
\begin{pmatrix}
1 & 1 & 3 \\
0 & 1 & 1 \\
0 & 0 & -1
\end{pmatrix}
$$

Back substitution gives the solution:

$$
c = 2, \quad b = 4, \quad a = 5
$$

## Going from Gaussian elimination to finding the inverse matrix

### Apples and Bananas Problem:

The example starts with a system of linear equations describing the cost of apples and bananas:

$$
A \times r = s
$$

Where:
- \( A \) is the coefficient matrix,
- \( r \) is the vector of unknowns (cost of apples and bananas),
- \( s \) is the result vector (total cost).

To solve for \( r \), you can multiply both sides of the equation by the inverse of matrix \( A \), denoted \( A^{-1} \):

$$
r = A^{-1} \times s
$$

Finding the inverse of \( A \) allows you to solve for \( r \), the unknowns in the system of equations.

### Gaussian Elimination to Find the Inverse:

The process of **Gaussian elimination** is used to find the inverse of a matrix.

The goal is to transform matrix \( A \) into the identity matrix (a matrix with ones on the diagonal and zeros elsewhere) through row operations. At the same time, the matrix \( B \) (which will eventually become the inverse) is transformed.

This process involves performing row elimination to simplify the matrix, followed by back substitution to solve for the inverse.

### Step-by-Step Example:

A 3x3 matrix \( A \) is used as an example:

$$
A = \begin{pmatrix}
1 & 1 & 3 \\
1 & 2 & 4 \\
1 & 1 & 2
\end{pmatrix}
$$

The goal is to transform \( A \) into the identity matrix while performing the same row operations on an identity matrix \( I \), which will eventually give you the inverse matrix.

After performing the row operations and back substitution, the final matrix \( B \) is found, which is the inverse of \( A \).

### Computational Efficiency:

This method is computationally efficient, especially for smaller systems. For larger systems, there are faster methods, such as matrix decomposition.

In practical applications, a computational function (e.g., `inv(A)`) is used to automatically find the inverse of a matrix, which selects the best algorithm based on the matrix provided.

## General Method:

The process demonstrated shows how to solve a system of linear equations in the general case by finding the inverse of the matrix.

This method can be applied to any system of equations with any vector \( s \) on the right-hand side.

## Key Concepts and Definitions:

### Inverse of a Matrix:

The inverse of a matrix \( A \), denoted \( A^{-1} \), satisfies the equation:

$$
A \times A^{-1} = I
$$

Where \( I \) is the identity matrix.

#### Gaussian Elimination:

A method of solving systems of linear equations by transforming the coefficient matrix into an identity matrix through row operations. The same row operations are applied to the identity matrix to find the inverse.

### Row Operations:

Operations like adding, subtracting, or multiplying rows are used to simplify the matrix. The goal is to make all elements below the diagonal zero, creating a triangular matrix.

#### Back Substitution:

After row reduction, the matrix is in a form where you can backtrack (starting from the last row) to find the values of the unknowns in the system.

### Computational Methods:

For practical use, matrix inversion is often computed using specialized algorithms or functions (like `inv(A)`), which are optimized for computational efficiency.

### Mathematical Concepts:

#### System of Equations:

The system of equations is written as:

$$
A \times r = s
$$

Where:
- \( A \) is the matrix of coefficients,
- \( r \) is the vector of unknowns,
- \( s \) is the result vector.

#### Inverse of Matrix \( A \):

By multiplying both sides of the equation \( A \times r = s \) by \( A^{-1} \), we solve for \( r \):

$$
r = A^{-1} \times s
$$

#### Row Operations for Gaussian Elimination:

Through row operations, you transform the matrix \( A \) into the identity matrix while applying the same operations to the identity matrix to get \( A^{-1} \).

### Back Substitution:

Once the matrix is in triangular form, back substitution is used to solve for each unknown.

# Week 4

## Einstein's Summation Convention:

Einstein's Summation Convention is a shorthand notation used to express matrix operations concisely. It helps in reducing the need for explicit summation signs in equations.

For a matrix \( A \) with elements \( a_{ij} \) and another matrix \( B \) with elements \( b_{jk} \), the product of these matrices results in a new matrix \( AB \) where each element \( c_{ik} \) is calculated by multiplying corresponding elements of the row of matrix \( A \) with the column of matrix \( B \) and summing them up:

$$
c_{ik} = \sum_j a_{ij} b_{jk}
$$

The summation convention simplifies this by assuming the sum over \( j \) is understood and does not need to be explicitly written out, making it easier to express matrix multiplications and other operations.

## Multiplying Non-Square Matrices:

The convention allows for matrix multiplication even when the matrices are not square. If matrix \( A \) is \( 2 \times 3 \) and matrix \( B \) is \( 3 \times 4 \), their product is a \( 2 \times 4 \) matrix:

$$
AB = \text{a } 2 \times 4 \text{ matrix}
$$

This is possible as long as the number of columns in \( A \) matches the number of rows in \( B \), which is the key requirement for matrix multiplication.

## Dot Product and Matrix Multiplication:

The dot product between two vectors \( u \) and \( v \) can be viewed as a matrix multiplication. For two column vectors:

$$
u = \begin{pmatrix} u_1 \\ u_2 \\ \vdots \\ u_n \end{pmatrix}, \quad v = \begin{pmatrix} v_1 \\ v_2 \\ \vdots \\ v_n \end{pmatrix}
$$

Their dot product is:

$$
u \cdot v = \sum_i u_i v_i
$$

This is equivalent to matrix multiplication when one vector is treated as a row matrix and the other as a column matrix. This insight shows the equivalence between matrix multiplication and the dot product.

## Projection and Symmetry:

The geometric interpretation of the dot product is linked to projection. When a vector \( u \) is projected onto the unit vector \( e_1 \), the length of the projection is just \( u_1 \), the component of \( u \) along the \( e_1 \)-axis. This symmetry is true for any vector and axis.

The projection of one vector onto another is symmetric, which is reflected in the symmetry of the dot product. If you flip the vectors being dotted, the result remains the same:

$$
u \cdot e_1 = e_1 \cdot u
$$

## Connection Between Matrix Multiplication, Dot Product, and Projection:

Matrix multiplication can be viewed as the projection of a vector onto the columns of a matrix. Each element in the resulting matrix corresponds to the projection of one vector onto another.

## Geometrical Interpretation:

The video provides a geometric visualization of how matrix multiplication and the dot product relate to projections. The symmetry of the projection is a key insight that connects matrix multiplication with the dot product, revealing a deeper understanding of how transformations work.

## Key Concepts and Definitions:

### Einstein's Summation Convention:

A shorthand notation for representing matrix operations, especially matrix multiplication, without explicitly writing out the summation signs. If an index appears twice in a term, it is summed over all possible values of that index.

### Matrix Multiplication:

Matrix multiplication involves multiplying the rows of the first matrix with the columns of the second matrix and summing the results. The summation convention simplifies this process by assuming the sum over repeated indices.

### Dot Product as Matrix Multiplication:

The dot product between two vectors is equivalent to matrix multiplication when one vector is treated as a row matrix and the other as a column matrix.

### Projection:

The projection of one vector onto another is the length of the component of one vector along the direction of the other. This concept is used to explain the symmetry of the dot product.

### Symmetry of the Dot Product:

The dot product is symmetric, meaning:

$$
u \cdot v = v \cdot u
$$

This symmetry is geometrically interpreted as the invariance of the projection length regardless of the order in which the vectors are projected.

## Mathematical Equations:

### Matrix Multiplication in Einstein's Summation Convention:

$$
c_{ik} = \sum_j a_{ij} b_{jk}
$$

Where \( a_{ij} \) are the elements of matrix \( A \), and \( b_{jk} \) are the elements of matrix \( B \).

### Dot Product:

$$
u \cdot v = \sum_i u_i v_i
$$

This is equivalent to matrix multiplication when \( u \) is written as a row vector and \( v \) as a column vector.

### Projection of \( u \) onto \( e_1 \):

$$
u \cdot e_1 = u_1
$$

This represents the length of the projection of \( u \) onto the \( e_1 \)-axis.


## Panda’s World by the Vector Expressed in Panda's Coordinates

### Transformation Matrix:

The transformation matrix for Panda’s world in the speaker’s coordinates is:

$$
B = \begin{pmatrix} 3 & 1 \\ 1 & 1 \end{pmatrix}
$$

If we have a vector \( v = \begin{pmatrix} 3 \\ 2 \end{pmatrix} \) in Panda’s world, we can compute the corresponding vector in the speaker’s world by matrix multiplication.

### Inverse Transformation:

To perform the reverse transformation (from the speaker's world back into Panda’s world), we need to compute the inverse of the transformation matrix \( B \).

The inverse matrix \( B^{-1} \) is calculated as:

$$
B^{-1} = \frac{1}{\text{det}(B)} \begin{pmatrix} 1 & -1 \\ -1 & 3 \end{pmatrix}
$$

This reverse process involves multiplying the vector in the speaker's world by the inverse matrix to obtain the corresponding vector in Panda's world.

### Orthonormal Basis Vectors:

When the basis vectors are orthonormal (i.e., they are perpendicular and have unit length), the transformation becomes easier.

In the case of orthonormal vectors in Bear's world (with vectors \( v_1 = \frac{1}{\sqrt{2}} (1, 1) \) and \( v_2 = \frac{1}{\sqrt{2}} (1, -1) \)), the transformation is simplified, and the inverse matrix can be easily computed.

With orthonormal vectors, the transformation from one coordinate system to another can be performed using dot products rather than matrix multiplication.

### Dot Product and Projection:

The dot product can be used to project a vector onto the new basis vectors. When the basis vectors are orthonormal, you can directly compute the coordinates of the vector in the new coordinate system by taking the dot product of the vector with each of the new basis vectors.

This simplifies the process of transforming a vector, as you don’t need to perform matrix multiplication but can instead use the dot product to project the vector onto the new axes.

### Non-Orthonormal Basis Vectors:

If the basis vectors are not orthogonal (as in the earlier example with Panda’s world vectors \( (3, 1) \) and \( (1, 1) \)), the dot product will not work directly for projections. In such cases, matrix multiplication is required to perform the transformation, as the dot product would not yield the correct result for non-orthogonal vectors.

## Key Concepts and Definitions:

### Transformation Matrix:

A matrix that describes how to transform a vector from one coordinate system to another. The columns of the matrix are the new basis vectors expressed in the old coordinate system.

### Inverse Transformation:

The process of transforming a vector from one coordinate system back into another using the inverse of the transformation matrix. This allows you to reverse the effect of the original transformation.

### Orthonormal Basis Vectors:

Basis vectors that are both orthogonal (perpendicular) and normalized (have unit length). If the basis vectors are orthonormal, the transformation and its inverse are simplified, and projections can be computed using dot products.

### Dot Product and Projection:

The dot product of two vectors measures the degree to which they align. When the basis vectors are orthonormal, the dot product can be used to project a vector onto the new basis, simplifying the transformation process.

### Non-Orthonormal Basis Vectors:

If the basis vectors are not orthogonal, you cannot directly use dot products for transformations or projections. Instead, matrix multiplication is required.

## Mathematical Equations:

### Transformation Matrix:

For Panda’s world:

$$
B = \begin{pmatrix} 3 & 1 \\ 1 & 1 \end{pmatrix}
$$

This matrix transforms vectors from Panda’s world into the speaker’s world.

### Inverse Transformation:

The inverse of matrix \( B \):

$$
B^{-1} = \frac{1}{\text{det}(B)} \begin{pmatrix} 1 & -1 \\ -1 & 3 \end{pmatrix}
$$

The inverse matrix is used to transform vectors from the speaker's world back into Panda's world.

### Orthonormal Basis Transformation:

When the basis vectors are orthonormal:

$$
v_1 = \frac{1}{\sqrt{2}} \begin{pmatrix} 1 \\ 1 \end{pmatrix}, \quad v_2 = \frac{1}{\sqrt{2}} \begin{pmatrix} 1 \\ -1 \end{pmatrix}
$$

The transformation is simplified, and projections are computed using dot products.

### Dot Product for Projection:

The projection of a vector \( u \) onto a unit vector \( v \):

$$
\text{proj}_v u = u \cdot v
$$

This is used to transform a vector when the basis vectors are orthonormal.

## Transpose of a Matrix:

The transpose of a matrix is formed by swapping the rows and columns. For a matrix \( A \), the \( ij \)-th element of \( A^T \) (the transpose) is equal to the \( ji \)-th element of \( A \).

### Example:

If 

$$
A = \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}
$$

then:

$$
A^T = \begin{pmatrix} 1 & 3 \\ 2 & 4 \end{pmatrix}
$$

The transpose operation is useful in many mathematical and computational processes, including solving systems of equations and working with transformation matrices.

## Orthonormal Basis Vectors:

Orthonormal basis vectors are vectors that are both orthogonal (perpendicular) and normalized (unit length).

These vectors are particularly useful in defining coordinate systems for transformations because they simplify the computation of the inverse matrix and make certain operations easier to perform.

The transformation matrix composed of orthonormal basis vectors is called an **orthogonal matrix**.

## Properties of Orthogonal Matrices:

An orthogonal matrix has the property that its transpose is also its inverse:

$$
A^T = A^{-1}
$$

This means that multiplying an orthogonal matrix by its transpose results in the identity matrix:

$$
A^T \times A = I
$$

The determinant of an orthogonal matrix is always \( 1 \) or \( -1 \). A determinant of 1 indicates no reflection, while a determinant of -1 indicates a reflection (e.g., flipping or inversion of space).

## Geometric Interpretation:

When transforming vectors in data science, it's ideal to use an orthonormal basis set because it ensures that the transformation is reversible (due to the orthogonal matrix), the inverse is easy to compute, and projections can be computed with simple dot products.

The dot product of a vector with an orthonormal basis vector gives the projection of the vector onto that basis. This simplifies vector transformations and is computationally efficient.

## Inverse of an Orthonormal Matrix:

For an orthonormal basis set, the inverse matrix is the transpose of the transformation matrix. This simplifies many operations, especially when working with transformations in data science, machine learning, and computational geometry.

## Practical Use in Data Science:

In data science, using orthonormal basis vectors for transformations means that the computations involved in projecting data points, reversing transformations, and applying inverse operations are all straightforward and efficient.

Ensuring that the transformation matrix is orthogonal also guarantees that the transformation does not collapse the space, and the operation is reversible.

## Key Concepts and Definitions:

### Transpose of a Matrix:

The transpose of a matrix \( A \) is obtained by swapping its rows and columns:

$$
A^T = \text{Matrix with } a_{ij} \text{ becoming } a_{ji}
$$

The transpose is useful for operations involving changes in the coordinate system, transformations, and inverting matrices.

### Orthonormal Basis Vectors:

A set of vectors is orthonormal if:
- The vectors are **orthogonal**: Their dot product is zero.
- The vectors are **normalized**: Each vector has unit length (dot product with itself equals 1).

These vectors form the simplest and most convenient coordinate systems for transformations.

### Orthogonal Matrix:

A matrix whose columns (and rows) are orthonormal vectors. This matrix has the following properties:
- Its transpose is equal to its inverse:

  $$
  A^T = A^{-1}
  $$

- The determinant of an orthogonal matrix is either \( 1 \) or \( -1 \).

### Projection via Dot Product:

The dot product between a vector and an orthonormal basis vector gives the projection of the vector onto that basis vector. This is a simple and efficient way to compute vector projections, which is central to many transformations in data science.

### Inverse of an Orthonormal Matrix:

For an orthonormal matrix \( A \), its inverse is simply its transpose:

$$
A^{-1} = A^T
$$

This property makes working with orthogonal matrices computationally efficient.

## Mathematical Equations:

### Matrix Transpose:

The transpose of matrix \( A \) is given by:

$$
A^T = \begin{pmatrix} a_{11} & a_{12} & \cdots \\ a_{21} & a_{22} & \cdots \\ \vdots & \vdots & \ddots \end{pmatrix}
$$

Where the elements \( a_{ij} \) of \( A \) become \( a_{ji} \) in \( A^T \).

### Dot Product for Projection:

The dot product of vectors \( u \) and \( v \) is:

$$
u \cdot v = \sum_i u_i v_i
$$

This computes the projection of \( u \) onto \( v \), especially when \( v \) is a unit vector (orthonormal).

### Orthogonal Matrix Properties:

If \( A \) is an orthogonal matrix, then:

$$
A^T \times A = I
$$

Where \( I \) is the identity matrix. The determinant of \( A \) is either 1 or -1:

$$
\text{det}(A) = \pm 1
$$

### Inverse of an Orthonormal Matrix:

For an orthonormal matrix \( A \), its inverse is simply its transpose:

$$
A^{-1} = A^T
$$

## Gram-Schmidt Process

### Key Concepts:

#### Orthonormal Basis Vectors:

Orthonormal basis vectors are vectors that are **orthogonal** (perpendicular) to each other and **normalized** (having unit length). These vectors are essential because they simplify computations involving matrix transformations, inverses, and projections.

#### Gram-Schmidt Process:

The **Gram-Schmidt process** is a method for converting a set of linearly independent vectors into an orthonormal basis. This is done by normalizing the first vector and then subtracting the projections of the subsequent vectors onto the already obtained orthonormal vectors to ensure orthogonality.

### Steps of the Gram-Schmidt Process:

**Step 1**: Start with a set of linearly independent vectors \( v_1, v_2, \dots, v_n \).

**Step 2**: Normalize the first vector \( v_1 \) to obtain the first orthonormal vector \( e_1 \):

$$
e_1 = \frac{v_1}{\|v_1\|}
$$

**Step 3**: For each subsequent vector \( v_2, v_3, \dots \), subtract the projection onto the previous orthonormal vectors. This removes any components in the directions of previously obtained orthonormal vectors.

The projection of \( v_2 \) onto \( e_1 \) is computed as:

$$
\text{proj}_{e_1}(v_2) = (v_2 \cdot e_1) e_1
$$

Subtract this projection from \( v_2 \) to get a new vector \( u_2 \):

$$
u_2 = v_2 - \text{proj}_{e_1}(v_2)
$$

Normalize \( u_2 \) to get \( e_2 \):

$$
e_2 = \frac{u_2}{\|u_2\|}
$$

**Step 4**: Repeat the process for subsequent vectors \( v_3, v_4, \dots \), subtracting the projections onto the previous orthonormal vectors and normalizing the resulting vector.

### Why This Process Works:

The Gram-Schmidt process ensures that the resulting vectors are orthogonal to each other. By subtracting projections, you remove components along the directions of previously obtained vectors, ensuring orthogonality. The normalization step ensures that each vector has unit length.

### Benefits of Orthonormal Vectors:

- **Easier computations**: Orthonormal bases make matrix operations such as finding inverses and transposes much easier. The inverse of an orthogonal matrix is its transpose.
- **Projections**: When the basis vectors are orthonormal, projections can be computed simply using dot products.
- **Transformations**: Orthonormal basis vectors make transformations like rotations and scalings much simpler to handle mathematically.

### Application in Data Science:

In data science, orthonormal bases simplify many tasks, including data transformation, rotation, and scaling. By transforming data into an orthonormal basis, computations involving projections and matrix operations are more efficient.

---

## Example Process:

### Starting with Linearly Independent Vectors:

Assume we have a set of linearly independent vectors \( v_1, v_2, \dots, v_n \).

**Step 1**: Normalize \( v_1 \) to get \( e_1 \):

$$
e_1 = \frac{v_1}{\|v_1\|}
$$

**Step 2**: For \( v_2 \), subtract the projection of \( v_2 \) onto \( e_1 \), and normalize the result to get \( e_2 \):

$$
e_2 = \frac{v_2 - \text{proj}_{e_1}(v_2)}{\|v_2 - \text{proj}_{e_1}(v_2)\|}
$$

**Step 3**: Repeat for \( v_3, v_4, \dots \), subtracting projections and normalizing each new vector.

### Final Outcome:

After the process, you will have a set of orthonormal vectors \( e_1, e_2, \dots, e_n \) that form an orthonormal basis.

---

## Mathematical Notation:

### Projection of a Vector \( v \) onto a Vector \( u \):

$$
\text{proj}_{u}(v) = \frac{v \cdot u}{\|u\|^2} u
$$

### Gram-Schmidt Process:

Normalize \( v_1 \):

$$
e_1 = \frac{v_1}{\|v_1\|}
$$

Subtract projection to make \( v_2 \) orthogonal to \( e_1 \):

$$
u_2 = v_2 - \text{proj}_{e_1}(v_2)
$$

Normalize \( u_2 \) to get \( e_2 \):

$$
e_2 = \frac{u_2}{\|u_2\|}
$$

Repeat for \( v_3, v_4, \dots \).

## Reflection of a Vector in a Plane using Gram-Schmidt Process

### Key Concepts:

#### Vector Reflection in a Plane:

This process explores how to reflect a vector in a plane that is not aligned with the standard coordinate axes. The steps involve transforming the vector into the new basis of the plane, performing the reflection in the plane, and then transforming it back into the original coordinate system.

#### Gram-Schmidt Process:

The **Gram-Schmidt process** is used to generate an orthonormal basis from a set of linearly independent vectors. By transforming the original vectors into orthonormal vectors, matrix operations (like reflection) become much easier to handle.

### Steps for Reflection:

**Step 1**: Define the plane's vectors, where two vectors lie in the plane and one vector is normal to the plane.

**Step 2**: Use the Gram-Schmidt process to create an orthonormal basis for the plane and its normal.

**Step 3**: Express the vector to be reflected in the new orthonormal basis.

**Step 4**: Perform the reflection by manipulating the vector components. The components parallel to the plane stay the same, while the component perpendicular to the plane is inverted.

**Step 5**: Convert the transformed vector back to the original basis by using the inverse of the transformation matrix.

### Transformation Matrix:

The matrix that describes the transformation in the new basis is **orthogonal** because the basis vectors are orthonormal. An orthogonal matrix is particularly useful because its transpose is also its inverse, making the reflection operation computationally straightforward.

### Use of Projections:

The reflection is computed by projecting the vector onto the plane's basis, then adjusting the components to reflect across the plane.

### Computational Steps:

The process of reflecting a vector is reduced to matrix multiplication, where the vector is first transformed into the new basis, the transformation (reflection) is applied in the new basis, and then the result is transformed back into the original coordinate system.

### Final Result:

After performing the transformations and reflection, the vector is reflected in the plane, and the computation yields the transformed vector, which is easier to obtain using matrix operations rather than trigonometric methods.

---

## Key Mathematical Tools:

### Gram-Schmidt Process:

- Normalizes vectors and ensures orthogonality, which makes subsequent operations (like reflection) easier to compute.

### Transformation Matrix:

- Involves the matrix of the orthonormal vectors that define the plane, making it possible to convert the vector into the new coordinate system of the plane.

### Matrix Operations:

- Using matrix multiplication, transpose, and inverse simplifies complex geometric transformations.

### Reflection Matrix:

- The reflection matrix in the new basis is particularly simple because the transformation involves just flipping the component along the normal to the plane.

### Projections:

- The projections of vectors onto the plane and its normal vector help break down the transformation into manageable parts.

---

## Example of Reflection Process:

### Starting Vectors:

The plane's basis vectors are:

$$
v_1 = (1, 1, 1), \quad v_2 = (2, 0, 1)
$$

The normal vector is:

$$
v_3 = (3, 1, -1)
$$

**Step 1**: Apply the Gram-Schmidt process to normalize \( v_1 \) and make it orthogonal to \( v_2 \) and \( v_3 \). The resulting orthonormal basis vectors are \( e_1, e_2, e_3 \), with \( e_3 \) being normal to the plane.

**Step 2**: The vector to be reflected is:

$$
r = (2, 3, 5)
$$

This is expressed in terms of the orthonormal basis \( e_1, e_2, e_3 \).

**Step 3**: The reflection is performed in the plane by flipping the component of \( r \) along \( e_3 \) (the normal to the plane) while leaving the components along \( e_1 \) and \( e_2 \) unchanged.

**Step 4**: The final reflected vector is transformed back to the original coordinate system.

---

## Mathematical Notation:

### Projection of a Vector \( v \) onto a Vector \( u \):

$$
\text{proj}_u(v) = \frac{v \cdot u}{\|u\|^2} u
$$

### Gram-Schmidt Process:

**Normalize \( v_1 \):**

$$
e_1 = \frac{v_1}{\|v_1\|}
$$

**Subtract projection to make \( v_2 \) orthogonal to \( e_1 \):**

$$
u_2 = v_2 - \text{proj}_{e_1}(v_2)
$$

**Normalize \( u_2 \) to get \( e_2 \):**

$$
e_2 = \frac{u_2}{\|u_2\|}
$$

**Repeat for \( v_3, v_4, \dots \).**

# Week 5
### Eigenvectors and Eigenvalues (Mathematical Formulation)

Given a linear transformation represented by a square matrix \( A \), an eigenvector \( v \) and its corresponding eigenvalue \( \lambda \) satisfy the following equation:

\[
A v = \lambda v
\]

Where:

- \( A \) is the transformation matrix (such as a scaling, rotation, or shear matrix),
- \( v \) is the eigenvector, which represents a vector that remains along its original direction under the transformation,
- \( \lambda \) is the eigenvalue, which tells you by how much the eigenvector is scaled during the transformation.

#### Geometric Interpretation

##### Scaling:
In the case of scaling, certain vectors (e.g., the horizontal and vertical vectors) remain in the same direction but may change length.

For example, in a vertical scaling by a factor of 2:
- The horizontal eigenvector stays the same (\( \lambda = 1 \)),
- The vertical eigenvector is scaled by a factor of 2 (\( \lambda = 2 \)).

##### Shearing:
In a shear transformation, some vectors, like the horizontal vector, will remain unchanged in direction (i.e., the horizontal eigenvector), while others will shift.

Only vectors lying along certain lines (like the green horizontal line) are eigenvectors.

##### Rotation:
Rotation does not have any eigenvectors because all vectors are rotated off their original span. No vector remains in the same direction after the rotation.

#### Characteristic Equation

To compute the eigenvalues and eigenvectors for a transformation matrix \( A \), we need to solve the characteristic equation:

\[
\text{det}(A - \lambda I) = 0
\]

Where:

- \( A \) is the matrix of the transformation,
- \( \lambda \) is the eigenvalue,
- \( I \) is the identity matrix,
- \( \text{det}(A - \lambda I) \) is the determinant of \( (A - \lambda I) \).

Solving this equation gives the eigenvalues \( \lambda \), and then you substitute each eigenvalue back into the equation \( (A - \lambda I) v = 0 \) to find the corresponding eigenvectors.

#### Key Points

- Eigenvectors represent directions that remain unchanged under the linear transformation, but may be scaled (i.e., stretched or shrunk).
- Eigenvalues represent how much the eigenvectors are scaled during the transformation.
- The geometric concept of eigenvectors and eigenvalues can be understood in terms of how shapes (like squares or vectors) are distorted under transformations such as scaling, shearing, and rotation.
- For scaling transformations, the eigenvectors are the directions that are scaled by the eigenvalues.
- Rotation has no eigenvectors, as all vectors are rotated and no vector remains in the same direction.

This is the geometric and algebraic foundation of eigenvectors and eigenvalues, and it applies to transformations in higher dimensions (such as 3D transformations), as well.

## 1. Eigenvectors and Eigenvalues Overview:

**Eigenvectors**: Vectors that remain along the same span both before and after a linear transformation.

**Eigenvalues**: The scalar values that indicate how much the eigenvectors are stretched or shrunk during the transformation.

---

# 2. Special Cases:

#### Uniform Scaling (Scaling by the same amount in all directions):
- Every vector is an eigenvector in this case.

##### Mathematical Explanation:
For a transformation \( T \), if \( T(v) = \lambda v \), then for uniform scaling, all vectors \( v \) satisfy this, with the eigenvalue \( \lambda \) being the scaling factor.

---

#### 180-degree Rotation:
- **Eigenvectors**: Vectors along the axes of rotation remain eigenvectors.
- **Eigenvalue**: The eigenvalue is \( \lambda = -1 \), indicating that vectors are flipped in direction but remain of the same length.

##### Mathematical Explanation:
Rotation matrix for a 180-degree rotation can be represented as:

\[
R = \begin{pmatrix}
-1 & 0 \\
0 & -1
\end{pmatrix}
\]

Eigenvectors for this transformation will have eigenvalue \( -1 \), meaning they are inverted in direction.

---

#### Horizontal Shear and Vertical Scaling Combination:
- **Eigenvectors**: In this case, only certain vectors (e.g., horizontal vectors) remain as eigenvectors.

##### Mathematical Explanation:
A horizontal shear may stretch or distort vectors horizontally, but vectors along the shear direction still satisfy the eigenvector condition.

Scaling factor: For shear, eigenvalues might not always be straightforward but are identifiable.

---

# 3. Eigenvectors in 3D:

#### Scaling and Shear in 3D:
- Similar behavior to 2D scaling and shear.

#### Rotation in 3D:
- **Eigenvector**: The axis of rotation remains unchanged while other vectors rotate around it.

##### Mathematical Explanation:
For a 3D rotation matrix, the axis of rotation will correspond to an eigenvector. This means the vector does not change direction, only other components rotate.

##### Example:
In a rotation matrix \( R \), the vector \( v \) for which \( Rv = v \) is the eigenvector, and its eigenvalue will be 1.

---

## 4. Challenges in 3D and Beyond:
- Eigenvectors in higher dimensions (such as 3D or more) can be harder to visualize but follow similar principles.
- **Physical Interpretation**: For 3D rotations, the eigenvector represents the axis around which rotation occurs.

---

## Key Takeaways:
- Eigenvectors maintain direction in a transformation (scaling, rotation, shear, etc.), and eigenvalues tell us how much the eigenvectors stretch or shrink.
- In 3D, finding eigenvectors can provide insights into the geometry of rotations (e.g., identifying the axis of rotation).
- Eigen theory is foundational in fields like machine learning, especially in high-dimensional spaces, and needs formal mathematical tools for higher dimensions.

# 1. Formalizing Eigenvectors and Eigenvalues:

### Eigenvector Definition:
If a transformation \( A \) has eigenvectors, these vectors \( x \) stay in the same span after transformation, though they may change length or direction.

The relationship can be expressed as:

\[
A x = \lambda x
\]

Where:
- \( A \) is the transformation matrix,
- \( x \) is the eigenvector, and
- \( \lambda \) is the eigenvalue (scalar factor by which the eigenvector is stretched or compressed).

---

# 2. Algebraic Approach:

### Rewriting the Equation:
We can rewrite the equation as:

\[
(A - \lambda I) x = 0
\]

Where \( I \) is the identity matrix (same size as \( A \)).

This represents the fact that applying \( A - \lambda I \) to \( x \) results in the zero vector.

### Solving for Eigenvalues:
For non-trivial solutions, the determinant of the matrix \( A - \lambda I \) must be zero:

\[
\text{det}(A - \lambda I) = 0
\]

This leads to the characteristic equation, which is a polynomial whose solutions give the eigenvalues.

---

# 3. Example: 2x2 Transformation:

### Given Transformation Matrix:
\[
A = \begin{pmatrix}
a & b \\
c & d
\end{pmatrix}
\]

We find the eigenvalues by calculating the determinant:

\[
\text{det} \left( \begin{pmatrix} a & b \\ c & d \end{pmatrix} - \lambda \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix} \right) = 0
\]

This gives the characteristic polynomial:

\[
\lambda^2 - (a+d)\lambda + (ad - bc) = 0
\]

The eigenvalues \( \lambda \) are the solutions to this quadratic equation.

---

# 4. Example: Vertical Scaling:

### Transformation Matrix for Vertical Scaling by 2:
\[
A = \begin{pmatrix}
1 & 0 \\
0 & 2
\end{pmatrix}
\]

Applying the eigenvalue equation:

\[
\text{det}(A - \lambda I) = \text{det} \left( \begin{pmatrix} 1 - \lambda & 0 \\ 0 & 2 - \lambda \end{pmatrix} \right) = 0
\]

Solving the determinant gives the eigenvalues \( \lambda = 1 \) and \( \lambda = 2 \).

#### Eigenvectors for \( \lambda = 1 \):
\[
(A - 1I) x = 0 \quad \Rightarrow \quad x = \begin{pmatrix} x_1 \\ 0 \end{pmatrix}
\]

Any horizontal vector is an eigenvector for \( \lambda = 1 \).

#### Eigenvectors for \( \lambda = 2 \):
\[
(A - 2I) x = 0 \quad \Rightarrow \quad x = \begin{pmatrix} 0 \\ x_2 \end{pmatrix}
\]

Any vertical vector is an eigenvector for \( \lambda = 2 \).

---

# 5. Example: 90-Degree Rotation (No Real Eigenvectors):

### Rotation Matrix for 90-degree Anti-clockwise:
\[
A = \begin{pmatrix}
0 & -1 \\
1 & 0
\end{pmatrix}
\]

The determinant of \( A - \lambda I \) gives:

\[
\text{det} \left( \begin{pmatrix} -\lambda & -1 \\ 1 & -\lambda \end{pmatrix} \right) = \lambda^2 + 1 = 0
\]

This has no real solutions, meaning there are no real eigenvectors for this rotation.

Complex eigenvectors can be calculated, but they are not required here.

---

# 6. Practical Considerations:

### Computational Efficiency:
Calculating eigenvectors by hand is impractical for large matrices.

Computers use iterative numerical methods for high-dimensional problems (e.g., 100-dimensional matrices).

Conceptual understanding of eigenvectors is more valuable than manually solving them, as computational tools handle the heavy lifting.


# Diagonalization and Eigenbasis:

## 1. Concept of Diagonalization:
Diagonalization is a powerful method that simplifies matrix operations by converting a matrix into a diagonal form, making repeated matrix multiplications more efficient.

**Eigenbasis** is a basis formed by the eigenvectors of a matrix. When a matrix is diagonalized, the transformation matrix becomes simpler to handle.

---

## 2. Problem of Repeated Matrix Multiplication:
Consider a transformation matrix \( T \) applied to a particle's position vector \( v_0 \). The position after multiple time steps can be expressed as:

\[
v_n = T^n v_0
\]

If \( n \) is large (e.g., millions of time steps), directly multiplying \( T \) by itself multiple times is computationally expensive. Diagonalization helps to simplify this process.

---

## 3. Diagonal Matrices:
A diagonal matrix is a matrix where all the off-diagonal terms are zero. For example, the matrix:

\[
D = \begin{pmatrix}
\lambda_1 & 0 \\
0 & \lambda_2
\end{pmatrix}
\]

allows easy exponentiation:

\[
D^n = \begin{pmatrix}
\lambda_1^n & 0 \\
0 & \lambda_2^n
\end{pmatrix}
\]

---

## 4. Eigenbasis and Diagonalization:
**Eigenbasis**: To diagonalize a matrix \( T \), we convert to a basis where the transformation matrix is diagonal. This is achieved by using the matrix \( C \), which consists of the eigenvectors of \( T \).

The diagonal matrix \( D \) contains the eigenvalues of \( T \).

The diagonalization relation is:

\[
T = C D C^{-1}
\]

Where:
- \( C \) is the matrix of eigenvectors,
- \( D \) is the diagonal matrix of eigenvalues.

---

## 5. Efficient Power Computation with Diagonalization:
To compute powers of \( T \) (e.g., \( T^n \)), the formula becomes:

\[
T^n = C D^n C^{-1}
\]

This allows us to compute \( D^n \) (which is simple since \( D \) is diagonal) and then transform back using \( C \).

---

## 6. Example: 2D Matrix Transformation:
Consider the transformation matrix:

\[
T = \begin{pmatrix}
1 & 1 \\
0 & 2
\end{pmatrix}
\]

### Eigenvectors and Eigenvalues:
- \( \lambda = 1 \) gives eigenvector \( \begin{pmatrix} 1 \\ 0 \end{pmatrix} \)
- \( \lambda = 2 \) gives eigenvector \( \begin{pmatrix} 1 \\ 1 \end{pmatrix} \)

### Transformation and Eigenbasis:
After applying \( T \), the transformation is represented in the eigenbasis.

The conversion matrix \( C \) is:

\[
C = \begin{pmatrix} 1 & 1 \\ 0 & 1 \end{pmatrix}
\]

Its inverse \( C^{-1} \) is:

\[
C^{-1} = \begin{pmatrix} 1 & -1 \\ 0 & 1 \end{pmatrix}
\]

### Applying Diagonalization:
\[
T^2 = C D^2 C^{-1}
\]

Where:

\[
D = \begin{pmatrix} 1 & 0 \\ 0 & 2 \end{pmatrix}
\]

---

## 7. General Process:

### Diagonalization Steps:
1. Find the eigenvectors and eigenvalues of \( T \).
2. Construct the matrix \( C \) with eigenvectors as columns.
3. Construct the diagonal matrix \( D \) with eigenvalues on the diagonal.
4. Compute powers of \( T \) using \( T^n = C D^n C^{-1} \).

# 1. Introduction to PageRank:
PageRank is an algorithm developed by Google in 1998 by Larry Page and Sergey Brin to rank websites based on their importance, using the links between webpages.

The algorithm assumes that the importance of a webpage is determined by the number and quality of links to it.

---

# 2. Concept of Procrastinating Pat:
**Procrastinating Pat** is an imaginary user who randomly clicks links to simulate browsing behavior.

The probability of Pat visiting a page can be represented as a vector, where each element corresponds to the likelihood of visiting a particular webpage based on the links.

---

# 3. Link Matrix (L):
The link matrix \( L \) represents the probabilities of transitioning from one webpage to another.

Each column in the matrix represents the links from a webpage, and the values in the column are normalized by the total number of links on that page.

For example, the link vector for webpage A (if it links to B, C, and D) would be:

\[
L_A = \left( 0, \frac{1}{3}, \frac{1}{3}, \frac{1}{3} \right)
\]

This process is done for all webpages to form a square link matrix.

---

# 4. PageRank Calculation:
The rank vector \( r \) represents the importance of each webpage.

The rank of page A is based on the rank of pages that link to A, weighted by their link probabilities.

The formula for page A's rank is:

\[
r_A = \sum_{j=1}^{n} L_{A,j} \cdot r_j
\]

Where:
- \( L_{A,j} \) is the link probability from page \( j \) to page A, and
- \( r_j \) is the rank of page \( j \).

---

# 5. Matrix Formulation:
The rank calculation for all pages can be written as a matrix equation:

\[
r = L \cdot r
\]

Initially, all pages are assumed to have equal rank, so:

\[
r = \left( \frac{1}{n}, \frac{1}{n}, \dots, \frac{1}{n} \right)
\]

where \( n \) is the total number of pages.

---

# 6. Iterative Process:
To solve \( r = L \cdot r \), the iterative method is used:

\[
r_{i+1} = L \cdot r_i
\]

Repeated multiplication of the rank vector \( r \) by the link matrix \( L \) gradually converges to the final rank vector, which represents the steady-state of PageRank.

---

# 7. Convergence:
The iterative process will converge after a number of iterations, and the rank vector \( r \) will stabilize. The final rank vector shows the relative importance of each webpage.

The result might look like:

\[
r = (0.12, 0.24, 0.24, 0.40)
\]

Page D is the most important (40%), while page A is the least important (12%).

---

# 8. Eigenvalue and Eigenvector Interpretation:
The equation \( r = L \cdot r \) is equivalent to finding an eigenvector of matrix \( L \) with eigenvalue 1. The rank vector \( r \) is the eigenvector of the link matrix \( L \), which describes the steady-state distribution of "importance" across the network.

---

# 9. Power Method:
The **Power Method** is an iterative technique for finding the dominant eigenvector. It works well for PageRank because:
- The eigenvalue corresponding to the rank vector is 1.
- The link matrix is typically sparse (most links are zero), making matrix multiplications efficient.

---

# 10. Damping Factor (d):
The **damping factor** \( d \) is introduced to model the probability that a user might randomly jump to any page instead of following links.

The updated rank formula is:

\[
r_{i+1} = d \cdot L \cdot r_i + \frac{1 - d}{n}
\]

\( d \) is typically between 0 and 1. It prevents the algorithm from focusing solely on the link structure by introducing randomness.

---

# 11. Application to Large Networks:
For large networks (e.g., billions of webpages), the power method is still effective because the link matrix is sparse, and sparse matrix algorithms allow efficient computation.

















