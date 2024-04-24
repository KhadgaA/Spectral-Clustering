# Image Segmentation

> Alli Khadga Jyoth - M23CSA003

## Ratio Cut Segmentaion

### Steps:

1. Convert the image into a Graph. This can be done by simply reshaping the image into a vector of `N = height*width` nodes.
2. Compute the affinity of each node of the graph/image w.r.to all other nodes/pixels. Therefore `Affinity (A) = (N , N)` matrix. We can either use the euclidean based affinity : $ e^{||I(i) - I(j)||^2_2} $ or we can use the Cosine Similarity: $ A_{i,j} = \frac{I(i) \cdot I(j)}{||I(i)|| \cdot ||I(j)||}$.
3. Refine Affinity Matrix
4. Compute the Degree Matrix. $D_{i} = \sum_{j = 1}^N A_{ij}$
5. Compute the Laplacian of the graph. `L = D - A`. Normalize Laplacian Matrix.
6. Compute the eigenvalues and eigenvectors of the Laplacian matrix.
7. Take eigenvectors corresponding to $K^{th}$ smallest eigenvalue.
8. Perform Clustering on K.

> Please See Appendix Section for exact details.

### Segmentation Results

#### Image 1:

![1713907993621](image/M23CSA003/1713907993621.png)

![1713907884042](image/M23CSA003/1713907884042.png)![1713907894137](image/M23CSA003/1713907894137.png)![1713907944518](image/M23CSA003/1713907944518.png)![1713907949725](image/M23CSA003/1713907949725.png)![1713907955081](image/M23CSA003/1713907955081.png)

#### Image 2:

![1713908009339](image/M23CSA003/1713908009339.png)

![1713908027281](image/M23CSA003/1713908027281.png)![1713908031799](image/M23CSA003/1713908031799.png)![1713908036127](image/M23CSA003/1713908036127.png)![1713908040339](image/M23CSA003/1713908040339.png)![1713908044462](image/M23CSA003/1713908044462.png)

## KMeans Clustering:

![1713908114710](image/M23CSA003/1713908114710.png)![1713908120214](image/M23CSA003/1713908120214.png)

# Appendix

## Affine Matrix Refinement

### Steps:

1. Blurring : Perform Gaussian Blurring on Affinity Matrix to smoothout the Affinities
2. Thresholding: Perform thresholding on the matrix to remove outliers, and constraint the Laplacian Matrix Range
3. Symmetrize: Since Blurring followed by Thresholding is non linear operation and non-symmetric. It will messup the symmetric property of Laplacian Matrix. Therefore we Re-Symmetrize the Laplacian.
4. Diffussion: Its done as $XX^T$
5. Normalization: Normalize each row of Laplacian, since each row of Laplacian is a different Vertex of the Graph.

   ![refinement](https://raw.githubusercontent.com/wq2012/SpectralCluster/master/resources/refinement.png)

The Parameters for each of the operations is dependent on the problem at hand. 

# References

```bibtex
@misc{wang2022speaker,
    title={Speaker Diarization with LSTM},
      author={Quan Wang and Carlton Downey and Li Wan and Philip Andrew Mansfield and Ignacio Lopez Moreno},
      year={2022},
      eprint={1710.10468},
      archivePrefix={arXiv},
      primaryClass={eess.AS},
      url = {https://arxiv.org/abs/1710.10468},
      github = {https://github.com/wq2012/SpectralCluster}
}
```
