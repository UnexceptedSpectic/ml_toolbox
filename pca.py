import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json


class PCA:

    def __init__(self, data_file_path, transpose_input=False):
        self.x = pd.read_csv(data_file_path)
        if transpose_input:
            self.x = self.x.transpose()
        if isinstance(self.x.iloc[0, 0], str):
            # remove first row if non numeric
            self.x = self.x.iloc[1:, :]
        self.x_cov = None
        self.x_cov_eigenvalues = None
        self.x_cov_eigenvectors = None
        self.projected_x = None

    def mean_center(self):
        self.x = self.x - self.x.mean()
        self.x = self.x.astype(float)

    def set_x_cov(self):
        self.x_cov = pd.DataFrame(np.cov(self.x, rowvar=False))

    def eigendecompose_x_cov(self):
        self.x_cov_eigenvalues, self.x_cov_eigenvectors = np.linalg.eig(self.x_cov)
        zero_matrix = np.zeros(np.repeat(len(self.x_cov_eigenvalues), 2), float)
        np.fill_diagonal(zero_matrix, np.real(self.x_cov_eigenvalues))
        self.x_cov_eigenvalues = pd.DataFrame(zero_matrix)
        self.x_cov_eigenvectors = pd.DataFrame(np.real(self.x_cov_eigenvectors))

    def project_x(self):
        self.projected_x = pd.DataFrame(np.dot(self.x, self.x_cov_eigenvectors))

    def project_onto_pc(self, pc_number):
        ascending_indexes = np.argsort(np.diagonal(self.x_cov_eigenvalues))[::-1]
        pc = self.x_cov_eigenvectors.iloc[:, ascending_indexes[pc_number-1]]
        return np.dot(self.x, pc)

    def get_pc_variance_info(self):
        pc_var = np.sort(np.diagonal(self.x_cov_eigenvalues))[::-1]
        pc_var_info = {}
        for i in range(len(pc_var)):
            pc_var_info['pc' + str(i + 1)] = {
                'variance': round(pc_var[i], 2),
                'percentage': round(pc_var[i]/sum(pc_var) * 100, 2)
            }
        return pc_var_info

    def plot_x(self, col1_number, col2_number):
        fig, ax = plt.subplots(figsize=(10, 6))
        data = self.x
        ax.scatter(x=data.iloc[:, col1_number - 1], y=data.iloc[:, col2_number - 1])
        plt.xlabel('Var' + str(col1_number))
        plt.ylabel('Var' + str(col2_number))
        plt.title('Input Data')
        fig.show()

    def plot_scree(self):
        fig, ax = plt.subplots(figsize=(10, 6))
        data = self.get_pc_variance_info()
        variances = list(map(lambda data_key: data[data_key].get('variance'), list(data.keys())))
        ax.scatter(x=range(1, len(data.keys()) + 1), y=variances)
        plt.xlabel('Principal Component')
        plt.xticks([])
        plt.ylabel('Variance (Eigenvalues)')
        plt.title('Scree Plot')
        fig.show()

    def plot_scores(self, data):
        fig, ax = plt.subplots(figsize=(10, 6))
        if len(np.shape(data)) < 2:
            x = data
            y = np.repeat(0, np.shape(data)[0])
        else:
            d = self.sort_by_var(data)[1]
            x = d.iloc[:, 0]
            y = d.iloc[:, 1]
        ax.scatter(x=x, y=y)
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.title('Scores Plot, PC Space')
        fig.show()

    def sort_by_var(self, pc_df):
        ascending_indexes = np.argsort(np.diagonal(self.x_cov_eigenvalues))[::-1]
        return self.x_cov_eigenvalues.iloc[:, ascending_indexes], pc_df.iloc[:, ascending_indexes]

    def plot_loadings(self):
        fig, ax = plt.subplots(figsize=(10, 6))
        sorted_pcs = self.sort_by_var(self.x_cov_eigenvectors)[1]
        ax.scatter(x=sorted_pcs.iloc[:, 0], y=sorted_pcs.iloc[:, 1])
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.title('Loadings Plot')
        for row_ind in range(np.shape(sorted_pcs)[0]):
            ax.text(sorted_pcs.iloc[row_ind, 0], sorted_pcs.iloc[row_ind, 1], 'var' + str(row_ind + 1))
        fig.show()

    def fit_transform(self):
        self.mean_center()
        self.set_x_cov()
        self.eigendecompose_x_cov()
        self.project_x()
        self.x_cov_eigenvalues, self.x_cov_eigenvectors = self.sort_by_var(self.x_cov_eigenvectors)

    def head_x(self):
        print('Data head:')
        print(self.x.head(5))


if __name__ == '__main__':
    print('Testing the PCA class.')
    file_path = './data/hw2_prob3.csv'
    pca = PCA(file_path)
    print('Opened ' + file_path)
    pca.head_x()
    print('Mean centered data:')
    pca.mean_center()
    pca.head_x()
    print('Data shape:')
    print(np.shape(pca.x))
    pca.set_x_cov()
    print('Data cov matrix')
    print(pca.x_cov.head(5))
    print('Verify that cov matrix is nxn, where n is number of variables in data')
    print(np.shape(pca.x_cov))
    print('Eigendecompose x covariance matrix')
    pca.eigendecompose_x_cov()
    print('x covariance matrix eigenvalues')
    print(pca.x_cov_eigenvalues)
    print('x covariance matrix eigenvectors')
    print(pca.x_cov_eigenvectors)
    print('Projecting x into eigenvector space')
    pca.project_x()
    print('Weights/Loadings used:')
    print(pca.x_cov_eigenvectors)
    print('PC coordinates/Scores head:')
    print(pca.projected_x.head(5))
    print('Verify that the projected data shape equals that of the input data')
    print(np.shape(pca.projected_x))
    print('Principal component variance info:')
    print(json.dumps(pca.get_pc_variance_info(), indent=4))
    print('Verify that the projected data covariance matrix [cov(y)] has ~zeros off-diagonal')
    print(pd.DataFrame(np.cov(pca.projected_x, rowvar=False)))
    print('Verify that P^T*cov(x)*P ~= cov(y)')
    print(pd.DataFrame(np.dot(np.dot(pca.x_cov_eigenvectors.values.transpose(), pca.x_cov.values), pca.x_cov_eigenvectors.values)))
    print('See input data plot')
    pca.plot_x(col1_number=1, col2_number=2)
    print('Separation in the raw data can be slightly seen')
    print('#3')
    print('See the scores plot for the two variables with the highest variance')
    pca.plot_scores(pca.projected_x)
    print('The raw data in PC space is clearly separated')
    print('See raw data projected only onto single PC with greatest variance')
    pca.plot_scores(pca.project_onto_pc(pc_number=1))
    print('The data is no longer clearly separated when projected onto the first principal component as although most '
          'of the variance was in PC1, PC2 showed the greatest separation. Therefore, it is not always possible to '
          'reduce dimensionality without either losing separation information or a large quantity of variance.')
    print('The variance of the projections onto pc1 and pc2 corresponds to the eigenvalues of the the cov(x) matrix. '
          'This suggests that variance is preserved during projection from one space to another.')
    print('#4')
    pca2 = PCA('./data/hw2_prob4.csv', transpose_input=True)
    pca2.fit_transform()
    print('See scree plot')
    pca2.plot_scree()
    var_info = pca2.get_pc_variance_info()
    print('pc1 variance')
    print(json.dumps(var_info.get('pc1'), indent=4))
    print('pc2 variance')
    print(json.dumps(var_info.get('pc2'), indent=4))
    print('See scores plot')
    pca2.plot_scores(pca2.projected_x)
    pca2.plot_loadings()


