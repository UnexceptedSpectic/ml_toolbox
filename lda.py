import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import pca

# Two-class LDA
class LDA:

    def __init__(self, labels=None, data=None, data_file_path=None, transpose_input=False):
        if data is not None:
            self.x = pd.DataFrame(data)
        elif data_file_path:
            self.x = pd.read_csv(data_file_path)
            # remove first row if non numeric
            if isinstance(self.x.iloc[0, 0], str):
                self.x = self.x.iloc[1:, :]
        else:
            print('Error: Either x or data_file_path must be specified.')
            return
        if labels is None:
            if data_file_path:
                self.labels = np.array(self.x['label'])
            else:
                print('Error: Labels must be specified.')
                return
        else:
            self.labels = labels
        if transpose_input:
            self.x = self.x.transpose()
        self.x_c1 = None
        self.x_c2 = None
        self.x_c1_mean = None
        self.x_c2_mean = None
        self.x_scatter_within = None
        self.x_scatter_between = None
        self.w = None
        self.y_c1 = None
        self.y_c2 = None
        self.skl_labels = None
        self.skl = None
        self.skl_y = None

    def head_x(self):
        print('Data head:')
        print(self.x.head(5))

    def label_summary(self):
        print('Target classes:')
        print(set(self.labels))

    # Define clusters based on class labels
    def define_clusters(self, c1_label, c2_label):
        # setosa
        self.x_c1 = self.x.iloc[self.labels == c1_label, :]
        # viriginica
        self.x_c2 = self.x.iloc[self.labels == c2_label, :]

    # Calculate means for for c1 and c2 in x space
    def set_cluster_means(self):
        self.x_c1_mean = self.x_c1.mean()
        self.x_c2_mean = self.x_c2.mean()

    # Calculate within-cluster scatter in x space
    def set_x_scatter_within(self):
        c1_diff = pd.DataFrame(self.x_c1 - self.x_c1_mean).transpose()
        c2_diff = pd.DataFrame(self.x_c2 - self.x_c2_mean).transpose()
        self.x_scatter_within = pd.DataFrame(np.dot(c1_diff, c1_diff.transpose())
                                             + np.dot(c2_diff, c2_diff.transpose()))

    # Calculate between-cluster scatter in x space
    def set_x_scatter_between(self):
        diff = self.x_c1_mean - self.x_c2_mean
        self.x_scatter_between = np.dot(diff, diff.transpose())

    # Find W by maximizing the Fisher criterion -- J(W)
    def set_w(self):
        inv_scatter = np.linalg.inv(self.x_scatter_within)
        mean_diff = self.x_c1_mean - self.x_c2_mean
        self.w = pd.DataFrame(np.dot(inv_scatter, mean_diff))

    def project_clusters_onto_w(self):
        self.y_c1 = np.dot(self.x_c1, self.w)
        self.y_c2 = np.dot(self.x_c2, self.w)

    def do_skl_lda(self, c1_label, c2_label):
        self.skl = LinearDiscriminantAnalysis()
        x = np.concatenate((self.x_c1, self.x_c2), axis=0)
        self.skl_labels = np.concatenate((self.labels[self.labels == c1_label], self.labels[self.labels == c2_label]), axis=0)
        labels = self.skl_labels
        self.skl.fit(x, labels)
        self.skl_y = pd.DataFrame(self.skl.fit_transform(x, labels))

    @staticmethod
    def plot_labeled(x, y, labels, target_labels, colors, title):
        fig, ax = plt.subplots(figsize=(10, 6))
        for color, label in zip(colors, target_labels):
            x_subset = x[labels == label]
            y_subset = y[labels == label]
            ax.scatter(x_subset, y_subset, color=color, alpha=0.8, lw=2, label=label)
        plt.title(title)
        fig.show()
        return fig, ax

    @staticmethod
    def add_line(f, ax, initial, final, title):
        ax.plot([initial[0], final[0]], [initial[1], final[1]], linestyle='dashed')
        ax.set_title(title)
        f.show()

    def fit_transform(self, c1_label, c2_label):
        self.define_clusters(c1_label, c2_label)
        self.set_cluster_means()
        self.set_x_scatter_within()
        self.set_x_scatter_between()
        self.set_w()
        self.project_clusters_onto_w()

if __name__ == '__main__':
    # 1
    print('Testing the LDA class.')
    iris = datasets.load_iris()
    lda = LDA(labels=iris.target, data=iris.data)
    lda.x.columns = iris['feature_names']
    lda.head_x()
    lda.label_summary()
    print('The above labels correspond to %s' %iris['target_names'])
    lda.define_clusters(c1_label=0, c2_label=2)
    lda.set_cluster_means()
    print('C1 mean: \n%s' % lda.x_c1_mean)
    print('C2 mean: \n%s' % lda.x_c2_mean)
    lda.set_x_scatter_within()
    print('Scatter within (S1 + S2) shape: %s' % str(lda.x_scatter_within.shape))
    lda.set_x_scatter_between()
    print('Scatter between: %s' % str(lda.x_scatter_between))
    lda.set_w()
    print('W: %s' % lda.w)
    lda.project_clusters_onto_w()
    print('X projected onto W (Y): %s' % pd.DataFrame(np.concatenate((lda.y_c1, lda.y_c2), axis=0)))
    lda.do_skl_lda(c1_label=0, c2_label=2)
    print('SKL Y: %s' % lda.skl_y)
    # Compare methods. Plot projections of clusters onto W for this class and scikit-learn LDA
    projected_clusters = np.concatenate((lda.y_c1, lda.y_c2), axis=0)
    lda.plot_labeled(x=projected_clusters, y=np.repeat(0, len(projected_clusters)),
                     labels=np.concatenate((lda.labels[lda.labels == 0], lda.labels[lda.labels == 2]), axis=0),
                     target_labels=[0, 2], colors=["red", "navy"], title='This Class. Projection of Clusters onto W.')
    lda.plot_labeled(x=lda.skl_y, y=np.repeat(0, len(lda.skl_y)), labels=lda.skl_labels, target_labels=[0, 2],
                     colors=["brown", "orange"], title='Scikit-Learn LDA. Projection of Clusters onto W.')
    print('The above plots show that this LDA class produces results very similar to those produced '
          'when using scikit-learn LDA.')
    # 2
    lda2 = LDA(data_file_path='./data/lda_dataset_1.csv')
    print('MC LDA x')
    # Remove label column from data
    dataset2_labels = lda2.x.iloc[:, -1]
    lda2.x = lda2.x.iloc[:, :-1]
    # Mean center data
    lda2.x = lda2.x - np.mean(lda2.x)
    print(lda2.x.head(5))
    f1, ax1 = lda2.plot_labeled(x=lda2.x['V1'], y=lda2.x['V2'], labels=dataset2_labels,
                                target_labels=[0, 1], colors=['blue', 'orange'], title='V1 vs V2')
    p = pca.PCA('./data/lda_dataset_1.csv')
    # Remove label column from data
    p.x = p.x.iloc[:, :-1]
    p.fit_transform()  # mean centers p.x
    print('MC PCA x')
    print(p.x.head(5))
    pc1_proj = p.project_onto_pc(pc_number=1)
    # Plot projection of data onto PC1
    lda2.plot_labeled(x=pc1_proj, y=np.repeat(0, np.shape(pc1_proj)[0]), labels=dataset2_labels,
                      target_labels=[0, 1], colors=['pink', 'purple'], title='Data projected onto PC1')
    # Plot PC1 axis - blue dashed
    pc1_x_min, pc1_x_max = [np.dot(np.max(p.projected_x.iloc[:, 1]), p.x_cov_eigenvectors.iloc[:, 1].transpose()),
                            np.dot(np.min(p.projected_x.iloc[:, 1]), p.x_cov_eigenvectors.iloc[:, 1].transpose())]
    # Scale line to revert x shift during mean-centering
    i = pc1_x_max * (1 / pc1_x_max[0])
    f = pc1_x_max * (np.max(lda2.x.iloc[:, 0]) / pc1_x_max[0])
    # Compare to least squares regression line - purple solid
    a, b = np.polyfit(lda2.x.iloc[:, 0], lda2.x.iloc[:, 1], 1)
    lda2.add_line(f=f1, ax=ax1, initial=pc1_x_min, final=pc1_x_max, title='Add PC1 axis')
    ax1.plot(lda2.x.iloc[:, 0], a*lda2.x.iloc[:, 0]+b, color='purple')
    ax1.set_title('Add Least Squares Regression line')
    f1.show()
    # Perform LDA
    lda2.fit_transform(c1_label=0, c2_label=1)
    print('W: \n%s' % lda2.w)
    # Clear separation of data is present
    projected_clusters = pd.DataFrame(np.concatenate((lda2.y_c1, lda2.y_c2), axis=0))
    lda2.plot_labeled(x=projected_clusters, y=np.repeat(0, len(projected_clusters), axis=0), labels=lda2.labels,
                      target_labels=[0, 1], colors=["green", "blue"], title='This Class. Projection of Clusters onto W.')
    w_x_min, w_x_max = [np.dot(np.max(projected_clusters), lda2.w.transpose()),
                        np.dot(np.min(projected_clusters), lda2.w.transpose())]
    lda2.add_line(f=f1, ax=ax1, initial=w_x_min*150, final=w_x_max*150, title='Add this class\'s W axis.')
    # Doesn't look perpendicular to PC1 axis...
    # Compare with scikit-learn LDA
    lda2.do_skl_lda(c1_label=0, c2_label=1)
    w_x_min, w_x_max = [np.dot(np.max(lda.skl_y)[0], lda2.skl.coef_[0]),
                        np.dot(np.min(lda.skl_y)[0], lda2.skl.coef_[0])]
    lda2.add_line(f=f1, ax=ax1, initial=w_x_min/30, final=w_x_max/30, title='Add Scikit-Learn\'s W axis.')
    # Scikit-learn's W doesn't look perpenicular to PC1 either...
    print('The W determined by this class and the scikit-learn W closely match')
    print('Variance of clusters when projected onto W: %s' % np.var(projected_clusters)[0])
    print('Variance of cluster 1 when projected onto W: %s' % np.var(lda2.y_c1))
    print('Variance of cluster 2 when projected onto W: %s' % np.var(lda2.y_c2))
    print('LDA does a good job of preserving variance that contributes to class separation,'
          'while PCA does a good job of preserving variance that along the axis that best '
          'decorrelates the data. W is nearly perpendicular to PC1.')
