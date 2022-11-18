from collections import defaultdict
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from math import exp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

def make_plot(vector_data, group, axis, title, show_legend = False,
legend_anchor = (3, -.2), this_map = 'cividis', nclust = 5):

    #Subset to plot data
    plot_data = vector_data.loc[(vector_data.group_cat == group),]

    #Marker styles
    marker_dict = {0.0: 'o', 1.0: 'X'}

    #Continuous colormap
    norm = plt.Normalize(0, 1)
    sm = plt.cm.ScalarMappable(cmap = this_map, norm = norm)
    sm.set_array([])

    #color palette
    my_palette = dict({0.0 : "#2d004b", 1.0 : "#fdae61"})

    #Main plot
    my_plot = sns.scatterplot(data = plot_data,
    x = 'd1', y = 'd2', hue = 'cited', ax = axis, alpha = 0.8,
    palette = my_palette, style = 'cited', markers = marker_dict)

    if show_legend:
        leg = my_plot.legend(loc = 'lower right',
        bbox_to_anchor=legend_anchor,
        ncol = 3, frameon = False)
        # replace labels
        new_labels = ['No', 'Yes']
        for t, l in zip(leg.texts, new_labels): t.set_text(l)
        leg.set_title('Cited')
    else:
        #Remove legend
        axis.get_legend().remove()

    axis.set_title(title, y = 0.95, fontsize = 16, pad=30)

    #Axis points
    ref_points = []
    for c in range(nclust):
        coords = vector_data.sort_values('c_' + str(c), \
        ascending = False).iloc[0][['d1', 'd2']]
        ref_points.append((c, coords[0], coords[1]))

    ref_points = pd.DataFrame(ref_points, columns = ['cluster', 'd1', 'd2'])
    ref_points['slope'] = ref_points['d2']/ref_points['d1']
    ref_points['label'] = ['Economic', 'Consumer', 'Technical',
    'Regulatory', 'Legal']
    ref_points['ylabel'] = ref_points['d2']
    ref_points['xlabel'] = ref_points['ylabel']/ref_points['slope']

    ylims = axis.get_ylim()
    xlims = axis.get_xlim()
    for c in range(nclust):
        xvals = np.array(xlims)
        yvals = np.array(ylims)
        if ref_points['d1'][c] > 0:
            xvals[0] = yvals[0] = 0
        else:
            xvals[1] = yvals[1] = 0
        yvals = ref_points['slope'][c] * xvals
        axis.plot(xvals, yvals, color = "black", linestyle = ':')


        #Add labels
        axis.text(ref_points['xlabel'][c],
        ref_points['ylabel'][c],
        ref_points['label'][c], fontsize = 12)

    #Fix plot
    axis.set_ylim(ylims)
    axis.set_xlim(xlims)
    axis.axis('off')

def main():

    #read in document vector data
    comment_vectors = pd.read_csv('../data/csvs/topic_vectors.csv')

    #read in words
    pickle_in = open('../data/pickles/topic_words.pickle','rb')
    word_dict = pickle.load(pickle_in)
    ['Economic', 'Consumer', 'Technical',
    'Regulatory', 'Legal']

    #Reduce dimensionalty for visualization
    tsne_2d = TSNE(perplexity = 40, n_components = 2, random_state = 123,
    init = 'pca')
    vectors_2d = np.array(tsne_2d.fit_transform(comment_vectors[['c_0', 'c_1',
    'c_2', 'c_3', 'c_4']]))
    comment_vectors['d1'] = vectors_2d[:,0]
    comment_vectors['d2'] = vectors_2d[:,1] * -1 #reflect for better visual

    #Create interest group category
    comment_vectors['group_cat'] = 'Individual'
    comment_vectors.loc[((comment_vectors['interest_group'] == 1) & \
    (comment_vectors['business_group'] == 0)),'group_cat'] = 'Non-Business'
    comment_vectors.loc[((comment_vectors['interest_group'] == 1) & \
    (comment_vectors['business_group'] == 1)),'group_cat'] = 'Business'

    #Plot
    fig, axarr = plt.subplots(ncols = 3)
    fig.set_size_inches(18, 6)
    fig.suptitle("Comment pages colored by agency citation", fontsize=16)
    fig.subplots_adjust(top=0.8)
    plot_titles = ['Individual', 'Business', 'Non-Business']
    for i in [0,1,2]:
        if i == 1:
            this_legend = True
        else:
            this_legend = False

        make_plot(vector_data = comment_vectors, group = plot_titles[i],
        axis = axarr[i], title = plot_titles[i],
        show_legend = this_legend, legend_anchor = (0.7, -.2))

    fig.savefig('../output/figures/comment_segmentation_citation.jpg',
    bbox_inches='tight')

if __name__ == '__main__':
    main()
