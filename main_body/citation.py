import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import pandas as pd
import pickle
import seaborn as sns

def main():

    #Comment data
    pickle_in = open("../data/pickles/information_scores.pickle","rb")
    comments = pickle.load(pickle_in)

    #Loop through comment campaign thresholds
    thresholds = [2, 10, 1000]

    #Weight up comment dataset
    comments_wtd = comments.reindex(comments.index.repeat(comments['sweight']))
    comments_wtd.reset_index(inplace = True)

    #plot distribution of score by duplication threshold
    max_dup = comments_wtd.score_dup_count.max()
    comments_wtd['dup_cat'] = pd.cut(comments_wtd['score_dup_count'],
    bins = [0, 1, 9, 999, max_dup])
    fig, ax1 = plt.subplots(nrows = 1, ncols = 1)
    vgraph = sns.violinplot(data = comments_wtd, x = 'dup_cat',
    y = 'log_raw_score', ax = ax1)
    ax1.set_xticklabels(["None", "2-9", "10-999", "1000 or more"])
    ax1.set_xlabel("Comment Duplication")
    ax1.set_ylabel("Log(Information Score)")
    fig.savefig('../output/figures/information_score_by_duplication.jpg',
    bbox_inches='tight', dpi = 200)

    for this_threshold in thresholds:

        #Set comment threshold
        comments_wtd['comment_campaign'] = 0
        comments_wtd.loc[(comments_wtd.sweight >= this_threshold),
        'comment_campaign'] = 1
        comments_wtd.loc[((comments_wtd.sweight < this_threshold) &
        (comments_wtd.score_dup_count >= this_threshold)),
        'comment_campaign'] = 2

        #Create interest group categorization
        comments_wtd['group_cat'] = ''
        comments_wtd.loc[((comments_wtd['interest_group'] == 0) & \
        (comments_wtd['comment_campaign'] == 1)),
        'group_cat'] = 'Exact Duplicate'
        comments_wtd.loc[((comments_wtd['interest_group'] == 0) & \
        (comments_wtd['comment_campaign'] == 2)),
        'group_cat'] = 'Near Duplicate'
        comments_wtd.loc[((comments_wtd['interest_group'] == 0) & \
        (comments_wtd['comment_campaign'] == 0)), 'group_cat'] = 'Individual'
        comments_wtd.loc[((comments_wtd['interest_group'] == 1) & \
        (comments_wtd['business_group'] == 0)),'group_cat'] = 'Non-Business'
        comments_wtd.loc[((comments_wtd['interest_group'] == 1) & \
        (comments_wtd['business_group'] == 1)),'group_cat'] = 'Business'

        #Compare citation rate across different types of filers
        count_table = comments_wtd.groupby('group_cat').count()[['doc_id']]
        cite_table = comments_wtd.groupby('group_cat').sum()[['cited']]
        count_table.reset_index(inplace = True)
        table1 = count_table.merge(cite_table, on = 'group_cat')
        table1['citation_rate'] = np.round((table1['cited']/
        table1['doc_id'])*100, 3)

        table1.to_csv('../output/tables/citation_table_' + str(this_threshold) \
        + '.csv', index = False)

        if this_threshold == 10:
            #In-person meetings (use unweighted distribution)
            in_person = comments.groupby(['interest_group',
            'business_group']).sum()[['in_person']].reset_index()
            in_person['group_cat'] = ['Individual', 'Non-Business',
            'Business']
            in_person_table = in_person.merge(table1, on = 'group_cat')
            in_person_table['in_person_rate'] = (in_person_table['in_person'] /\
            in_person_table['doc_id'])*100

            in_person_table[['group_cat', 'doc_id', 'in_person_rate']].\
            to_csv('../output/tables/in_person_table_' + \
            str(this_threshold) + '.csv', index = False)

        #Bin info score
        comments_wtd['score_bin'] = pd.cut(comments_wtd['log_raw_score'],
        bins = [0, 1, 2, 3, 4, 5, 6, 8], include_lowest = True)

        #Group exact and near duplicates into mass category
        comments_wtd.loc[(comments_wtd['group_cat'].isin(['Exact Duplicate',
        'Near Duplicate'])), 'group_cat'] = 'Mass'

        #Plot data
        map_table = comments_wtd.groupby(['group_cat', \
        'score_bin']).mean()[['cited']].reset_index()
        map_pivot = map_table.pivot(index='group_cat', columns='score_bin',
        values='cited')
        map_pivot = map_pivot.fillna(0)
        map_pivot = map_pivot.reindex(['Business', 'Non-Business',
        'Individual', 'Mass'])
        this_cmap = sns.color_palette("mako", as_cmap=True)

        #Boxplot
        fig, axarr = plt.subplots(ncols = 2, gridspec_kw={'width_ratios':
        [1, 1.2]})
        fig.set_size_inches(15, 5)
        fig.subplots_adjust(wspace=0.1)
        plot1 = sns.boxplot(data=comments_wtd, x="group_cat", y="log_raw_score",
        ax = axarr[0], palette="Blues",
        order = ['Mass', 'Individual', 'Non-Business', 'Business'])
        axarr[0].set_ylabel("Log(Information Score)")
        axarr[0].set_xlabel("Commenter Type")
        axarr[0].set_title("Policy Content")
        hm = sns.heatmap(map_pivot, ax = axarr[1],
        cbar_kws = dict(label = "P(Cited)"),
        cmap = this_cmap)
        axarr[1].set_xlabel("Log(Information Score)")
        axarr[1].set_ylabel("Commenter Type")
        axarr[1].set_title("Probability of Citation")
        for tick in axarr[1].get_yticklabels():
            tick.set_verticalalignment("center")

        fig.savefig('../output/figures/information_boxplots_' + \
        str(this_threshold) + '.jpg', bbox_inches='tight', dpi = 200)

if __name__ == '__main__':
    main()
